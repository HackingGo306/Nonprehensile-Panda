import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { ModelManifest, RendererState, SortDrop, VisualMeshAsset } from './types';

// MuJoCo uses an x-right, y-forward, z-up world.  Three.js is y-up, so keep
// every visual in the same physical frame with a -90 degree rotation around x.
const setMuJoCoVector = (target: THREE.Vector3, values: ArrayLike<number>, offset = 0) => target.set(values[offset], values[offset + 2], -values[offset + 1]);
const worldRotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const inverseWorldRotation = worldRotation.clone().invert();
const worldMatrix = new THREE.Matrix4().makeRotationFromQuaternion(worldRotation);
const sourceMatrix = new THREE.Matrix4();
const transformedMatrix = new THREE.Matrix4();
const nextFrame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));
const clamp = (value: number, low: number, high: number): number => Math.min(high, Math.max(low, value));

const setMuJoCoQuaternion = (target: THREE.Quaternion, values: ArrayLike<number>, offset: number): void => {
  target.set(values[offset + 4], values[offset + 5], values[offset + 6], values[offset + 3]);
  target.premultiply(worldRotation).multiply(inverseWorldRotation);
};
const setMuJoCoRotation = (target: THREE.Quaternion, quaternion: [number, number, number, number]): void => {
  target.set(quaternion[1], quaternion[2], quaternion[3], quaternion[0]);
  target.premultiply(worldRotation).multiply(inverseWorldRotation);
};

// Compiled mesh vertices stay in MuJoCo's local basis.  Apply B * R directly
// to each mesh instance, while positions are mapped into Three's world frame.
const setMuJoCoMeshRotation = (target: THREE.Quaternion, values: ArrayLike<number>, offset: number): void => {
  sourceMatrix.set(
    values[offset], values[offset + 1], values[offset + 2], 0,
    values[offset + 3], values[offset + 4], values[offset + 5], 0,
    values[offset + 6], values[offset + 7], values[offset + 8], 0,
    0, 0, 0, 1,
  );
  target.setFromRotationMatrix(transformedMatrix.copy(worldMatrix).multiply(sourceMatrix));
};

type VisualMesh = { geomId: number; mesh: THREE.Mesh; material: THREE.MeshStandardMaterial; color: THREE.Color };
type SuccessEffect = {
  group: THREE.Group;
  pulse: THREE.Mesh<THREE.TorusGeometry, THREE.MeshBasicMaterial>;
  sparks: Array<{ mesh: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>; direction: THREE.Vector3 }>;
  startedAt: number;
};
type FallingBox = {
  mesh: THREE.Mesh<THREE.BoxGeometry, THREE.MeshStandardMaterial>;
  start: THREE.Vector3;
  landingY: number;
  impactY: number;
  startedAt: number;
  duration: number;
  correct: boolean;
  targetBin: 0 | 1;
  impacted: boolean;
};

export class ConveyorRenderer {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(42, 1, 0.01, 20);
  private readonly renderer?: THREE.WebGLRenderer;
  private readonly orbitControls?: OrbitControls;
  private readonly fallbackContext?: CanvasRenderingContext2D | null;
  private readonly boxMeshes: THREE.Group[] = [];
  private readonly packageMaterials: THREE.MeshStandardMaterial[] = [];
  private readonly packageTextures: THREE.Texture[] = [];
  private readonly floorTextures: THREE.Texture[] = [];
  private floor?: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshStandardMaterial>;
  private readonly visualMeshes: VisualMesh[] = [];
  private readonly indicatorUntil = [0, 0];
  private readonly binMaterials: THREE.MeshStandardMaterial[];
  private readonly successEffects: SuccessEffect[] = [];
  private readonly fallingBoxes: FallingBox[] = [];
  private readonly meshReady: Promise<void>;
  private animationFrame = 0;
  private lastState: RendererState | undefined;
  private readonly resizeObserver?: ResizeObserver;
  private readonly onWindowResize = () => this.resize();

  constructor(private readonly canvas: HTMLCanvasElement, private readonly manifest: ModelManifest, meshAssets: VisualMeshAsset[]) {
    try {
      this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
      this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      this.renderer.setClearColor(0x10161d);
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    } catch {
      this.fallbackContext = canvas.getContext('2d');
      if (!this.fallbackContext) throw new Error('The browser could not create a WebGL or 2D canvas context. Enable hardware graphics and reload.');
    }
    this.camera.position.set(2.2, 1.65, 2.35);
    this.camera.lookAt(0.55, 0.28, 0);
    if (this.renderer) {
      this.orbitControls = new OrbitControls(this.camera, this.renderer.domElement);
      this.orbitControls.target.set(0.58, 0.22, 0);
      this.orbitControls.enableDamping = true;
      this.orbitControls.dampingFactor = 0.08;
      this.orbitControls.enablePan = true;
      this.orbitControls.panSpeed = 0.7;
      this.orbitControls.minDistance = 0.85;
      this.orbitControls.maxDistance = 6;
      this.orbitControls.minPolarAngle = 0.12;
      this.orbitControls.maxPolarAngle = Math.PI * 0.49;
      this.orbitControls.update();
    }
    this.scene.add(new THREE.HemisphereLight(0xc9deff, 0x101318, 1.65));
    const key = new THREE.DirectionalLight(0xf4f7ff, 2.7);
    key.position.set(-1.6, 3.5, 2.2); key.target.position.set(0.55, 0, 0); key.castShadow = Boolean(this.renderer);
    key.shadow.mapSize.set(1024, 1024); key.shadow.camera.left = -2.8; key.shadow.camera.right = 2.8; key.shadow.camera.top = 2.8; key.shadow.camera.bottom = -1.2; key.shadow.camera.near = 0.2; key.shadow.camera.far = 8; key.shadow.bias = -0.0003; key.shadow.normalBias = 0.018; key.shadow.radius = 2.2;
    this.scene.add(key, key.target);
    this.addFloor();
    this.addStaticScene();
    this.addWarehouseBackdrop();
    this.binMaterials = [new THREE.MeshStandardMaterial({ color: 0x19c766, emissive: 0x052414 }), new THREE.MeshStandardMaterial({ color: 0x2879f3, emissive: 0x071735 })];
    this.addBins();
    this.addBoxes();
    this.meshReady = this.addRobotMeshes(meshAssets);
    try {
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas.parentElement ?? canvas);
    } catch {
      window.addEventListener('resize', this.onWindowResize);
    }
    this.resize();
    this.renderLoop();
  }

  private addStaticScene(): void {
    const table = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.28, 0.5), new THREE.MeshStandardMaterial({ color: 0x3f4852, metalness: 0.35, roughness: 0.58 }));
    table.position.set(0.65, 0.14, 0); table.castShadow = true; table.receiveShadow = true; this.scene.add(table);
    const belt = new THREE.Mesh(new THREE.BoxGeometry(2, 0.036, 0.43), new THREE.MeshStandardMaterial({ color: 0x151b22, metalness: 0.25, roughness: 0.45 }));
    belt.position.set(0.65, 0.292, 0); belt.castShadow = true; belt.receiveShadow = true; this.scene.add(belt);
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.22, 0.1, 32), new THREE.MeshStandardMaterial({ color: 0xe6e8eb, metalness: 0.75, roughness: 0.3 }));
    base.position.set(0.25, 0.05, 0); base.castShadow = true; base.receiveShadow = true; this.scene.add(base);
  }

  private addFloor(): void {
    const textureLoader = new THREE.TextureLoader();
    const texturePath = (name: string) => `${import.meta.env.BASE_URL}textures/${name}`;
    const colorMap = textureLoader.load(texturePath('ambientcg-concrete024-color-1k.jpg'));
    const normalMap = textureLoader.load(texturePath('ambientcg-concrete024-normal-gl-1k.jpg'));
    const roughnessMap = textureLoader.load(texturePath('ambientcg-concrete024-roughness-1k.jpg'));
    colorMap.colorSpace = THREE.SRGBColorSpace;
    for (const texture of [colorMap, normalMap, roughnessMap]) {
      texture.wrapS = THREE.RepeatWrapping; texture.wrapT = THREE.RepeatWrapping; texture.repeat.set(1.8, 1.6);
      this.floorTextures.push(texture);
    }
    const material = new THREE.MeshStandardMaterial({ color: 0x7d858a, map: colorMap, normalMap, roughnessMap, normalScale: new THREE.Vector2(0.1, 0.1), roughness: 0.94, metalness: 0 });
    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(5.2, 4.6), material);
    this.floor.rotation.x = -Math.PI / 2; this.floor.position.set(0.65, -0.004, 0); this.floor.receiveShadow = true; this.scene.add(this.floor);
  }

  private addWarehouseBackdrop(): void {
    // Keep these props behind the workcell and deliberately low-saturation so
    // they establish the warehouse setting without competing with the robot.
    const backdrop = new THREE.Group();
    const wallMaterial = new THREE.MeshStandardMaterial({ color: 0x111c25, roughness: 0.88, metalness: 0.08 });
    const frameMaterial = new THREE.MeshStandardMaterial({ color: 0x354653, roughness: 0.44, metalness: 0.68 });
    const shelfMaterial = new THREE.MeshStandardMaterial({ color: 0x263540, roughness: 0.6, metalness: 0.45 });
    const cartonMaterial = new THREE.MeshStandardMaterial({ color: 0x8c714e, roughness: 0.82, metalness: 0 });
    const palletMaterial = new THREE.MeshStandardMaterial({ color: 0x5a4632, roughness: 0.92, metalness: 0 });
    const safetyMaterial = new THREE.MeshStandardMaterial({ color: 0x8f792b, roughness: 0.7, metalness: 0.15 });
    const wall = new THREE.Mesh(new THREE.BoxGeometry(3.7, 2.1, 0.06), wallMaterial);
    wall.position.set(0, 1.05, -1.72); backdrop.add(wall);

    const rack = new THREE.Group(); rack.position.set(0.75, 0, -1.38); backdrop.add(rack);
    for (const x of [-0.92, 0.92]) for (const z of [-0.24, 0.24]) {
      const post = new THREE.Mesh(new THREE.BoxGeometry(0.055, 1.58, 0.055), frameMaterial);
      post.position.set(x, 0.79, z); rack.add(post);
    }
    for (const height of [0.16, 0.65, 1.14]) {
      const shelf = new THREE.Mesh(new THREE.BoxGeometry(1.95, 0.055, 0.54), shelfMaterial);
      shelf.position.y = height; rack.add(shelf);
      const frontBeam = new THREE.Mesh(new THREE.BoxGeometry(1.98, 0.085, 0.045), frameMaterial);
      frontBeam.position.set(0, height - 0.015, 0.27); rack.add(frontBeam);
    }
    const storedCartons: Array<[number, number, number, number, number, number]> = [
      [-0.54, 0.36, -0.02, 0.42, 0.35, 0.32], [-0.04, 0.33, 0.02, 0.48, 0.29, 0.38], [0.55, 0.39, -0.04, 0.33, 0.43, 0.3],
      [-0.46, 0.82, 0.03, 0.5, 0.29, 0.33], [0.2, 0.85, -0.03, 0.45, 0.35, 0.36], [0.67, 0.78, 0.05, 0.24, 0.22, 0.26],
      [-0.55, 1.29, -0.02, 0.38, 0.24, 0.29], [-0.05, 1.31, 0.02, 0.43, 0.28, 0.31], [0.53, 1.25, -0.04, 0.48, 0.2, 0.28],
    ];
    storedCartons.forEach(([x, y, z, width, height, depth]) => {
      const carton = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), cartonMaterial);
      carton.position.set(x, y, z); rack.add(carton);
    });

    const pallet = new THREE.Group(); pallet.position.set(-0.78, 0, -0.93); backdrop.add(pallet);
    for (const z of [-0.18, 0, 0.18]) {
      const slat = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.035, 0.105), palletMaterial);
      slat.position.set(0, 0.07, z); pallet.add(slat);
    }
    for (const x of [-0.2, 0.2]) {
      const rail = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.07, 0.49), palletMaterial);
      rail.position.set(x, 0.03, 0); pallet.add(rail);
    }
    const palletLoad = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.3, 0.38), cartonMaterial);
    palletLoad.position.set(0, 0.24, 0); pallet.add(palletLoad);

    const safetyLine = new THREE.Mesh(new THREE.BoxGeometry(3.15, 0.001, 0.035), safetyMaterial);
    safetyLine.position.set(0.58, 0.004, -0.76); backdrop.add(safetyLine);
    
    // const fixtureMaterial = new THREE.MeshStandardMaterial({ color: 0xc7d9e6, emissive: 0x263c4d, emissiveIntensity: 0.55, roughness: 0.35, metalness: 0.25 });
    // const cableMaterial = new THREE.MeshStandardMaterial({ color: 0x1e2a33, roughness: 0.65, metalness: 0.55 });
    // for (const x of [-0.05, 1.35]) {
    //   const fixture = new THREE.Mesh(new THREE.BoxGeometry(0.86, 0.035, 0.12), fixtureMaterial);
    //   fixture.position.set(x, 1.86, -0.98); backdrop.add(fixture);
    //   const cable = new THREE.Mesh(new THREE.BoxGeometry(0.025, 0.5, 0.025), cableMaterial);
    //   cable.position.set(x, 1.61, -0.98); backdrop.add(cable);
    // }
    // const placard = new THREE.Group(); placard.position.set(-0.82, 1.38, -1.675); backdrop.add(placard);
    // const placardBack = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.27, 0.025), new THREE.MeshStandardMaterial({ color: 0x223b4b, roughness: 0.56, metalness: 0.38 })); placard.add(placardBack);
    // for (const [x, y, width, color] of [[-0.12, 0.06, 0.25, 0x8db4cc], [-0.05, 0, 0.38, 0x5f8196], [-0.13, -0.065, 0.22, 0x5f8196]] as const) {
    //   const stripe = new THREE.Mesh(new THREE.BoxGeometry(width, 0.026, 0.012), new THREE.MeshStandardMaterial({ color, roughness: 0.46, metalness: 0.25 }));
    //   stripe.position.set(x, y, 0.02); placard.add(stripe);
    // }
    this.enableShadows(backdrop);
    this.scene.add(backdrop);
  }

  private enableShadows(object: THREE.Object3D): void {
    object.traverse((child) => { if (child instanceof THREE.Mesh) { child.castShadow = true; child.receiveShadow = true; } });
  }

  private addBins(): void {
    // Keep the open, table-facing edge where the trained policy expects it,
    // then extend each bin to the far end of the conveyor.  Moving the bins
    // 14 mm outward puts their inner wall flush with (rather than through)
    // the table edge, while retaining 30 mm of clearance around the arm base.
    const specs = this.manifest.sorting_bin_regions.map((region) => ({
      x: (region.x_min + region.x_max) / 2,
      y: (region.y_min + region.y_max) / 2,
      color: region.target_bin,
      length: region.x_max - region.x_min,
      width: region.y_max - region.y_min,
    }));
    specs.forEach(({ x, y, color, length, width }) => {
      const group = new THREE.Group(); const material = this.binMaterials[color];
      // Rest the bin bottoms on the same Z=0 ground plane as the table.
      const bottom = new THREE.Mesh(new THREE.BoxGeometry(length, 0.035, width), material); bottom.position.y = 0.0175; group.add(bottom);
      for (const [dx, dy, sx, sy] of [[length / 2, 0, 0.018, width], [0, width / 2, length, 0.018], [0, -width / 2, length, 0.018]] as const) { const wall = new THREE.Mesh(new THREE.BoxGeometry(sx, 0.18, sy), material); wall.position.set(dx, 0.125, -dy); group.add(wall); }
      group.position.set(x, 0, -y); this.enableShadows(group); this.scene.add(group);
    });
  }

  private startSuccessEffect(position: THREE.Vector3, color: THREE.Color, startedAt: number): void {
    const group = new THREE.Group(); group.position.copy(position); this.scene.add(group);
    const pulseMaterial = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending });
    const pulse = new THREE.Mesh(new THREE.TorusGeometry(0.12, 0.012, 8, 32), pulseMaterial); pulse.rotation.x = Math.PI / 2; group.add(pulse);
    const sparkDirections = [[-0.14, -0.08], [-0.06, 0.12], [0.08, -0.1], [0.16, 0.07]];
    const sparks = sparkDirections.map(([sparkX, sparkZ]) => {
      const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.018, 10, 8), material); group.add(mesh);
      return { mesh, direction: new THREE.Vector3(sparkX, 0.16, sparkZ) };
    });
    this.successEffects.push({ group, pulse, sparks, startedAt });
  }

  private updateSuccessEffects(now: number): void {
    for (let index = this.successEffects.length - 1; index >= 0; index -= 1) {
      const effect = this.successEffects[index];
      const progress = (now - effect.startedAt) / 720;
      if (progress < 0) continue;
      if (progress >= 1) { this.disposeSuccessEffect(effect); this.successEffects.splice(index, 1); continue; }
      const easeOut = 1 - (1 - progress) ** 3;
      const fade = (1 - progress) ** 1.7;
      effect.pulse.scale.setScalar(0.55 + easeOut * 2.3);
      effect.pulse.material.opacity = 0.95 * fade;
      effect.sparks.forEach(({ mesh, direction }) => {
        mesh.position.copy(direction).multiplyScalar(easeOut);
        mesh.position.y += 0.02 + 0.07 * Math.sin(progress * Math.PI);
        mesh.scale.setScalar(1 - progress * 0.4);
        mesh.material.opacity = 0.9 * fade;
      });
    }
  }

  private startFallingBox(drop: SortDrop, startedAt: number): void {
    const half = this.manifest.box_half_extents[drop.boxIndex];
    const material = new THREE.MeshStandardMaterial({ color: drop.targetBin === 0 ? 0x1bd66c : 0x3485ff, roughness: 0.48 });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(half[0] * 2, half[2] * 2, half[1] * 2), material);
    setMuJoCoVector(mesh.position, drop.position); setMuJoCoRotation(mesh.quaternion, drop.quaternion); mesh.castShadow = true; mesh.receiveShadow = true; this.scene.add(mesh);
    const landingY = 0.035 + half[2];
    this.fallingBoxes.push({ mesh, start: mesh.position.clone(), landingY, impactY: 0.038, startedAt, duration: clamp(Math.sqrt(2 * Math.max(mesh.position.y - landingY, 0) / 9.81), 0.18, 0.46), correct: drop.correct, targetBin: drop.targetBin, impacted: false });
  }

  private updateFallingBoxes(now: number): void {
    for (let index = this.fallingBoxes.length - 1; index >= 0; index -= 1) {
      const falling = this.fallingBoxes[index]; const progress = (now - falling.startedAt) / falling.duration;
      if (progress < 1) {
        const distance = falling.landingY - falling.start.y;
        falling.mesh.position.y = falling.start.y + distance * progress * progress;
        continue;
      }
      falling.mesh.position.y = falling.landingY;
      if (!falling.impacted) {
        falling.impacted = true;
        if (falling.correct) {
          this.indicatorUntil[falling.targetBin] = now + 750;
          this.startSuccessEffect(new THREE.Vector3(falling.mesh.position.x, falling.impactY, falling.mesh.position.z), this.binMaterials[falling.targetBin].color, now);
        }
      }
      const settle = (progress - 1) / 0.24;
      falling.mesh.material.transparent = true;
      falling.mesh.material.opacity = Math.max(0, 1 - settle);
      if (settle >= 1) { this.disposeFallingBox(falling); this.fallingBoxes.splice(index, 1); }
    }
  }

  private disposeSuccessEffect(effect: SuccessEffect): void {
    this.scene.remove(effect.group); effect.pulse.geometry.dispose(); effect.pulse.material.dispose(); effect.sparks.forEach(({ mesh }) => { mesh.geometry.dispose(); mesh.material.dispose(); });
  }

  private disposeFallingBox(falling: FallingBox): void {
    this.scene.remove(falling.mesh); falling.mesh.geometry.dispose(); falling.mesh.material.dispose();
  }

  private addBoxes(): void {
    const textureLoader = new THREE.TextureLoader();
    const texturePath = (name: string) => `${import.meta.env.BASE_URL}textures/${name}`;
    const colorMap = textureLoader.load(texturePath('ambientcg-cardboard001-color-1k.jpg'));
    const normalMap = textureLoader.load(texturePath('ambientcg-cardboard001-normal-gl-1k.jpg'));
    const roughnessMap = textureLoader.load(texturePath('ambientcg-cardboard001-roughness-1k.jpg'));
    colorMap.colorSpace = THREE.SRGBColorSpace;
    for (const texture of [colorMap, normalMap, roughnessMap]) {
      texture.wrapS = THREE.RepeatWrapping; texture.wrapT = THREE.RepeatWrapping; texture.repeat.set(2, 2);
      this.packageTextures.push(texture);
    }
    const tapeColors = [0x1bd66c, 0x3485ff];
    this.manifest.box_half_extents.forEach((half, index) => {
      const [width, height, depth] = [half[0] * 2, half[2] * 2, half[1] * 2];
      const packageMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, map: colorMap, normalMap, roughnessMap, normalScale: new THREE.Vector2(0.35, 0.35), roughness: 0.9, metalness: 0 });
      const tapeMaterial = new THREE.MeshStandardMaterial({ color: tapeColors[this.manifest.target_bins[index]], roughness: 0.46, metalness: 0.02 });
      this.packageMaterials.push(packageMaterial, tapeMaterial);
      const group = new THREE.Group();
      group.add(new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), packageMaterial));
      const tapeWidth = Math.max(0.012, Math.min(width, depth) * 0.24); const tapeThickness = 0.0018;
      const topLongitudinal = new THREE.Mesh(new THREE.BoxGeometry(width * 0.98, tapeThickness, tapeWidth), tapeMaterial);
      topLongitudinal.position.y = height / 2 + tapeThickness / 2; group.add(topLongitudinal);
      const topCross = new THREE.Mesh(new THREE.BoxGeometry(tapeWidth, tapeThickness, depth * 0.98), tapeMaterial);
      topCross.position.y = height / 2 + tapeThickness * 1.5; group.add(topCross);
      group.visible = false; this.enableShadows(group); this.scene.add(group); this.boxMeshes.push(group);
    });
  }

  private addVisualMesh(asset: VisualMeshAsset): number {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(asset.positions, 3));
    if (asset.normals) geometry.setAttribute('normal', new THREE.BufferAttribute(asset.normals, 3));
    else geometry.computeVertexNormals();
    geometry.setIndex(new THREE.BufferAttribute(asset.indices, 1));
    const color = new THREE.Color(asset.rgba[0], asset.rgba[1], asset.rgba[2]);
    const material = new THREE.MeshStandardMaterial({ color, metalness: 0.45, roughness: 0.32, transparent: asset.rgba[3] < 1, opacity: asset.rgba[3] });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false; mesh.castShadow = true; mesh.receiveShadow = true;
    this.scene.add(mesh);
    this.visualMeshes.push({ geomId: asset.geomId, mesh, material, color: color.clone() });
    return asset.positions.length / 3;
  }

  private async addRobotMeshes(assets: VisualMeshAsset[]): Promise<void> {
    try {
      // Yield before and during buffer adoption so the loading layer stays animated.
      await nextFrame();
      let verticesSinceYield = 0;
      for (const asset of assets) {
        verticesSinceYield += this.addVisualMesh(asset);
        if (verticesSinceYield >= 50_000) { verticesSinceYield = 0; await nextFrame(); }
      }
      if (this.lastState) this.updateRobot(this.lastState);
    } catch {
      this.visualMeshes.forEach(({ mesh, material }) => { this.scene.remove(mesh); mesh.geometry.dispose(); material.dispose(); });
      this.visualMeshes.length = 0;
    }
  }

  private updateRobot(state: RendererState): void {
    this.visualMeshes.forEach(({ geomId, mesh, material, color }) => {
      setMuJoCoVector(mesh.position, state.geomXpos, geomId * 3);
      setMuJoCoMeshRotation(mesh.quaternion, state.geomXmat, geomId * 9);
      material.color.copy(color);
    });
  }

  update(state: RendererState, dropEvents: SortDrop[] = []): void {
    this.lastState = state;
    this.boxMeshes.forEach((mesh, index) => { const address = this.manifest.box_qpos_addresses[index]; mesh.visible = state.qpos[address] > -1.5; setMuJoCoVector(mesh.position, state.qpos, address); setMuJoCoQuaternion(mesh.quaternion, state.qpos, address); });
    this.updateRobot(state);
    const now = performance.now(); dropEvents.forEach((drop) => this.startFallingBox(drop, now));
    this.binMaterials.forEach((material, index) => material.emissiveIntensity = now < this.indicatorUntil[index] ? 1.8 : 0.35);
    this.updateFallingBoxes(now);
    this.updateSuccessEffects(now);
  }

  whenMeshSceneReady(): Promise<void> { return this.meshReady; }

  private resize(): void {
    const width = Math.max(1, this.canvas.clientWidth), height = Math.max(1, this.canvas.clientHeight);
    if (this.renderer) this.renderer.setSize(width, height, false);
    else if (this.fallbackContext) {
      const pixelRatio = Math.min(devicePixelRatio, 2);
      this.canvas.width = Math.round(width * pixelRatio); this.canvas.height = Math.round(height * pixelRatio);
      this.fallbackContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    }
    this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
  }

  private drawFallback(): void {
    const context = this.fallbackContext, state = this.lastState;
    if (!context || !state) return;
    const width = this.canvas.clientWidth, height = this.canvas.clientHeight, pad = 24;
    const map = (x: number, y: number): [number, number] => [pad + ((x + 0.45) / 2.4) * (width - pad * 2), height - pad - ((y + 0.72) / 1.44) * (height - pad * 2)];
    const rect = (x: number, y: number, w: number, h: number, color: string): void => { const [left, bottom] = map(x - w / 2, y - h / 2), [right, top] = map(x + w / 2, y + h / 2); context.fillStyle = color; context.fillRect(left, top, right - left, bottom - top); };
    context.fillStyle = '#242b30'; context.fillRect(0, 0, width, height);
    rect(0.7, 0, 2.0, 0.43, '#18212a'); rect(1.085, 0.404, 1.15, 0.28, '#126c43'); rect(0.835, -0.404, 1.65, 0.28, '#164a92');
    const now = performance.now(); this.indicatorUntil.forEach((until, index) => {
      if (now >= until) return;
      context.globalAlpha = (until - now) / 750 * 0.48;
      rect(index === 0 ? 1.085 : 0.835, index === 0 ? 0.404 : -0.404, index === 0 ? 1.15 : 1.65, 0.28, '#ffffff');
      context.globalAlpha = 1;
    });
    this.manifest.box_qpos_addresses.forEach((address, index) => { if (state.qpos[address] <= -1.5) return; const half = this.manifest.box_half_extents[index]; rect(state.qpos[address], state.qpos[address + 1], half[0] * 2, half[1] * 2, this.manifest.target_bins[index] === 0 ? '#32d879' : '#458fff'); });
    context.fillStyle = '#a7b7c6'; context.font = '12px system-ui'; context.fillText('2D state view — WebGL unavailable', 14, 20);
  }

  private renderLoop = (): void => {
    const now = performance.now(); this.updateFallingBoxes(now); this.updateSuccessEffects(now);
    this.binMaterials.forEach((material, index) => material.emissiveIntensity = now < this.indicatorUntil[index] ? 1.8 : 0.35);
    this.orbitControls?.update(); if (this.renderer) this.renderer.render(this.scene, this.camera); else this.drawFallback(); this.animationFrame = requestAnimationFrame(this.renderLoop);
  };
  dispose(): void {
    cancelAnimationFrame(this.animationFrame); this.resizeObserver?.disconnect(); window.removeEventListener('resize', this.onWindowResize);
    this.orbitControls?.dispose();
    this.visualMeshes.forEach(({ mesh, material }) => { mesh.geometry.dispose(); material.dispose(); });
    this.successEffects.forEach((effect) => this.disposeSuccessEffect(effect));
    this.fallingBoxes.forEach((falling) => this.disposeFallingBox(falling));
    this.boxMeshes.forEach((group) => { this.scene.remove(group); group.traverse((object) => { if (object instanceof THREE.Mesh) object.geometry.dispose(); }); });
    this.packageMaterials.forEach((material) => material.dispose()); this.packageTextures.forEach((texture) => texture.dispose());
    this.floor?.geometry.dispose(); this.floor?.material.dispose(); this.floorTextures.forEach((texture) => texture.dispose());
    this.renderer?.dispose();
  }
}
