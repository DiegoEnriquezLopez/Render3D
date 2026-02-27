
/// <reference types="@webgpu/types" />
import "./style.css";
import shaderCode from "./shader.wgsl?raw";
import { Camera } from "./camera";
import { mat4 } from "./math";

if (!navigator.gpu) {
  throw new Error("WebGPU not supported");
}

const canvas = document.querySelector("#gfx-main") as HTMLCanvasElement;
if (!canvas) throw new Error("Canvas #gfx-main not found");

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("No GPU adapter found");

const device = await adapter.requestDevice();

const context = canvas.getContext("webgpu");
if (!context) throw new Error("Could not get WebGPU context");

const format = navigator.gpu.getPreferredCanvasFormat();
let depthTexture: GPUTexture | null = null;

function resize() {
  canvas.width = Math.max(1, Math.floor(window.innerWidth * devicePixelRatio));
  canvas.height = Math.max(1, Math.floor(window.innerHeight * devicePixelRatio));

  context.configure({
    device,
    format,
    alphaMode: "premultiplied",
  });

  depthTexture?.destroy();
  depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
  });
}
resize();
window.addEventListener("resize", resize);

/**
 * Vertex format: [x, y, z, u, v]
 * 6 faces × 2 triangles × 3 vertices = 36 vertices
 */
const vertices = new Float32Array([
  // Front (+Z)
  -1, -1,  1, 0, 1,
   1, -1,  1, 1, 1,
   1,  1,  1, 1, 0,
  -1, -1,  1, 0, 1,
   1,  1,  1, 1, 0,
  -1,  1,  1, 0, 0,

  // Back (-Z)
   1, -1, -1, 0, 1,
  -1, -1, -1, 1, 1,
  -1,  1, -1, 1, 0,
   1, -1, -1, 0, 1,
  -1,  1, -1, 1, 0,
   1,  1, -1, 0, 0,

  // Left (-X)
  -1, -1, -1, 0, 1,
  -1, -1,  1, 1, 1,
  -1,  1,  1, 1, 0,
  -1, -1, -1, 0, 1,
  -1,  1,  1, 1, 0,
  -1,  1, -1, 0, 0,

  // Right (+X)
   1, -1,  1, 0, 1,
   1, -1, -1, 1, 1,
   1,  1, -1, 1, 0,
   1, -1,  1, 0, 1,
   1,  1, -1, 1, 0,
   1,  1,  1, 0, 0,

  // Top (+Y)
  -1,  1,  1, 0, 1,
   1,  1,  1, 1, 1,
   1,  1, -1, 1, 0,
  -1,  1,  1, 0, 1,
   1,  1, -1, 1, 0,
  -1,  1, -1, 0, 0,

  // Bottom (-Y)
  -1, -1, -1, 0, 1,
   1, -1, -1, 1, 1,
   1, -1,  1, 1, 0,
  -1, -1, -1, 0, 1,
   1, -1,  1, 1, 0,
  -1, -1,  1, 0, 0,
]);

const vertexCount = vertices.length / 5;

const vertexBuffer = device.createBuffer({
  size: vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);

const uniformBuffer = device.createBuffer({
  size: 64, // mat4x4<f32>
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const shader = device.createShaderModule({
  label: "Textured Cube Shader",
  code: shaderCode,
});

const pipeline = device.createRenderPipeline({
  label: "Textured Cube Pipeline",
  layout: "auto",
  vertex: {
    module: shader,
    entryPoint: "vs_main",
    buffers: [
      {
        arrayStride: 5 * 4, // x,y,z,u,v
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },     // position
          { shaderLocation: 1, offset: 3 * 4, format: "float32x2" }, // uv
        ],
      },
    ],
  },
  fragment: {
    module: shader,
    entryPoint: "fs_main",
    targets: [{ format }],
  },
  primitive: {
    topology: "triangle-list",
    cullMode: "back",
  },
  depthStencil: {
    format: "depth24plus",
    depthWriteEnabled: true,
    depthCompare: "less",
  },
});

async function loadTextureOrCheckerboard(url: string): Promise<GPUTexture> {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);

    const blob = await res.blob();
    const image = await createImageBitmap(blob, { colorSpaceConversion: "none" });

    const tex = device.createTexture({
      size: [image.width, image.height, 1],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT, // safe, avoids Dawn validation issues
    });


    device.queue.copyExternalImageToTexture(
      { source: image },
      { texture: tex },
      [image.width, image.height]
    );

    return tex;
  } catch (err) {
    console.warn("Texture load failed, using checkerboard fallback:", err);

    const w = 128;
    const h = 128;
    const data = new Uint8Array(w * h * 4);

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const checker = ((x >> 4) & 1) ^ ((y >> 4) & 1);
        const c = checker ? 230 : 35;
        data[i + 0] = c;
        data[i + 1] = c;
        data[i + 2] = c;
        data[i + 3] = 255;
      }
    }

    const tex = device.createTexture({
      size: [w, h, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // 128 * 4 = 512 (multiple of 256, valid bytesPerRow)
    device.queue.writeTexture(
      { texture: tex },
      data,
      { bytesPerRow: w * 4, rowsPerImage: h },
      [w, h, 1]
    );

    return tex;
  }
}

//TEXTURAS
const texture  = await loadTextureOrCheckerboard("/textures/Negro.avif");
const texture1 = await loadTextureOrCheckerboard("/textures/Ladrillo.webp");
const texture2   = await loadTextureOrCheckerboard("/textures/Metal.avif");

const sampler = device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  addressModeU: "repeat",
  addressModeV: "repeat",
});

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: sampler },
    { binding: 2, resource: texture.createView() },
  ],
});

const camera = new Camera();
const keys = new Set<string>();

type ModelOrder = "TRS" | "RTS" | "SRT";
interface CubeState {
  tx: number;
  ty: number;
  tz: number;
  rx: number;
  ry: number;
  rz: number;
  s: number;
  order: ModelOrder;
  uniformBuffer?: GPUBuffer;
  bindGroup?: GPUBindGroup;
  textureView?: GPUTextureView;
  rotSpeedX: number;
  rotSpeedY: number;
  rotSpeedZ: number;
}

const cubes: CubeState[] = [];

// Función para crear uniform buffer y bind group para un cubo
function createCubeBindGroup(cube: CubeState): void {

  const uniBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: cube.textureView! },
    ],
  });

  cube.uniformBuffer = uniBuffer;
  cube.bindGroup = bGroup;
}

// Generar cubo aleatorio al presionar cualquier tecla
window.addEventListener("keydown", (e) => {
  keys.add(e.key);

  const excludeKeys = ["w", "W", "a", "A", "s", "S", "d", "D", 
                       "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
                       "q", "Q", "e", "E"];

  if (excludeKeys.includes(e.key)) return;

  //TEXTURA ALEATORIA
  const textures = [
    texture.createView(),
    texture1.createView(),
    texture2.createView(),
  ];

  const randomTexture =
    textures[Math.floor(Math.random() * textures.length)];

  const randomAngle1 = Math.random() * Math.PI * 2;
  const randomAngle2 = Math.random() * Math.PI * 2;
  const randomAngle3 = Math.random() * Math.PI * 2;

  const newCube: CubeState = {
    tx: (Math.random() - 0.5) * 50,
    ty: (Math.random() - 0.5) * 50,
    tz: (Math.random() - 0.5) * 50,
    rx: Math.random() * Math.PI * 2,
    ry: Math.random() * Math.PI * 2,
    rz: Math.random() * Math.PI * 2,
    s: Math.random() * 1.5 + 0.5,
    order: "TRS",
    textureView: randomTexture,
    rotSpeedX: Math.sin(randomAngle1) * 2,
    rotSpeedY: Math.cos(randomAngle2) * 2,
    rotSpeedZ: Math.sin(randomAngle3) * 2,
  };

  createCubeBindGroup(newCube);
  cubes.push(newCube);

  console.log(`Cubo creado. Total: ${cubes.length}`);
});

window.addEventListener("keyup", (e) => keys.delete(e.key));

const initialAngle1 = Math.random() * Math.PI * 2;
const initialAngle2 = Math.random() * Math.PI * 2;
const initialAngle3 = Math.random() * Math.PI * 2;

const modelState = {
  tx: 0, ty: 0, tz: 0,
  rx: 0, ry: 0, rz: 0,
  s: 1,
  order: "TRS" as ModelOrder,

  rotSpeedX: Math.sin(initialAngle1) * 2,
  rotSpeedY: Math.cos(initialAngle2) * 2,
  rotSpeedZ: Math.sin(initialAngle3) * 2,
};

function updateModelState() {
  // Order toggles
  if (keys.has("1")) modelState.order = "TRS";
  if (keys.has("2")) modelState.order = "RTS";
  if (keys.has("3")) modelState.order = "SRT";
}

function buildModelMatrix(cube: CubeState) {
  const T = mat4.translation(cube.tx, cube.ty, cube.tz);
  const S = mat4.scaling(cube.s, cube.s, cube.s);

  const Rx = mat4.rotationX(cube.rx);
  const Ry = mat4.rotationY(cube.ry);
  const Rz = mat4.rotationZ(cube.rz);

  const R = mat4.multiply(Rz, mat4.multiply(Ry, Rx));

  switch (cube.order) {
    case "TRS": return mat4.multiply(T, mat4.multiply(R, S));
    case "RTS": return mat4.multiply(R, mat4.multiply(T, S));
    case "SRT": return mat4.multiply(S, mat4.multiply(R, T));
    default:    return mat4.multiply(T, mat4.multiply(R, S));
  }
}

let lastTime = performance.now();

function frame(now: number) {
  const dt = Math.min(0.033, (now - lastTime) / 1000);
  lastTime = now;

  camera.update(keys, dt);
  updateModelState(dt);

  const aspect = canvas.width / canvas.height;
  const proj = mat4.perspective((60 * Math.PI) / 180, aspect, 0.1, 100.0);
  const view = camera.getViewMatrix();

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.06, g: 0.08, b: 0.12, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depthTexture!.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });

  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vertexBuffer);

  const vp = mat4.multiply(proj, view);

// Rotación automática del cubo inicial
  modelState.rx += modelState.rotSpeedX * dt;
  modelState.ry += modelState.rotSpeedY * dt;
  modelState.rz += modelState.rotSpeedZ * dt;

  const model = buildModelMatrix(modelState);
  const mvp = mat4.multiply(vp, model);
  device.queue.writeBuffer(uniformBuffer, 0, mvp);
  pass.setBindGroup(0, bindGroup);
  pass.draw(vertexCount);

  // Renderizar todos los cubos generados
  for (const cube of cubes) {
    // Actualizar rotación dinámicamente usando velocidades
    cube.rx += cube.rotSpeedX * dt;
    cube.ry += cube.rotSpeedY * dt;
    cube.rz += cube.rotSpeedZ * dt;
    
    const cubeModel = buildModelMatrix(cube);
    const cubeMvp = mat4.multiply(vp, cubeModel);
    if (cube.uniformBuffer && cube.bindGroup) {
      device.queue.writeBuffer(cube.uniformBuffer, 0, cubeMvp);
      pass.setBindGroup(0, cube.bindGroup);
      pass.draw(vertexCount);
    }
  }

  pass.end();

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
