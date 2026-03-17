const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl', {
  preserveDrawingBuffer: true,
  antialias: false,
  alpha: false,
  powerPreference: 'high-performance'
});

if (!gl) {
  alert('WebGL not supported');
  throw new Error('WebGL not supported');
}

gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
gl.clearColor(0, 0, 0, 1.0);

// --- State ---
let time = 0;
let flockCx = 0, flockCy = 0;
let flockVx = 0, flockVy = 0;
let flockTargetX = 0, flockTargetY = 0;
let flockTargetTimer = 0;

// --- Musical conductor state ---
let murmuEnergy     = 0;   // 0–1, current energy level
let murmuEnergyPrev = 0;   // previous frame value, for dE/dt
let murmuEnergyDot  = 0;   // smoothed rate of change (rubato)
let sweepPhase      = 0;   // integrates over time → circular sweep direction
let waveAngle       = Math.PI * 0.25;  // direction of traveling plane wave through flock
let vortexBlend     = 0;   // 0=calm sheet, 1=full spiral arms — driven by energy peaks

const state = {
  scale: 30,
  speed: 20,
  palette: 'mono',
  shape: 'circle',
  tightness: 65,
  motion: 35,
  cursorStrength: 100,
  frequency: 3,
  waveSpeed: 40,
  trail: 0,
  murmuMotion: 'off',
};

const paletteIndex = { mono: 0, ice: 1, ember: 2, acid: 3, dusk: 4, aurora: 5 };

// --- Shaders ---
const VS = `
attribute vec2 a_position;
attribute float a_t;
attribute float a_size;
uniform vec2 u_resolution;
varying float v_t;
void main() {
  vec2 clip = (a_position / u_resolution) * 2.0 - 1.0;
  clip.y = -clip.y;
  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = a_size;
  v_t = a_t;
}
`;

const FS = `
precision mediump float;
uniform int u_palette;
varying float v_t;

vec3 palette_mono(float t) {
  return vec3(t);
}

vec3 palette_ice(float t) {
  return vec3(mix(10.0/255.0,  220.0/255.0, t),
              mix(20.0/255.0,  240.0/255.0, t),
              mix(80.0/255.0,  255.0/255.0, t));
}

vec3 palette_ember(float t) {
  float r = t < 0.5 ? mix(0.0,        200.0/255.0, t*2.0)
                    : mix(200.0/255.0, 1.0,         (t-0.5)*2.0);
  float g = t < 0.5 ? 0.0 : mix(0.0, 200.0/255.0, (t-0.5)*2.0);
  float b = t < 0.2 ? 0.0 : mix(0.0, 30.0/255.0,  (t-0.2)/0.8);
  return vec3(r, g, b);
}

vec3 palette_acid(float t) {
  float r = t < 0.5 ? mix(30.0/255.0, 0.0,         t*2.0)
                    : mix(0.0,         200.0/255.0,  (t-0.5)*2.0);
  float g = mix(0.0, 1.0, t);
  float b = t < 0.5 ? mix(60.0/255.0, 40.0/255.0,  t*2.0)
                    : mix(40.0/255.0,  100.0/255.0,  (t-0.5)*2.0);
  return vec3(r, g, b);
}

vec3 palette_dusk(float t) {
  float r = t < 0.5 ? mix(10.0/255.0,  200.0/255.0, t*2.0)
                    : mix(200.0/255.0,  1.0,          (t-0.5)*2.0);
  float g = t < 0.5 ? mix(5.0/255.0,   60.0/255.0,  t*2.0)
                    : mix(60.0/255.0,   210.0/255.0,  (t-0.5)*2.0);
  float b = t < 0.5 ? mix(40.0/255.0,  120.0/255.0, t*2.0)
                    : mix(120.0/255.0,  120.0/255.0,  (t-0.5)*2.0);
  return vec3(r, g, b);
}

vec3 palette_aurora(float t) {
  float r = t < 0.5 ? 0.0 : mix(0.0, 123.0/255.0, (t-0.5)*2.0);
  float g = t < 0.5 ? mix(26.0/255.0, 1.0, t*2.0)
                    : mix(1.0, 47.0/255.0, (t-0.5)*2.0);
  float b = t < 0.5 ? mix(15.0/255.0, 204.0/255.0, t*2.0)
                    : mix(204.0/255.0, 1.0, (t-0.5)*2.0);
  return vec3(r, g, b);
}

vec3 applyPalette(float t, int p) {
  if (p == 1) return palette_ice(t);
  if (p == 2) return palette_ember(t);
  if (p == 3) return palette_acid(t);
  if (p == 4) return palette_dusk(t);
  if (p == 5) return palette_aurora(t);
  return palette_mono(t);
}

void main() {
  vec2 coord = gl_PointCoord - vec2(0.5);
  float r = dot(coord, coord);
  if (r > 0.25) discard;
  float alpha = exp(-r * 12.0);
  float colorT = clamp(v_t * 1.4, 0.0, 1.0);
  gl_FragColor = vec4(applyPalette(colorT, u_palette), alpha);
}
`;

// --- Shader compilation ---
function compileShader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(s));
  }
  return s;
}

const prog = gl.createProgram();
gl.attachShader(prog, compileShader(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
  throw new Error(gl.getProgramInfoLog(prog));
}
gl.useProgram(prog);

// --- Fade program (fullscreen dark quad for trail effect) ---
const FADE_VS = `
attribute vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }
`;
const FADE_FS = `
precision mediump float;
uniform float u_alpha;
void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, u_alpha); }
`;
const fadeProg = gl.createProgram();
gl.attachShader(fadeProg, compileShader(gl.VERTEX_SHADER,   FADE_VS));
gl.attachShader(fadeProg, compileShader(gl.FRAGMENT_SHADER, FADE_FS));
gl.linkProgram(fadeProg);
const fadeAPos   = gl.getAttribLocation(fadeProg,  'a_pos');
const uFadeAlpha = gl.getUniformLocation(fadeProg, 'u_alpha');

// Fullscreen clip-space quad (triangle strip: BL, BR, TL, TR)
const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
  -1, -1,   1, -1,   -1, 1,   1, 1,
]), gl.STATIC_DRAW);
gl.bindBuffer(gl.ARRAY_BUFFER, null);

gl.useProgram(prog);

// --- Murmuration spread constants ---
const MURM_SPREAD_X = 0.24;  // half-width fraction of W (full width = 48% canvas)
const MURM_SPREAD_Y = 0.13;  // half-height fraction of H (full height = 26% canvas)

// --- Particle data ---
let N = 1600;
let posArr = new Float32Array(N * 2);
let tArr   = new Float32Array(N);
let particles = new Array(N);
const posBuf  = gl.createBuffer();
const tBuf    = gl.createBuffer();
const sizeBuf = gl.createBuffer();
let sizeArr   = new Float32Array(N);

const aPos  = gl.getAttribLocation(prog, 'a_position');
const aTee  = gl.getAttribLocation(prog, 'a_t');
const aSize = gl.getAttribLocation(prog, 'a_size');
const uRes     = gl.getUniformLocation(prog, 'u_resolution');
const uPalette = gl.getUniformLocation(prog, 'u_palette');

gl.enableVertexAttribArray(aPos);
gl.enableVertexAttribArray(aTee);
gl.enableVertexAttribArray(aSize);

// --- Noise helpers (CPU) ---
function hash21(x, y) {
  let n = Math.imul(x | 0, 1597334677) ^ Math.imul(y | 0, 3812015801);
  n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
  n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
  return ((n >>> 0) / 0xffffffff) * 2.0 - 1.0;
}

function noiseXY(px, py) {
  const ix = Math.floor(px), iy = Math.floor(py);
  const fx = px - ix, fy = py - iy;
  const ux = fx*fx*(3-2*fx), uy = fy*fy*(3-2*fy);
  const a = hash21(ix,   iy),   b = hash21(ix+1, iy);
  const c = hash21(ix,   iy+1), d = hash21(ix+1, iy+1);
  return (a*(1-ux)+b*ux)*(1-uy) + (c*(1-ux)+d*ux)*uy;
}

// --- Wave math helpers ---
function hermite(n, x) {
  switch (n) {
    case 0: return 1;
    case 1: return 2 * x;
    case 2: return 4*x*x - 2;
    case 3: return 8*x*x*x - 12*x;
    case 4: return 16*x*x*x*x - 48*x*x + 12;
    case 5: return 32*x*x*x*x*x - 160*x*x*x + 120*x;
    case 6: return 64*Math.pow(x,6) - 480*Math.pow(x,4) + 720*x*x - 120;
    default: {
      // recurrence: H_{n+1} = 2x·Hₙ - 2n·Hₙ₋₁
      let h0 = 1, h1 = 2 * x;
      for (let k = 1; k < n; k++) {
        const h2 = 2 * x * h1 - 2 * k * h0;
        h0 = h1; h1 = h2;
      }
      return h1;
    }
  }
}

function qhoPsi(n, x) { return hermite(n, x) * Math.exp(-0.5 * x * x); }

const LISSAJOUS_RATIOS = [null,[1,1],[1,2],[1,3],[2,3],[3,4],[3,5],[4,5],[5,6]];

// --- Hopf fibration helpers ---
const HOPF_FIBERS = 32;
const HOPF_GOLDEN = (1 + Math.sqrt(5)) / 2; // golden ratio for Fibonacci fiber spacing

// Project a Hopf fiber point (theta, phi, psi) from S³ → ℝ³ → canvas 2D.
// theta: colatitude of base point on S² (0=north pole, π/2=equator)
// phi:   longitude of base point on S²
// psi:   position along the fiber circle (0..2π)
// rotY:  time-driven spin around Y-axis; tiltX: freq-driven tilt around X-axis
function hopfTo2D(theta, phi, psi, cx, cy, R, rotY, tiltX) {
  const alpha = theta / 2;
  const a = Math.cos(alpha) * Math.cos((psi + phi) / 2);
  const b = Math.cos(alpha) * Math.sin((psi + phi) / 2);
  const c = Math.sin(alpha) * Math.cos((psi - phi) / 2);
  const d = Math.sin(alpha) * Math.sin((psi - phi) / 2);
  // Stereographic projection S³ → ℝ³ (from north pole (0,0,0,1))
  const denom = Math.max(1 - d, 0.12);
  const X = a / denom;
  const Y = b / denom;
  const Z = c / denom;
  // Tilt around X-axis (freq-controlled static viewing angle)
  const cosT = Math.cos(tiltX), sinT = Math.sin(tiltX);
  const Y1 =  Y * cosT - Z * sinT;
  const Z1 =  Y * sinT + Z * cosT;
  // Spin around Y-axis (time-driven)
  const cosR = Math.cos(rotY), sinR = Math.sin(rotY);
  const X2 =  X * cosR + Z1 * sinR;
  const Z2 = -X * sinR + Z1 * cosR;
  // Weak perspective projection to 2D
  const scale = R / (1.8 + Z2 * 0.18);
  return [cx + X2 * scale, cy + Y1 * scale];
}

// --- Shape generators ---
// All return array of {hx, hy, bx, by} in pixel space
// bx/by = base coordinates used to recompute hx/hy each tick for wave shapes
function generateCircle(cx, cy, W, H) {
  const R = Math.min(W, H) * 0.38;
  return Array.from({ length: N }, () => {
    const r = R * Math.sqrt(Math.random());
    const a = Math.random() * Math.PI * 2;
    return { hx: cx + Math.cos(a) * r, hy: cy + Math.sin(a) * r };
  });
}

function generateSquare(cx, cy, W, H) {
  const half = Math.min(W, H) * 0.36;
  return Array.from({ length: N }, () => ({
    hx: cx + (Math.random() * 2 - 1) * half,
    hy: cy + (Math.random() * 2 - 1) * half,
  }));
}

function generateTriangle(cx, cy, W, H) {
  const size = Math.min(W, H) * 0.42;
  // Equilateral pointing up, vertices relative to center
  const v0x = 0,               v0y = -size * (2/3);
  const v1x = -size * 0.5,     v1y =  size * (1/3);
  const v2x =  size * 0.5,     v2y =  size * (1/3);
  return Array.from({ length: N }, () => {
    let u = Math.random(), v = Math.random();
    if (u + v > 1) { u = 1 - u; v = 1 - v; }
    const w = 1 - u - v;
    return {
      hx: cx + u * v0x + v * v1x + w * v2x,
      hy: cy + u * v0y + v * v1y + w * v2y,
    };
  });
}

function generateFreeform(cx, cy, W, H) {
  const sx = W * 0.22, sy = H * 0.22;
  return Array.from({ length: N }, () => {
    // Box-Muller
    const u1 = Math.random() + 1e-10, u2 = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u1));
    const nx = mag * Math.cos(2 * Math.PI * u2);
    const ny = mag * Math.sin(2 * Math.PI * u2);
    return { hx: cx + nx * sx, hy: cy + ny * sy };
  });
}

function generateWave(cx, cy, W, H) {
  const spanW = W * 0.78, x0 = cx - spanW / 2;
  return Array.from({ length: N }, (_, i) => {
    const t = i / (N - 1);
    return { hx: x0 + t * spanW, hy: cy, bx: x0 + t * spanW, by: t };
  });
}

function generateHarmonic(cx, cy, W, H) {
  const n = Math.max(0, Math.round(state.frequency) - 1);
  const xRange = 3.5 + n * 0.5;
  const spanW = W * 0.72;
  const result = [];

  // Find max |ψₙ|² for rejection sampling
  let maxPsi2 = 0;
  const samples = 500;
  for (let i = 0; i <= samples; i++) {
    const xi = -xRange + (2 * xRange * i / samples);
    const psi = qhoPsi(n, xi);
    maxPsi2 = Math.max(maxPsi2, psi * psi);
  }
  if (maxPsi2 < 1e-10) maxPsi2 = 1;

  // Rejection sampling: density ∝ ψₙ²(x)
  let attempts = 0;
  while (result.length < N && attempts < N * 30) {
    const xi = (Math.random() * 2 - 1) * xRange;
    const psi = qhoPsi(n, xi);
    if (Math.random() < (psi * psi) / maxPsi2) {
      const bx = cx - spanW / 2 + (xi + xRange) / (2 * xRange) * spanW;
      result.push({ hx: bx, hy: cy, bx, by: xi });
    }
    attempts++;
  }

  // Fallback: uniform fill if rejection sampling runs short
  while (result.length < N) {
    const xi = (Math.random() * 2 - 1) * xRange;
    const bx = cx - spanW / 2 + (xi + xRange) / (2 * xRange) * spanW;
    result.push({ hx: bx, hy: cy, bx, by: xi });
  }

  return result;
}

function generateLissajous(cx, cy, W, H) {
  const freq = Math.max(1, Math.min(8, Math.round(state.frequency)));
  const [a, b] = LISSAJOUS_RATIOS[freq];
  const R = Math.min(W, H) * 0.36;
  return Array.from({ length: N }, (_, i) => {
    const theta = (i / N) * 2 * Math.PI;
    return { hx: cx + Math.sin(a * theta) * R, hy: cy + Math.sin(b * theta) * R, bx: theta, by: 0 };
  });
}

function generateMurmuration(cx, cy, W, H) {
  const spreadX = W * MURM_SPREAD_X, spreadY = H * MURM_SPREAD_Y;
  const sigZ    = spreadX * 0.28;
  return Array.from({ length: N }, () => {
    const r     = Math.sqrt(Math.random());          // sqrt → area-uniform distribution
    const angle = Math.random() * 2 * Math.PI;
    const lobeMod = 1.0 + 0.22 * Math.cos(3 * angle + 1.1);  // subtle 3-lobe amoeba
    const bxr = r * spreadX * lobeMod * Math.cos(angle);
    const byr = r * spreadY * lobeMod * Math.sin(angle);
    // Z-depth per bird (organic thickness)
    const u3 = Math.random() + 1e-10;
    const bzr = Math.sqrt(-2 * Math.log(u3)) * Math.cos(2 * Math.PI * Math.random()) * sigZ;
    // Two-octave noise warp (frequencies scaled down for wider spread)
    const wLx = noiseXY(bxr * 0.00025 + 7.1,  byr * 0.00025)        * W * 0.022;
    const wLy = noiseXY(bxr * 0.00025 + 13.4, byr * 0.00025 + 5.9)  * W * 0.022;
    const wHx = noiseXY(bxr * 0.00065 + 21.2, byr * 0.00065)        * W * 0.009;
    const wHy = noiseXY(bxr * 0.00065 + 9.7,  byr * 0.00065 + 3.3)  * W * 0.009;
    const hx = cx + bxr + wLx + wHx;
    const hy = cy + byr + wLy + wHy;
    return { hx, hy, bx_rel: bxr + wLx + wHx, by_rel: byr + wLy + wHy, bz_rel: bzr };
  });
}

function generateHopf(cx, cy, W, H) {
  const M = HOPF_FIBERS;
  const perFiber = Math.ceil(N / M);
  const R = Math.min(W, H) * 0.38;
  const result = [];
  for (let f = 0; f < M && result.length < N; f++) {
    // Fibonacci lattice on northern hemisphere of S²: cosTheta ∈ [0,1] → theta ∈ [π/2,0]
    const cosTheta = f / Math.max(M - 1, 1);
    const theta    = Math.acos(cosTheta);
    const phi      = (2 * Math.PI * f) / HOPF_GOLDEN;
    for (let j = 0; j < perFiber && result.length < N; j++) {
      const psi  = (j / perFiber) * 2 * Math.PI;
      // bx = psi (position along fiber), by = f (fiber index, encodes theta+phi)
      const [hx, hy] = hopfTo2D(theta, phi, psi, cx, cy, R, 0, 0);
      result.push({ hx, hy, bx: psi, by: f });
    }
  }
  return result;
}

function updateMurmuralGrid(dt) {
  if (state.shape !== 'murmuration') return;

  // A: Cursor injection
  const cursorCellX = cursor.x * GRID_COLS;
  const cursorCellY = cursor.y * GRID_ROWS;
  const injRadius = (state.frequency / 8) * 5.0 + 1.5;
  const speedFactor = Math.min(cursor.smoothedSpeed / 400, 1.0);
  const cStr = state.cursorStrength / 100;

  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const d = Math.hypot(c - cursorCellX, r - cursorCellY);
      if (d < injRadius) {
        const t = 1 - d / injRadius;
        const inject = t * speedFactor * 0.8 * cStr;
        const idx = r * GRID_COLS + c;
        distGrid[idx] = Math.min(distGrid[idx] + inject * dt * 20, 1.0);
      }
    }
  }

  // Phantom predator: autonomous Lissajous path injects disturbance continuously
  const predSpeed = (state.waveSpeed / 100) * 0.55 + 0.10;
  const predX = 0.5 + Math.sin(time * predSpeed * 0.7) * 0.38;
  const predY = 0.5 + Math.sin(time * predSpeed * 0.41 + 1.57) * 0.28;
  const predCellX = predX * GRID_COLS;
  const predCellY = predY * GRID_ROWS;
  const predStr = 0.15 + 0.85 * murmuEnergy;

  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const d = Math.hypot(c - predCellX, r - predCellY);
      if (d < injRadius) {
        const t = 1 - d / injRadius;
        const idx = r * GRID_COLS + c;
        distGrid[idx] = Math.min(distGrid[idx] + t * predStr * cStr * dt * 20, 1.0);
      }
    }
  }

  // B: Diffusion (5-point Laplacian stencil)
  const diffRate = (state.waveSpeed / 100) * 0.22; // capped below 0.25 stability limit
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const idx = r * GRID_COLS + c;
      const up  = distGrid[Math.max(r-1, 0) * GRID_COLS + c];
      const dn  = distGrid[Math.min(r+1, GRID_ROWS-1) * GRID_COLS + c];
      const lt  = distGrid[r * GRID_COLS + Math.max(c-1, 0)];
      const rt  = distGrid[r * GRID_COLS + Math.min(c+1, GRID_COLS-1)];
      distGridNext[idx] = distGrid[idx] + (up + dn + lt + rt - 4 * distGrid[idx]) * diffRate;
    }
  }

  // Ambient background murmur: keeps grid above zero so flock always has low-level agitation
  const murmurStr = 0.018 + 0.062 * murmuEnergy;
  const murmurT = time * 0.09;
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const idx = r * GRID_COLS + c;
      const ambient = (noiseXY(c * 0.4 + murmurT, r * 0.4 + murmurT * 0.6) * 0.5 + 0.5) * murmurStr;
      distGrid[idx] = Math.max(distGrid[idx], ambient);
    }
  }

  // C: Decay + swap
  const decay = Math.pow(0.965, dt * 60);
  for (let i = 0; i < distGrid.length; i++) {
    distGrid[i] = Math.min(Math.max(distGridNext[i] * decay, 0), 1.0);
  }

  // D: Spatial gradient of distGrid (central differences) — used for flow field displacement
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const idx = r * GRID_COLS + c;
      const lt = distGrid[r * GRID_COLS + Math.max(c - 1, 0)];
      const rt = distGrid[r * GRID_COLS + Math.min(c + 1, GRID_COLS - 1)];
      const up = distGrid[Math.max(r - 1, 0) * GRID_COLS + c];
      const dn = distGrid[Math.min(r + 1, GRID_ROWS - 1) * GRID_COLS + c];
      distGradX[idx] = rt - lt;
      distGradY[idx] = dn - up;
    }
  }
}

function generateHomes(W, H) {
  const cx = W / 2, cy = H / 2;
  switch (state.shape) {
    case 'square':    return generateSquare(cx, cy, W, H);
    case 'triangle':  return generateTriangle(cx, cy, W, H);
    case 'freeform':  return generateFreeform(cx, cy, W, H);
    case 'wave':      return generateWave(cx, cy, W, H);
    case 'harmonic':  return generateHarmonic(cx, cy, W, H);
    case 'lissajous':    return generateLissajous(cx, cy, W, H);
    case 'murmuration': return generateMurmuration(cx, cy, W, H);
    case 'hopf':        return generateHopf(cx, cy, W, H);
    default:             return generateCircle(cx, cy, W, H);
  }
}

// --- Flock waypoint ---
function pickNewFlockTarget(W, H) {
  const margin = 0.15;
  flockTargetX    = (margin + Math.random() * (1 - 2 * margin)) * W;
  flockTargetY    = (margin + Math.random() * (1 - 2 * margin)) * H;
  flockTargetTimer = 10 + Math.random() * 10; // 10–20s before forcing new target
}

// --- Init particles ---
function initParticles() {
  const W = canvas.width, H = canvas.height;
  if (state.shape === 'murmuration') {
    flockCx = W / 2; flockCy = H / 2; flockVx = 0; flockVy = 0;
    pickNewFlockTarget(W, H);
  }
  const homes = generateHomes(W, H);
  for (let i = 0; i < N; i++) {
    const seed = Math.random() * 1000;
    particles[i] = {
      x:   homes[i].hx,
      y:   homes[i].hy,
      vx:  0,
      vy:  0,
      hx:  homes[i].hx,
      hy:  homes[i].hy,
      bx:  homes[i].bx     ?? homes[i].hx,
      by:  homes[i].by     ?? homes[i].hy,
      bxr: homes[i].bx_rel ?? 0,
      byr: homes[i].by_rel ?? 0,
      bzr: homes[i].bz_rel ?? 0,
      apTable: homes[i].apTable ?? null,
      seed,
    };
  }
}

// Apply a homes array onto existing particles (position + base + apTable)
function applyHomes(homes) {
  for (let i = 0; i < N; i++) {
    particles[i].hx      = homes[i].hx;
    particles[i].hy      = homes[i].hy;
    particles[i].bx      = homes[i].bx     ?? homes[i].hx;
    particles[i].by      = homes[i].by     ?? homes[i].hy;
    particles[i].bxr     = homes[i].bx_rel ?? 0;
    particles[i].byr     = homes[i].by_rel ?? 0;
    particles[i].bzr     = homes[i].bz_rel ?? 0;
    particles[i].apTable = homes[i].apTable ?? null;
  }
}

function reinitAll(newN) {
  N = newN;
  posArr    = new Float32Array(N * 2);
  tArr      = new Float32Array(N);
  sizeArr   = new Float32Array(N);
  particles = new Array(N);
  initParticles();
}

// --- Cursor ---
const cursor = {
  x: 0.5, y: 0.5,
  vx: 0, vy: 0,
  smoothedSpeed: 0,
  prevCX: null, prevCY: null, prevT: 0,
};

// --- Hopf per-frame rotation state (updated once per tick before particle loop) ---
let hopfRotY = 0, hopfTiltX = 0;

const GRID_COLS = 24;
const GRID_ROWS = 18;
let distGrid     = new Float32Array(GRID_COLS * GRID_ROWS);
let distGridNext = new Float32Array(GRID_COLS * GRID_ROWS);
let distGradX    = new Float32Array(GRID_COLS * GRID_ROWS);
let distGradY    = new Float32Array(GRID_COLS * GRID_ROWS);

// --- Boids spatial grid (murmuration nearest-neighbor) ---
const BOIDS_COLS = 36;
const BOIDS_ROWS = 27;
const boidsGrid  = Array.from({ length: BOIDS_COLS * BOIDS_ROWS }, () => []);
// Allocation-free scratch buffers for candidate search hot loop
const _boidsCandIdx  = new Int32Array(128);
const _boidsCandDist = new Float32Array(128);

window.addEventListener('mousemove', e => {
  const now = performance.now() / 1000;
  const dt  = Math.max(now - cursor.prevT, 0.001);
  if (cursor.prevCX !== null) {
    cursor.vx = (e.clientX - cursor.prevCX) / dt;
    cursor.vy = (e.clientY - cursor.prevCY) / dt;
    const raw = Math.hypot(cursor.vx, cursor.vy);
    cursor.smoothedSpeed = cursor.smoothedSpeed * 0.85 + raw * 0.15;
  }
  cursor.x = e.clientX / window.innerWidth;
  cursor.y = e.clientY / window.innerHeight;
  cursor.prevCX = e.clientX;
  cursor.prevCY = e.clientY;
  cursor.prevT  = now;
});

// --- Physics tick ---
function tick(dt) {
  const W = canvas.width, H = canvas.height;
  const k = (state.tightness / 100) * 8.0;
  const nScale = (state.scale / 100) * 0.008;
  const nTime  = time * (state.speed / 100) * 0.4;
  const motionAmp = (state.motion / 100) * 120;

  // cursor in pixel space (Y-flipped: canvas Y is top-down, cursor.y is 0=top)
  const cxPx = cursor.x * W;
  const cyPx = cursor.y * H;
  const radius = 0.2 * Math.min(W, H);

  // Wave animation state (computed once per tick)
  const cx        = W / 2;
  const cy        = H / 2;
  const phaseRate = (state.waveSpeed / 100) * 3.0;
  const phase     = time * phaseRate;
  const isWave    = ['wave','harmonic','lissajous','murmuration','hopf'].includes(state.shape);
  const freq      = Math.max(1, Math.min(8, Math.round(state.frequency)));
  const ampY      = H * 0.22;

  cursor.smoothedSpeed *= 0.92;

  // Musical conductor: energy envelope drives all murmuration dynamics
  if (state.shape === 'murmuration') {
    // Three oscillators with prime-ratio periods — never repeats within a session
    const e7  = Math.sin(2 * Math.PI * time / 7)  * 0.5 + 0.5;  // beat (~7s)
    const e17 = Math.sin(2 * Math.PI * time / 17) * 0.5 + 0.5;  // phrase (~17s)
    const e47 = Math.sin(2 * Math.PI * time / 47) * 0.5 + 0.5;  // section (~47s)
    const eLinear = e7 * 0.30 + e17 * 0.45 + e47 * 0.25;
    // Power-law: spends ~96% of time quiet; crescendos are steep and rare
    murmuEnergy = Math.pow(eLinear, 3.5);
    // Rubato: track rate of change for hold-back/release effect
    const rawDot = (murmuEnergy - murmuEnergyPrev) / Math.max(dt, 0.001);
    murmuEnergyDot  = murmuEnergyDot * 0.85 + rawDot * 0.15;
    murmuEnergyPrev = murmuEnergy;
    waveAngle += dt * (0.013 + 0.025 * murmuEnergy);

    // Vortex burst: fast attack when energy crescendos, slow lingering decay
    const vortexTarget = murmuEnergy > 0.35
      ? Math.pow((murmuEnergy - 0.35) / 0.65, 1.5)
      : 0;
    const vortexRate = vortexTarget > vortexBlend ? 1.5 : 0.35;
    vortexBlend += (vortexTarget - vortexBlend) * Math.min(dt * vortexRate, 1.0);
  }

  updateMurmuralGrid(dt);

  // Flock drift: entire murmuration travels between waypoints with ease-in/ease-out
  if (state.shape === 'murmuration') {
    // Advance timer; pick new target on arrival or timeout
    flockTargetTimer -= dt;
    const distToTarget = Math.hypot(flockTargetX - flockCx, flockTargetY - flockCy);
    if (flockTargetTimer <= 0 || distToTarget < W * 0.04) {
      pickNewFlockTarget(W, H);
    }

    // Steering vector toward target
    const tdx = flockTargetX - flockCx;
    const tdy = flockTargetY - flockCy;
    const tdist = Math.hypot(tdx, tdy) + 0.001;

    // Speed cap: base travel always present; energy adds urgency at crescendo
    const rubatoFactor = 1.0 - 0.25 * Math.sign(murmuEnergyDot)
                             * Math.min(Math.abs(murmuEnergyDot) / 0.08, 1.0);
    const maxFlockV = W * (0.04 + 0.06 * murmuEnergy * murmuEnergy) * rubatoFactor;

    // Ease-out: slow down as approaching target
    const desiredSpeed = Math.min(maxFlockV, tdist * 0.45);
    const desiredVx = (tdx / tdist) * desiredSpeed;
    const desiredVy = (tdy / tdist) * desiredSpeed;

    // Ease-in: gradually accelerate toward desired velocity
    flockVx += (desiredVx - flockVx) * 0.025;
    flockVy += (desiredVy - flockVy) * 0.025;

    // Soft boundary safety net (keeps flock in-bounds, rarely fires)
    const margin = 0.15;
    const fnx = flockCx / W, fny = flockCy / H;
    if (fnx < margin)     flockVx += (margin - fnx) * W * 0.04;
    if (fnx > 1 - margin) flockVx -= (fnx - (1 - margin)) * W * 0.04;
    if (fny < margin)     flockVy += (margin - fny) * H * 0.04;
    if (fny > 1 - margin) flockVy -= (fny - (1 - margin)) * H * 0.04;

    flockCx += flockVx * dt;
    flockCy += flockVy * dt;

    // Lissajous flock path: override waypoint drift to sweep a lissajous figure
    if (state.murmuMotion === 'lissajous') {
      const [la, lb] = LISSAJOUS_RATIOS[freq];
      const ls = phaseRate * 0.25;
      flockCx = W * 0.5 + Math.sin(la * time * ls) * W * 0.30;
      flockCy = H * 0.5 + Math.sin(lb * time * ls + Math.PI * 0.5) * H * 0.25;
    }
  }

  // Pre-compute 3D sheet fold constants for murmuration (used inside particle loop)
  let sigRef;
  let sheetAmpZ, sheetK1, sheetK2, sheetTilt, focalLen;
  let wdCos1, wdSin1, wdCos2, wdSin2, phase1, phase2;
  if (state.shape === 'murmuration') {
    const spreadX = W * MURM_SPREAD_X, spreadY = H * MURM_SPREAD_Y;
    sigRef     = spreadX;
    const calmAmp   = spreadY * (0.28 + 1.1  * murmuEnergy);
    const vortexAmp = spreadY * (0.12 + 0.45 * murmuEnergy);
    sheetAmpZ = calmAmp + (vortexAmp - calmAmp) * vortexBlend;
    sheetK1    = (Math.PI * freq) / (spreadX * 2.0);        // freq=1: one crease; freq=4: four
    sheetK2    = (Math.PI * (freq * 0.73)) / (spreadX * 2.0);
    wdCos1     = Math.cos(waveAngle);        wdSin1 = Math.sin(waveAngle);
    wdCos2     = Math.cos(waveAngle + 1.05); wdSin2 = Math.sin(waveAngle + 1.05);
    phase1     = time * phaseRate;
    phase2     = time * phaseRate * 1.31;
    sheetTilt  = 0.18 * Math.sin(time * 0.07) + 0.07 * Math.sin(time * 0.13);  // slow bank, ~80s cycle
    focalLen   = Math.min(W, H) * 1.1;
  }

  // Pre-fill boids spatial grid from current particle positions (murmuration only)
  if (state.shape === 'murmuration') {
    for (let ci = 0; ci < boidsGrid.length; ci++) boidsGrid[ci].length = 0;
    const bCellW = W / BOIDS_COLS, bCellH = H / BOIDS_ROWS;
    for (let i = 0; i < N; i++) {
      const p = particles[i];
      const bc = Math.min(Math.max(Math.floor(p.x / bCellW), 0), BOIDS_COLS - 1);
      const br = Math.min(Math.max(Math.floor(p.y / bCellH), 0), BOIDS_ROWS - 1);
      boidsGrid[br * BOIDS_COLS + bc].push(i);
    }
  }

  // Pre-compute Hopf rotation angles once per tick (used inside particle loop)
  if (state.shape === 'hopf') {
    hopfRotY  = phase * 0.25;
    hopfTiltX = ((freq - 1) / 7) * Math.PI * 0.6;
  }

  for (let i = 0; i < N; i++) {
    const p = particles[i];

    // Animated home update for wave shapes
    if (isWave) {
      if (state.shape === 'wave') {
        p.hx = p.bx;
        p.hy = cy + Math.sin(p.by * freq * 2 * Math.PI - phase) * ampY;
      } else if (state.shape === 'harmonic') {
        const n = freq - 1;
        const envelope = qhoPsi(n, p.by) * Math.cos(phase)
                       + qhoPsi(Math.min(n + 1, 7), p.by) * Math.sin(phase);
        p.hx = p.bx;
        p.hy = cy - envelope * ampY;
      } else if (state.shape === 'lissajous') {
        const [a, b] = LISSAJOUS_RATIOS[freq];
        const R = Math.min(W, H) * 0.36;
        p.hx = cx + Math.sin(a * p.bx + phase) * R;
        p.hy = cy + Math.sin(b * p.bx) * R;
      } else if (state.shape === 'hopf') {
        // Reconstruct fiber geometry from stored fiber index (p.by) and psi (p.bx)
        const f        = Math.round(p.by);
        const cosTheta = f / Math.max(HOPF_FIBERS - 1, 1);
        const theta    = Math.acos(cosTheta);
        const phi      = (2 * Math.PI * f) / HOPF_GOLDEN;
        const R        = Math.min(W, H) * 0.38;
        const [hx2, hy2] = hopfTo2D(theta, phi, p.bx, cx, cy, R, hopfRotY, hopfTiltX);
        p.hx = hx2;
        p.hy = hy2;
      } else if (state.shape === 'murmuration') {
        // Breathing (unchanged)
        const breatheAmp  = Math.max(0, 1.0 - murmuEnergy * 3) * 0.038 * W;
        const breathPhase = time * 0.65 + p.seed * 0.012;
        const breatheX    = Math.cos(breathPhase) * breatheAmp * 0.6;
        const rawSin = Math.sin(breathPhase * 1.13);
        const breatheY = (rawSin > 0
          ? Math.pow(rawSin, 0.7)
          : -Math.pow(-rawSin, 1.4)
        ) * breatheAmp;

        // Expansion (unchanged)
        const expansionScale = 1.0 + 1.8 * Math.pow(murmuEnergy, 1.5);

        // Differential rotation: inner birds faster than outer → arms sweep side to side
        const bRadius = Math.hypot(p.bxr, p.byr);
        const rNorm   = Math.min(bRadius / (W * MURM_SPREAD_X), 1.0);
        const omega   = phaseRate * 0.08 * (1.0 - 0.72 * rNorm) * (1.0 + 0.9 * murmuEnergy) * vortexBlend;
        const vortAng = time * omega + p.seed * 0.001;
        const cosV = Math.cos(vortAng), sinV = Math.sin(vortAng);
        const rotX = p.bxr * cosV - p.byr * sinV;
        const rotY = p.bxr * sinV + p.byr * cosV;

        // 3D fold on rotated position (secondary — adds organic crease density lines)
        const proj1 = rotX * wdCos1 + rotY * wdSin1;
        const proj2 = rotX * wdCos2 + rotY * wdSin2;
        const sZ = (sheetAmpZ * (0.62 * Math.sin(sheetK1 * proj1 - phase1)
                               + 0.38 * Math.sin(sheetK2 * proj2 - phase2))
                   + p.bzr * 0.15) * expansionScale;

        // Slow tilt + perspective (unchanged)
        const cosT = Math.cos(sheetTilt), sinT = Math.sin(sheetTilt);
        const baseY   = (rotY + breatheY) * expansionScale;
        const tiltedY = baseY * cosT - sZ * sinT;
        const tiltedZ = baseY * sinT + sZ * cosT;
        const perspScale = focalLen / Math.max(focalLen + tiltedZ, focalLen * 0.3);
        p.hx = flockCx + (rotX + breatheX) * expansionScale * perspScale;
        p.hy = flockCy + tiltedY * perspScale;

        // Lissajous-frequency undulation (tracks rotated position)
        const murmTheta = Math.atan2(rotY, rotX) + Math.PI;
        const [la, lb] = LISSAJOUS_RATIOS[freq];
        p.hx += Math.sin(la * murmTheta + phase * 0.5) * W * 0.018;
        p.hy += Math.sin(lb * murmTheta) * W * 0.018;

        // Collective wave roll across flock X extent
        if (state.murmuMotion === 'wave') {
          const wK = (Math.PI * freq) / (W * MURM_SPREAD_X);
          p.hy += Math.sin(p.bxr * wK - phase1) * H * MURM_SPREAD_Y * 0.8 * expansionScale;
        }

        // Flow field (unchanged)
        const cellW = W / GRID_COLS, cellH = H / GRID_ROWS;
        const gc = Math.min(Math.max(Math.floor(p.hx / cellW), 0), GRID_COLS - 1);
        const gr = Math.min(Math.max(Math.floor(p.hy / cellH), 0), GRID_ROWS - 1);
        const gradIdx = gr * GRID_COLS + gc;
        p.hx -= distGradX[gradIdx] * W * 0.012;
        p.hy -= distGradY[gradIdx] * H * 0.008;

        // Ambient noise drift (unchanged)
        const ambientT = time * 0.30;
        const noiseU = noiseXY(p.bxr * 0.0015 + ambientT * 0.6,       p.byr * 0.0015 + ambientT * 0.25 + 3.1);
        const noiseV = noiseXY(p.bxr * 0.0015 + ambientT * 0.4 + 8.4, p.byr * 0.0015 + ambientT * 0.5  + 2.7);
        p.hx += noiseU * W * 0.028;
        p.hy += noiseV * H * 0.032;

        // Topological boids: K=7 nearest neighbors (separation + alignment + cohesion)
        const K_NEIGHBORS = 7;
        const SEP_DIST = 28, SEP_STR = 60;
        const ALIGN_W = 0.08, COH_W = 0.04;

        let sepFx = 0, sepFy = 0, alignVx = 0, alignVy = 0, cohX = 0, cohY = 0;
        const bCellW2 = W / BOIDS_COLS, bCellH2 = H / BOIDS_ROWS;
        const pBc = Math.min(Math.max(Math.floor(p.x / bCellW2), 0), BOIDS_COLS - 1);
        const pBr = Math.min(Math.max(Math.floor(p.y / bCellH2), 0), BOIDS_ROWS - 1);

        let candCount = 0;
        const candIdx  = _boidsCandIdx;
        const candDist = _boidsCandDist;

        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            const nr = pBr + dr, nc = pBc + dc;
            if (nr < 0 || nr >= BOIDS_ROWS || nc < 0 || nc >= BOIDS_COLS) continue;
            const cell = boidsGrid[nr * BOIDS_COLS + nc];
            for (let ci = 0; ci < cell.length; ci++) {
              const j = cell[ci];
              if (j === i) continue;
              const q = particles[j];
              const ddx = q.x - p.x, ddy = q.y - p.y;
              const d2  = ddx * ddx + ddy * ddy;
              if (candCount < 128) { candIdx[candCount] = j; candDist[candCount] = d2; candCount++; }
            }
          }
        }

        // Partial insertion sort to find K nearest
        const kActual = Math.min(K_NEIGHBORS, candCount);
        for (let a = 0; a < kActual; a++) {
          let minPos = a;
          for (let b = a + 1; b < candCount; b++) {
            if (candDist[b] < candDist[minPos]) minPos = b;
          }
          const tmpI = candIdx[a]; candIdx[a] = candIdx[minPos]; candIdx[minPos] = tmpI;
          const tmpD = candDist[a]; candDist[a] = candDist[minPos]; candDist[minPos] = tmpD;
        }

        for (let a = 0; a < kActual; a++) {
          const q   = particles[candIdx[a]];
          const ddx = p.x - q.x, ddy = p.y - q.y;
          const d   = Math.sqrt(candDist[a]) + 0.001;
          if (d < SEP_DIST) {
            const push = (SEP_DIST - d) / SEP_DIST;
            sepFx += (ddx / d) * push * SEP_STR;
            sepFy += (ddy / d) * push * SEP_STR;
          }
          alignVx += q.vx; alignVy += q.vy;
          cohX    += q.x;  cohY    += q.y;
        }

        p.bfx = sepFx;
        p.bfy = sepFy;
        if (kActual > 0) {
          p.bfx += (alignVx / kActual - p.vx) * ALIGN_W + (cohX / kActual - p.x) * COH_W;
          p.bfy += (alignVy / kActual - p.vy) * ALIGN_W + (cohY / kActual - p.y) * COH_W;
        }
      }
    }

    // Spring (murmuration uses softer effective spring so ambient forces produce visible motion)
    const kEff = (state.shape === 'murmuration') ? k * 0.40 : k;
    const sfx = (p.hx - p.x) * kEff;
    const sfy = (p.hy - p.y) * kEff;

    // Noise drift
    const nx = noiseXY(p.x * nScale + nTime + p.seed,      p.y * nScale);
    const ny = noiseXY(p.x * nScale + p.seed + 31.7,       p.y * nScale + nTime + 17.3);
    let nfx = nx * motionAmp;
    let nfy = ny * motionAmp;
    // Add boids forces for murmuration (separation + alignment + cohesion)
    if (state.shape === 'murmuration') {
      nfx += p.bfx ?? 0;
      nfy += p.bfy ?? 0;
    }
    // Per-bird wingbeat jitter: each bird has an independent oscillation phase via p.seed
    if (state.shape === 'murmuration') {
      const jT = time * 2.8 + p.seed;
      const jitterAmp = motionAmp * 0.18;
      nfx = nfx * 0.65 + Math.cos(jT) * jitterAmp;
      nfy = nfy * 0.65 + Math.sin(jT * 1.37 + p.seed * 0.1) * jitterAmp * 0.7;
    }

    // Cursor force
    let cfx = 0, cfy = 0;
    const dx = p.x - cxPx, dy = p.y - cyPx;
    const dist = Math.hypot(dx, dy) + 0.001;
    if (dist < radius) {
      const t = dist / radius;
      const falloff = 1 - t*t*(3 - 2*t);
      const nx_ = dx / dist, ny_ = dy / dist;
      const cStr    = state.cursorStrength / 100;
      const proximity = 1 - dist / radius;
      const pushMag = proximity * proximity * 280 * cStr;

      const cvLen = Math.hypot(cursor.vx, cursor.vy) + 0.001;
      const cvx   = cursor.vx / cvLen;
      const cvy   = cursor.vy / cvLen;
      const kickMag = falloff * cursor.smoothedSpeed * 0.6 * cStr;

      cfx = nx_ * pushMag + cvx * kickMag;
      cfy = ny_ * pushMag + cvy * kickMag;
    }

    // Murmuration responds to cursor via disturbance grid, not direct push/kick
    if (state.shape === 'murmuration') { cfx = 0; cfy = 0; }

    // Integrate
    const ax = sfx + nfx + cfx;
    const ay = sfy + nfy + cfy;
    p.vx = (p.vx + ax * dt) * 0.88;
    p.vy = (p.vy + ay * dt) * 0.88;
    if (state.shape === 'murmuration') {
      const maxV = 400;
      const spd = Math.hypot(p.vx, p.vy);
      if (spd > maxV) { p.vx = p.vx / spd * maxV; p.vy = p.vy / spd * maxV; }
    }
    p.x += p.vx * dt;
    p.y += p.vy * dt;
  }
}

// --- Build GPU arrays ---
function buildArrays() {
  const phaseRate = (state.waveSpeed / 100) * 3.0;
  const phase     = time * phaseRate;
  const freq      = Math.max(1, Math.min(8, Math.round(state.frequency)));

  for (let i = 0; i < N; i++) {
    const p = particles[i];
    posArr[i*2]   = p.x;
    posArr[i*2+1] = p.y;

    let tVal;
    if (state.shape === 'wave') {
      // Colors travel with wave: crests bright, troughs dim
      tVal = Math.sin(p.by * freq * 2 * Math.PI - phase) * 0.5 + 0.5;
      sizeArr[i] = 5.0;
    } else if (state.shape === 'harmonic') {
      // Lobes glow, nodes dark; pulses with waveSpeed
      const n = freq - 1;
      const psi0 = qhoPsi(n, p.by);
      const psi1 = qhoPsi(Math.min(n + 1, 7), p.by);
      const envelope = psi0 * Math.cos(phase) + psi1 * Math.sin(phase);
      tVal = Math.min(Math.abs(envelope) / 1.5, 1.0);
      sizeArr[i] = 5.0;
    } else if (state.shape === 'lissajous') {
      // Rainbow gradient wraps around the figure
      tVal = p.bx / (2 * Math.PI);
      sizeArr[i] = 5.0;
    } else if (state.shape === 'murmuration') {
      tVal = 0.65;
      sizeArr[i] = 5.0;
    } else if (state.shape === 'hopf') {
      // Color cycles through the spectrum once per fiber — each linked ring gets a distinct hue
      tVal = Math.round(p.by) / Math.max(HOPF_FIBERS - 1, 1);
      sizeArr[i] = 3.5;
    } else {
      const noiseVal = noiseXY(p.x * 0.003 + time * 0.05 + p.seed, p.y * 0.003) * 0.5 + 0.5;
      const disp = Math.min(Math.hypot(p.x - p.hx, p.y - p.hy) / 80.0, 1.0);
      tVal = Math.max(noiseVal * 0.6, disp);
      sizeArr[i] = 5.0;
    }
    tArr[i] = tVal;
  }
}

// --- Render ---
function render() {
  if (state.trail === 0) {
    gl.clear(gl.COLOR_BUFFER_BIT);
  } else {
    // Fade previous frame toward background instead of hard-clearing
    // trail=1→fadeAlpha≈0.97 (short trail), trail=100→fadeAlpha=0.03 (very long trail)
    const fadeAlpha = 0.1 * Math.pow(0.02, state.trail / 100);
    gl.useProgram(fadeProg);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.enableVertexAttribArray(fadeAPos);
    gl.vertexAttribPointer(fadeAPos, 2, gl.FLOAT, false, 0, 0);
    gl.uniform1f(uFadeAlpha, fadeAlpha);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.useProgram(prog);
    gl.enableVertexAttribArray(aPos);
    gl.enableVertexAttribArray(aTee);
    gl.enableVertexAttribArray(aSize);
  }

  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, posArr, gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, tBuf);
  gl.bufferData(gl.ARRAY_BUFFER, tArr, gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(aTee, 1, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuf);
  gl.bufferData(gl.ARRAY_BUFFER, sizeArr, gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(aSize, 1, gl.FLOAT, false, 0, 0);

  // Trail mode: switch to normal blend so particles don't accumulate to white
  if (state.trail > 0) gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  gl.uniform2f(uRes, canvas.width, canvas.height);
  gl.uniform1i(uPalette, paletteIndex[state.palette] ?? 0);
  gl.drawArrays(gl.POINTS, 0, N);

  // Restore additive blend for next frame
  if (state.trail > 0) gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
}

// --- Resize ---
function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = Math.round(window.innerWidth  * dpr);
  canvas.height = Math.round(window.innerHeight * dpr);
  canvas.style.width  = window.innerWidth  + 'px';
  canvas.style.height = window.innerHeight + 'px';
  gl.viewport(0, 0, canvas.width, canvas.height);
  initParticles();
}

window.addEventListener('resize', resize);

// --- Loop ---
let lastTs = 0;
function loop(ts) {
  const dt = Math.min((ts - lastTs) / 1000, 0.05);
  lastTs = ts;
  time += dt;


  tick(dt);
  buildArrays();
  render();
  requestAnimationFrame(loop);
}

// --- Controls ---
function bindControl(id, valId, key, transform) {
  const el = document.getElementById(id);
  const display = document.getElementById(valId);
  el.addEventListener('input', () => {
    state[key] = transform(el.value);
    if (display) display.textContent = el.value;
  });
}

bindControl('scale',          'scale-val',          'scale',         Number);
bindControl('speed',          'speed-val',          'speed',         Number);
bindControl('tightness',      'tightness-val',      'tightness',     Number);
bindControl('motion',         'motion-val',         'motion',        Number);
bindControl('cursor-strength','cursor-strength-val','cursorStrength', Number);
bindControl('wave-speed',     'wave-speed-val',     'waveSpeed',     Number);
bindControl('trail',          'trail-val',          'trail',         Number);

document.getElementById('count').addEventListener('input', e => {
  const display = document.getElementById('count-val');
  if (display) display.textContent = e.target.value;
  reinitAll(Number(e.target.value));
});

document.querySelectorAll('.palette-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.palette-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    state.palette = card.dataset.value;
  });
});

document.querySelectorAll('.shape-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.shape-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    state.shape = card.dataset.value;
    const W = canvas.width, H = canvas.height;
    if (state.shape === 'murmuration') {
      flockCx = W / 2; flockCy = H / 2; flockVx = 0; flockVy = 0;
      murmuEnergy = 0; murmuEnergyPrev = 0; murmuEnergyDot = 0; sweepPhase = 0; waveAngle = Math.PI * 0.25; vortexBlend = 0;
      pickNewFlockTarget(W, H);
      document.getElementById('flock-section').style.display = 'flex';
    } else {
      document.getElementById('flock-section').style.display = 'none';
      state.murmuMotion = 'off';
      document.querySelectorAll('.flock-card').forEach(c => c.classList.toggle('active', c.dataset.value === 'off'));
    }
    applyHomes(generateHomes(W, H));
  });
});

document.getElementById('frequency').addEventListener('input', e => {
  const display = document.getElementById('frequency-val');
  if (display) display.textContent = e.target.value;
  state.frequency = Number(e.target.value);
  // Regenerate homes for wave shapes since formation depends on frequency
  if (['wave', 'harmonic', 'lissajous', 'murmuration'].includes(state.shape)) {
    applyHomes(generateHomes(canvas.width, canvas.height));
  }
});

document.querySelectorAll('.flock-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.flock-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    state.murmuMotion = card.dataset.value;
  });
});

document.getElementById('save-btn').addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = `visual-noise-${Date.now()}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
});

// --- Init ---
resize();
requestAnimationFrame(loop);
