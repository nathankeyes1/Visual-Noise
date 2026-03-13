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
gl.clearColor(0.04, 0.04, 0.04, 1.0);

// --- State ---
let time = 0;

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
};

const paletteIndex = { mono: 0, ice: 1, ember: 2, acid: 3, dusk: 4 };

// --- Shaders ---
const VS = `
attribute vec2 a_position;
attribute float a_t;
uniform vec2 u_resolution;
varying float v_t;
void main() {
  vec2 clip = (a_position / u_resolution) * 2.0 - 1.0;
  clip.y = -clip.y;
  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = 5.0;
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

vec3 applyPalette(float t, int p) {
  if (p == 1) return palette_ice(t);
  if (p == 2) return palette_ember(t);
  if (p == 3) return palette_acid(t);
  if (p == 4) return palette_dusk(t);
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

// --- Particle data ---
let N = 800;
let posArr = new Float32Array(N * 2);
let tArr   = new Float32Array(N);
let particles = new Array(N);
const posBuf = gl.createBuffer();
const tBuf   = gl.createBuffer();

const aPos = gl.getAttribLocation(prog, 'a_position');
const aTee  = gl.getAttribLocation(prog, 'a_t');
const uRes     = gl.getUniformLocation(prog, 'u_resolution');
const uPalette = gl.getUniformLocation(prog, 'u_palette');

gl.enableVertexAttribArray(aPos);
gl.enableVertexAttribArray(aTee);

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

function generateHomes(W, H) {
  const cx = W / 2, cy = H / 2;
  switch (state.shape) {
    case 'square':    return generateSquare(cx, cy, W, H);
    case 'triangle':  return generateTriangle(cx, cy, W, H);
    case 'freeform':  return generateFreeform(cx, cy, W, H);
    case 'wave':      return generateWave(cx, cy, W, H);
    case 'harmonic':  return generateHarmonic(cx, cy, W, H);
    case 'lissajous': return generateLissajous(cx, cy, W, H);
    default:          return generateCircle(cx, cy, W, H);
  }
}

// --- Init particles ---
function initParticles() {
  const W = canvas.width, H = canvas.height;
  const homes = generateHomes(W, H);
  for (let i = 0; i < N; i++) {
    const seed = Math.random() * 1000;
    particles[i] = {
      x:  homes[i].hx,
      y:  homes[i].hy,
      vx: 0,
      vy: 0,
      hx: homes[i].hx,
      hy: homes[i].hy,
      bx: homes[i].bx ?? homes[i].hx,
      by: homes[i].by ?? homes[i].hy,
      seed,
    };
  }
}

function reinitAll(newN) {
  N = newN;
  posArr    = new Float32Array(N * 2);
  tArr      = new Float32Array(N);
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
  const isWave    = ['wave','harmonic','lissajous'].includes(state.shape);
  const freq      = Math.max(1, Math.min(8, Math.round(state.frequency)));
  const ampY      = H * 0.22;

  cursor.smoothedSpeed *= 0.92;

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
      } else { // lissajous
        const [a, b] = LISSAJOUS_RATIOS[freq];
        const R = Math.min(W, H) * 0.36;
        p.hx = cx + Math.sin(a * p.bx + phase) * R;
        p.hy = cy + Math.sin(b * p.bx) * R;
      }
    }

    // Spring
    const sfx = (p.hx - p.x) * k;
    const sfy = (p.hy - p.y) * k;

    // Noise drift
    const nx = noiseXY(p.x * nScale + nTime + p.seed,      p.y * nScale);
    const ny = noiseXY(p.x * nScale + p.seed + 31.7,       p.y * nScale + nTime + 17.3);
    const nfx = nx * motionAmp;
    const nfy = ny * motionAmp;

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

    // Integrate
    const ax = sfx + nfx + cfx;
    const ay = sfy + nfy + cfy;
    p.vx = (p.vx + ax * dt) * 0.88;
    p.vy = (p.vy + ay * dt) * 0.88;
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
    } else if (state.shape === 'harmonic') {
      // Lobes glow, nodes dark; pulses with waveSpeed
      const n = freq - 1;
      const psi0 = qhoPsi(n, p.by);
      const psi1 = qhoPsi(Math.min(n + 1, 7), p.by);
      const envelope = psi0 * Math.cos(phase) + psi1 * Math.sin(phase);
      tVal = Math.min(Math.abs(envelope) / 1.5, 1.0);
    } else if (state.shape === 'lissajous') {
      // Rainbow gradient wraps around the figure
      tVal = p.bx / (2 * Math.PI);
    } else {
      const noiseVal = noiseXY(p.x * 0.003 + time * 0.05 + p.seed, p.y * 0.003) * 0.5 + 0.5;
      const disp = Math.min(Math.hypot(p.x - p.hx, p.y - p.hy) / 80.0, 1.0);
      tVal = Math.max(noiseVal * 0.6, disp);
    }
    tArr[i] = tVal;
  }
}

// --- Render ---
function render() {
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, posArr, gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, tBuf);
  gl.bufferData(gl.ARRAY_BUFFER, tArr, gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(aTee, 1, gl.FLOAT, false, 0, 0);

  gl.uniform2f(uRes, canvas.width, canvas.height);
  gl.uniform1i(uPalette, paletteIndex[state.palette] ?? 0);
  gl.drawArrays(gl.POINTS, 0, N);
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
    const homes = generateHomes(W, H);
    for (let i = 0; i < N; i++) {
      particles[i].hx = homes[i].hx;
      particles[i].hy = homes[i].hy;
      particles[i].bx = homes[i].bx ?? homes[i].hx;
      particles[i].by = homes[i].by ?? homes[i].hy;
    }
  });
});

document.getElementById('frequency').addEventListener('input', e => {
  const display = document.getElementById('frequency-val');
  if (display) display.textContent = e.target.value;
  state.frequency = Number(e.target.value);
  // Regenerate homes for wave shapes since formation depends on frequency
  if (['wave', 'harmonic', 'lissajous'].includes(state.shape)) {
    const W = canvas.width, H = canvas.height;
    const homes = generateHomes(W, H);
    for (let i = 0; i < N; i++) {
      particles[i].hx = homes[i].hx;
      particles[i].hy = homes[i].hy;
      particles[i].bx = homes[i].bx ?? homes[i].hx;
      particles[i].by = homes[i].by ?? homes[i].hy;
    }
  }
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
