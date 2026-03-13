const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// --- State ---
let width, height;
let time = 0;
let animId;

const state = {
  scale: 30,
  speed: 20,
  palette: 'mono',
  mode: 'flow',
};

// --- Palettes ---
// Each palette is a function: (t in [0,1]) => [r, g, b] (0-255)
const palettes = {
  mono: (t) => {
    const v = Math.round(t * 255);
    return [v, v, v];
  },
  ice: (t) => {
    // Deep navy → icy cyan → white
    const r = Math.round(lerp(10,  220, t));
    const g = Math.round(lerp(20,  240, t));
    const b = Math.round(lerp(80,  255, t));
    return [r, g, b];
  },
  ember: (t) => {
    // Black → deep red → orange → yellow
    const r = Math.round(t < 0.5 ? lerp(0, 200, t * 2) : lerp(200, 255, (t - 0.5) * 2));
    const g = Math.round(t < 0.5 ? 0 : lerp(0, 200, (t - 0.5) * 2));
    const b = Math.round(t < 0.2 ? 0 : lerp(0, 30, (t - 0.2) / 0.8));
    return [r, g, b];
  },
  acid: (t) => {
    // Dark purple → electric green → white
    const r = Math.round(t < 0.5 ? lerp(30, 0, t * 2) : lerp(0, 200, (t - 0.5) * 2));
    const g = Math.round(lerp(0, 255, t));
    const b = Math.round(t < 0.5 ? lerp(60, 40, t * 2) : lerp(40, 100, (t - 0.5) * 2));
    return [r, g, b];
  },
  dusk: (t) => {
    // Midnight → violet → coral → pale gold
    const r = Math.round(t < 0.5 ? lerp(10, 200, t * 2) : lerp(200, 255, (t - 0.5) * 2));
    const g = Math.round(t < 0.5 ? lerp(5,  60,  t * 2) : lerp(60,  210, (t - 0.5) * 2));
    const b = Math.round(t < 0.5 ? lerp(40, 120, t * 2) : lerp(120, 120, (t - 0.5) * 2));
    return [r, g, b];
  },
};

function lerp(a, b, t) {
  return a + (b - a) * Math.max(0, Math.min(1, t));
}

// --- Resize ---
function resize() {
  const dpr = window.devicePixelRatio || 1;
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';
  ctx.scale(dpr, dpr);
}

window.addEventListener('resize', () => {
  resize();
});

// --- Draw ---
function sampleNoise(nx, ny) {
  switch (state.mode) {
    case 'flow':    return SimplexNoise.fbm(nx, ny, 6);
    case 'domain':  return SimplexNoise.domainWarp(nx, ny);
    case 'ridged':  return SimplexNoise.ridged(nx, ny, 6);
    default:        return SimplexNoise.fbm(nx, ny, 6);
  }
}

function draw() {
  const imgData = ctx.createImageData(Math.round(width * (window.devicePixelRatio || 1)), Math.round(height * (window.devicePixelRatio || 1)));
  const data = imgData.data;
  const dpr = window.devicePixelRatio || 1;
  const pw = Math.round(width * dpr);
  const ph = Math.round(height * dpr);

  const scaleFactor = state.scale / 1000;
  const colorFn = palettes[state.palette] || palettes.mono;

  for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      const nx = (x / dpr) * scaleFactor + time * 0.3;
      const ny = (y / dpr) * scaleFactor;

      // noise in [-1, 1] → [0, 1]
      let n = sampleNoise(nx, ny);
      n = (n + 1) * 0.5;

      const [r, g, b] = colorFn(n);
      const i = (y * pw + x) * 4;
      data[i]     = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
}

// --- Loop ---
let lastTs = 0;
function loop(ts) {
  const dt = Math.min((ts - lastTs) / 1000, 0.05);
  lastTs = ts;
  time += dt * (state.speed / 100) * 3;
  draw();
  animId = requestAnimationFrame(loop);
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

bindControl('scale', 'scale-val', 'scale', Number);
bindControl('speed', 'speed-val', 'speed', Number);

document.getElementById('palette').addEventListener('change', (e) => {
  state.palette = e.target.value;
});

document.getElementById('mode').addEventListener('change', (e) => {
  state.mode = e.target.value;
});

document.getElementById('save-btn').addEventListener('click', () => {
  draw();
  const link = document.createElement('a');
  link.download = `visual-noise-${Date.now()}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
});

// --- Init ---
resize();
animId = requestAnimationFrame(loop);
