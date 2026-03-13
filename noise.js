/**
 * Simplex Noise (2D + 3D)
 * Based on Stefan Gustavson's public domain implementation.
 */
(function (global) {
  const F2 = 0.5 * (Math.sqrt(3) - 1);
  const G2 = (3 - Math.sqrt(3)) / 6;

  const grad3 = [
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
  ];

  const perm = new Uint8Array(512);
  const gradP = new Array(512);

  function seed(s) {
    let v = s & 0xffffffff;
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    for (let i = 255; i > 0; i--) {
      v = (v ^ (v >>> 16)) * 0x45d9f3b | 0;
      v = (v ^ (v >>> 16)) * 0x45d9f3b | 0;
      v ^= v >>> 16;
      const j = ((v >>> 0) % (i + 1));
      [p[i], p[j]] = [p[j], p[i]];
    }
    for (let i = 0; i < 512; i++) {
      perm[i] = p[i & 255];
      gradP[i] = grad3[perm[i] % 12];
    }
  }

  seed(Date.now());

  function dot2(g, x, y) {
    return g[0] * x + g[1] * y;
  }

  function noise2(xin, yin) {
    const s = (xin + yin) * F2;
    const i = Math.floor(xin + s);
    const j = Math.floor(yin + s);
    const t = (i + j) * G2;
    const x0 = xin - (i - t);
    const y0 = yin - (j - t);

    let i1, j1;
    if (x0 > y0) { i1 = 1; j1 = 0; }
    else          { i1 = 0; j1 = 1; }

    const x1 = x0 - i1 + G2;
    const y1 = y0 - j1 + G2;
    const x2 = x0 - 1 + 2 * G2;
    const y2 = y0 - 1 + 2 * G2;

    const ii = i & 255;
    const jj = j & 255;

    let n0, n1, n2;
    let t0 = 0.5 - x0 * x0 - y0 * y0;
    n0 = t0 < 0 ? 0 : (t0 *= t0, t0 * t0 * dot2(gradP[ii + perm[jj]], x0, y0));

    let t1 = 0.5 - x1 * x1 - y1 * y1;
    n1 = t1 < 0 ? 0 : (t1 *= t1, t1 * t1 * dot2(gradP[ii + i1 + perm[jj + j1]], x1, y1));

    let t2 = 0.5 - x2 * x2 - y2 * y2;
    n2 = t2 < 0 ? 0 : (t2 *= t2, t2 * t2 * dot2(gradP[ii + 1 + perm[jj + 1]], x2, y2));

    return 70 * (n0 + n1 + n2); // [-1, 1]
  }

  // Fractal Brownian Motion — layers of noise octaves
  function fbm(x, y, octaves = 5, lacunarity = 2.0, gain = 0.5) {
    let value = 0;
    let amplitude = 0.5;
    let frequency = 1;
    for (let i = 0; i < octaves; i++) {
      value += amplitude * noise2(x * frequency, y * frequency);
      frequency *= lacunarity;
      amplitude *= gain;
    }
    return value; // roughly [-1, 1]
  }

  // Ridged noise — emphasises ridges/crests
  function ridged(x, y, octaves = 5) {
    let value = 0;
    let amplitude = 0.5;
    let frequency = 1;
    for (let i = 0; i < octaves; i++) {
      const n = noise2(x * frequency, y * frequency);
      value += amplitude * (1 - Math.abs(n));
      frequency *= 2;
      amplitude *= 0.5;
    }
    return value * 2 - 1; // roughly [-1, 1]
  }

  // Domain-warped noise: q = fbm(p), r = fbm(p + q), return fbm(p + r)
  function domainWarp(x, y) {
    const qx = fbm(x + 0.0, y + 0.0);
    const qy = fbm(x + 5.2, y + 1.3);
    const rx = fbm(x + qx * 4.0, y + qy * 4.0);
    const ry = fbm(x + qx * 4.0 + 1.7, y + qy * 4.0 + 9.2);
    return fbm(x + rx * 4.0, y + ry * 4.0);
  }

  global.SimplexNoise = { noise2, fbm, ridged, domainWarp, seed };
})(window);
