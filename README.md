# Visual-Noise

An interactive generative art visualization built with vanilla JavaScript and WebGL. Particles form shapes, flock like starlings, and respond to your cursor in real time — no frameworks, no build tools, no dependencies.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/nathanke/Visual-Noise.git
cd Visual-Noise

# Option 1: Open directly in your browser
open index.html          # macOS
xdg-open index.html      # Linux
start index.html         # Windows

# Option 2: Run a local server (avoids some browser quirks)
python3 -m http.server 8000
# Then open http://localhost:8000
```

That's it. No `npm install`, no build step. Just HTML, CSS, and JS.

## What You'll See

A full-screen canvas of animated particles. By default it opens in **Murmuration** mode — particles flock together like a swarm of starlings, rippling with energy waves and responding to your mouse.

### Controls

The floating control bar at the bottom (or stacked below the canvas on mobile) lets you tweak the visualization:

| Control | What it does |
|---------|-------------|
| **Wave** | Sinusoidal traveling wave with colored crests |
| **Lissajous** (∞) | Mathematical Lissajous curves with rainbow gradients |
| **Murmuration** (∿) | Bird-flocking simulation with energy dynamics (default) |
| **Frequency** | Number of oscillations in wave/curve shapes |
| **Wave Speed** | How fast the animation phase moves |
| **Tightness** | How strongly particles snap back to their home positions — turn it down for looser, more fluid motion |

### Interaction

- **Move your mouse** (or finger on mobile) over the canvas to push particles around
- Particles are pulled back to their shape by a spring force, so they'll settle back when you stop

## Project Structure

```
Visual-Noise/
├── index.html    # Single-page app — canvas + controls
├── main.js       # All the physics, math, and WebGL rendering (~1250 lines)
├── style.css     # Responsive layout — mobile-first with floating pill on desktop
└── README.md
```

Everything lives in `main.js`:
- **Particle physics** — spring forces, velocity damping, cursor interaction
- **Shape generators** — wave, Lissajous, circle, square, triangle, Hermite polynomials, Hopf fibration
- **Murmuration system** — boids flocking (separation/alignment/cohesion), disturbance grid with wave propagation, multi-oscillator energy envelope, phantom predator
- **WebGL pipeline** — vertex/fragment shaders, additive blending, motion trails, 6 color palettes

## Hacking on It

Since there's no build step, the feedback loop is instant: edit a file, refresh the browser.

### Things to try

**Change the particle count** — find `PARTICLE_COUNT` near the top of `main.js` and bump it up (try 4000 or 10000). Performance depends on your GPU.

**Try different color palettes** — search for `PALETTES` in `main.js`. There are 6 built-in palettes (Mono, Ice, Ember, Acid, Dusk, Aurora). The active palette is set via `currentPalette`.

**Tweak the murmuration physics** — look for these constants in `main.js`:
- `BOIDS_SEP`, `BOIDS_ALI`, `BOIDS_COH` — separation, alignment, and cohesion weights
- `PREDATOR_INTERVAL_MIN/MAX` — how often the phantom predator appears
- The energy envelope uses three oscillators with periods of 7, 17, and 47 seconds — change these for different rhythms

**Add a new shape** — shapes are just functions that return `[x, y]` positions for each particle. Look at how `computeWavePositions` or `computeLissajousPositions` work, then add your own.

**Switch blending modes** — the default is additive blending (`gl.blendFunc(gl.SRC_ALPHA, gl.ONE)`). Try `gl.ONE_MINUS_SRC_ALPHA` as the second argument for standard alpha blending — gives a very different look.

## Requirements

- A modern browser with WebGL support (Chrome, Firefox, Safari, Edge)
- Works on desktop and mobile (touch supported)
- No internet connection needed after cloning

## License

MIT
