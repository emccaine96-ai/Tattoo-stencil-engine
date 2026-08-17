# Classical Pro Engine

Fully client-side tattoo stencil engine designed to sit alongside the Gemini path.

## Pipeline

1. CLAHE (lighting correction)
2. Bilateral-style smoothing (skin)
3. **XDoG** (line + whip) **or** **Floyd-Steinberg Dither** (pure thermal dotwork)
4. Optional Structure Tensor flow modulation (Portrait / Hair / Fur)
5. Morphological line weight
6. Hectograph purple or pure black transparent output

## Files

- `ClassicalProEngine.js` – the engine
- `presets.json` – 50+ subject-type starting points (babies, elderly, florals, geometric, mythology, stippling, animals, traditional, bad photos, lettering)

## Quick usage

```js
const engine = new ClassicalProEngine(document.getElementById('c'));

// Use a preset
const preset = presets.engine_calibration_presets['5_mythology_statues_stone'][0]; // Poseidon
const dataUrl = engine.processImage(img, {
  useClahe: true,
  shadingMode: 'xdog',          // or 'dither'
  useStructureTensor: true,     // for portraits / hair
  outputPurple: true
}, preset);
```

## Engine choice in the app

Expose three options to the user:

- **Classical** (this engine) – free / cheap, local, instant
- **Gemini** – highest artistic quality
- **Hybrid** (future) – classical base + Gemini refinement

## Subscription mapping suggestion

- Free: 10 classical generations
- $7.99: Unlimited classical
- $15.99: Classical + Gemini + Hybrid
- Master Pro: Everything + Bring-your-own OpenRouter key
