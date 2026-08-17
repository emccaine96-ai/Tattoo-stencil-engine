# Portrait Pro Calibration Layer

This folder contains the pieces needed to calibrate the flow-guided XDoG / Structure Tensor engine against 20–50 Gemini examples.

## Files

- `diagnostics.js` – Quick tests (banded gradient + diagonal stripes) to verify the math is not inverted before calibration.
- `fscore.js` – Edge-overlap F-score metric (binary ink masks + optional dilation tolerance).
- `calibration-harness.js` – Batch runner + simple coordinate-descent optimizer.

## Recommended Order

1. Run diagnostics on a few images to confirm the engine is directionally correct.
2. Prepare 20–50 **paired** examples (same source photo → Gemini output + classical output).
3. Use `runBatchCalibration` to score a parameter set.
4. Use `simpleCoordinateDescent` (or any optimizer you prefer) to improve the parameters.

## Example usage (browser)

```js
// After loading the engine + these scripts
const engine = new TattooStencilEngine(canvas);

// Diagnostics
const diag = await StencilDiagnostics.runDiagnostics(engine);
console.log(diag.messages);

// Scoring one pair
const score = await StencilFScore.scoreStencilPair(classicalDataUrl, geminiDataUrl, {
  threshold: 128,
  dilateRadius: 1
});
console.log(score); // { precision, recall, f1, ... }

// Batch
const report = await StencilCalibration.runBatchCalibration(engine, pairs, currentSettings);
console.log('Mean F1:', report.meanF1);
```

## Notes

- Keep the parameter surface small (the 8–10 most influential knobs).
- Edge-overlap F-score is deliberately simple and robust for stencil line work.
- Coordinate descent is enough at this data scale; no need for Bayesian optimization yet.
