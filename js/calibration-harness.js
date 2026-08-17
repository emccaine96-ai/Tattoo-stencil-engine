/**
 * Batch calibration harness skeleton
 * Runs a list of {source, geminiTarget} pairs through the classical engine,
 * scores each with edge-overlap F-score, and reports aggregate statistics.
 *
 * This is the piece needed before you can efficiently tune the 20–50 examples.
 */

/**
 * @param {Object} engine - instance that has processImage(source, settings)
 * @param {Array} pairs - [{ source: Image|Canvas|dataURL, gemini: Image|Canvas|dataURL, id?: string }]
 * @param {Object} settings - parameter set to test
 * @param {Object} scoreOptions - passed to scoreStencilPair
 */
async function runBatchCalibration(engine, pairs, settings = {}, scoreOptions = {}) {
  const results = [];
  let totalF1 = 0;

  for (let i = 0; i < pairs.length; i++) {
    const pair = pairs[i];
    const id = pair.id || `pair-${i}`;

    try {
      const predDataUrl = engine.processImage(pair.source, settings);
      const scores = await window.StencilFScore.scoreStencilPair(predDataUrl, pair.gemini, scoreOptions);

      results.push({
        id,
        ...scores,
        success: true
      });
      totalF1 += scores.f1;
    } catch (err) {
      results.push({
        id,
        success: false,
        error: err.message
      });
    }
  }

  const successful = results.filter(r => r.success);
  const meanF1 = successful.length > 0 ? totalF1 / successful.length : 0;

  return {
    results,
    meanF1,
    count: pairs.length,
    successCount: successful.length
  };
}

/**
 * Very simple coordinate-descent style optimizer (one parameter at a time).
 * Good enough for 8 parameters and 20–50 examples.
 */
async function simpleCoordinateDescent(engine, pairs, initialSettings, paramRanges, scoreOptions = {}) {
  let bestSettings = { ...initialSettings };
  let bestScore = -Infinity;

  // Evaluate initial
  const initial = await runBatchCalibration(engine, pairs, bestSettings, scoreOptions);
  bestScore = initial.meanF1;
  console.log('Initial mean F1:', bestScore.toFixed(4));

  for (const [param, range] of Object.entries(paramRanges)) {
    const [min, max, steps] = range;
    let localBest = bestSettings[param];
    let localBestScore = bestScore;

    for (let s = 0; s <= steps; s++) {
      const value = min + (max - min) * (s / steps);
      const trial = { ...bestSettings, [param]: value };
      const result = await runBatchCalibration(engine, pairs, trial, scoreOptions);

      if (result.meanF1 > localBestScore) {
        localBestScore = result.meanF1;
        localBest = value;
      }
    }

    bestSettings[param] = localBest;
    bestScore = localBestScore;
    console.log(`Best ${param} = ${localBest}  (mean F1 ${bestScore.toFixed(4)})`);
  }

  return { bestSettings, bestScore };
}

if (typeof window !== 'undefined') {
  window.StencilCalibration = { runBatchCalibration, simpleCoordinateDescent };
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { runBatchCalibration, simpleCoordinateDescent };
}
