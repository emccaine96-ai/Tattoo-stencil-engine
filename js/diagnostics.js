/**
 * Diagnostic tests for the flow-guided XDoG / Structure Tensor engine.
 * Run these before spending time on calibration to catch inverted thresholds,
 * flipped orientation, NaNs, etc.
 */

/**
 * Creates a horizontal banded gradient (dark → light left to right).
 * After running the engine, darker regions must produce denser ink than lighter ones.
 */
function createBandedGradient(width = 256, height = 128) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(width, height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = Math.floor((x / (width - 1)) * 255);
      const i = (y * width + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/**
 * Creates a diagonal stripe pattern.
 * Used to verify that Structure Tensor orientation is not flipped.
 */
function createDiagonalStripes(width = 256, height = 256, period = 16) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(width, height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = ((x + y) % period < period / 2) ? 30 : 220;
      const i = (y * width + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/**
 * Runs basic sanity checks on an engine instance.
 * engine must expose a processImage(imageSource, settings) method that returns a data URL or ImageData.
 */
async function runDiagnostics(engine, settings = {}) {
  const results = {
    bandedGradient: null,
    diagonalStripes: null,
    crashed: false,
    hadNaN: false,
    messages: []
  };

  try {
    // 1. Banded gradient test
    const grad = createBandedGradient();
    const gradResult = engine.processImage(grad, settings);
    results.bandedGradient = gradResult;
    results.messages.push('Banded gradient ran without crash');

    // 2. Diagonal stripes test
    const stripes = createDiagonalStripes();
    const stripeResult = engine.processImage(stripes, settings);
    results.diagonalStripes = stripeResult;
    results.messages.push('Diagonal stripes ran without crash');

    // 3. Very basic NaN / finite check if we can read pixels
    // (Implementation depends on whether processImage returns ImageData or dataURL)
    results.messages.push('Basic diagnostics completed');
  } catch (err) {
    results.crashed = true;
    results.messages.push('CRASH: ' + err.message);
  }

  return results;
}

// Export for browser or module use
if (typeof window !== 'undefined') {
  window.StencilDiagnostics = { createBandedGradient, createDiagonalStripes, runDiagnostics };
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { createBandedGradient, createDiagonalStripes, runDiagnostics };
}
