/**
 * Edge-overlap F-score metric for comparing classical engine output vs Gemini target.
 * Recommended by Claude for stencil calibration.
 *
 * Both images are reduced to binary ink / no-ink,
 * optionally dilated by 1–2 px for tolerance,
 * then precision / recall / F1 are computed on overlapping ink pixels.
 */

/**
 * Convert ImageData (or canvas) to binary ink mask.
 * Ink = 1 where luminance < threshold, else 0.
 */
function toBinaryMask(imageData, threshold = 128) {
  const { width, height, data } = imageData;
  const mask = new Uint8Array(width * height);

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    mask[p] = lum < threshold ? 1 : 0;
  }
  return { mask, width, height };
}

/**
 * Simple morphological dilate by radius (square kernel).
 */
function dilateMask(maskObj, radius = 1) {
  if (radius <= 0) return maskObj;
  const { mask, width, height } = maskObj;
  const out = new Uint8Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let val = 0;
      for (let dy = -radius; dy <= radius && !val; dy++) {
        for (let dx = -radius; dx <= radius && !val; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            if (mask[ny * width + nx]) val = 1;
          }
        }
      }
      out[y * width + x] = val;
    }
  }
  return { mask: out, width, height };
}

/**
 * Compute precision, recall, F1 between prediction mask and ground-truth mask.
 */
function edgeOverlapFScore(predMaskObj, gtMaskObj, dilateRadius = 1) {
  const pred = dilateRadius > 0 ? dilateMask(predMaskObj, dilateRadius) : predMaskObj;
  const gt = dilateRadius > 0 ? dilateMask(gtMaskObj, dilateRadius) : gtMaskObj;

  let tp = 0, fp = 0, fn = 0;
  const len = pred.mask.length;

  for (let i = 0; i < len; i++) {
    const p = pred.mask[i];
    const g = gt.mask[i];
    if (p && g) tp++;
    else if (p && !g) fp++;
    else if (!p && g) fn++;
  }

  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall    = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1        = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return { precision, recall, f1, tp, fp, fn };
}

/**
 * Convenience: score two canvases / ImageData / dataURLs.
 * Returns { precision, recall, f1 }
 */
async function scoreStencilPair(predSource, gtSource, options = {}) {
  const threshold = options.threshold ?? 128;
  const dilateRadius = options.dilateRadius ?? 1;

  const toImageData = async (src) => {
    if (src instanceof ImageData) return src;
    if (src instanceof HTMLCanvasElement) {
      return src.getContext('2d').getImageData(0, 0, src.width, src.height);
    }
    // assume dataURL or image element
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = img.width;
        c.height = img.height;
        const ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0);
        resolve(ctx.getImageData(0, 0, c.width, c.height));
      };
      img.src = typeof src === 'string' ? src : src.src;
    });
  };

  const predImg = await toImageData(predSource);
  const gtImg   = await toImageData(gtSource);

  const predMask = toBinaryMask(predImg, threshold);
  const gtMask   = toBinaryMask(gtImg, threshold);

  return edgeOverlapFScore(predMask, gtMask, dilateRadius);
}

if (typeof window !== 'undefined') {
  window.StencilFScore = { toBinaryMask, dilateMask, edgeOverlapFScore, scoreStencilPair };
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { toBinaryMask, dilateMask, edgeOverlapFScore, scoreStencilPair };
}
