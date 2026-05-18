import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

app = FastAPI(title="Ultra Advanced Tattoo Stencil AI Engine")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def apply_white_background(img_rgb):
    """
    Ensure output image has a solid white background (no transparency).
    Input: RGB image (3 channels)
    Output: RGB image with white background
    """
    if len(img_rgb.shape) == 2:
        # Convert grayscale to RGB
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    
    # Create white background
    height, width = img_rgb.shape[:2]
    white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # For binary images, use the image as a mask
    if np.max(img_rgb) <= 1 or (np.min(img_rgb) == 0 and np.max(img_rgb) == 255):
        # Binary image - white stays white, black stays black
        mask = img_rgb[:, :, 0] if img_rgb.shape[2] == 3 else img_rgb
        result = np.where(mask[:, :, np.newaxis] == 255, 255, 0)
        result = np.stack([result[:, :, 0]] * 3, axis=2).astype(np.uint8)
        return result
    
    return img_rgb

def colorize_stencil(mask, color_rgb):
    """
    Apply a color to a binary mask while maintaining white background.
    mask: Binary image (0-255), where 255 is the line color
    color_rgb: Tuple (R, G, B) for line color
    Returns: RGB image with colored lines on white background
    """
    # Ensure mask is 2D grayscale
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    height, width = mask.shape
    result = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Where mask is not white (255), apply color
    non_white = mask < 255
    result[non_white] = color_rgb
    
    return result

def bayer_dither_matrix(size):
    """
    Generate Bayer dither matrix for ordered dithering.
    """
    if size == 2:
        return np.array([[0, 2], [3, 1]], dtype=np.uint8)
    elif size == 4:
        return np.array([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ], dtype=np.uint8)
    else:  # 8x8
        return np.array([
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ], dtype=np.uint8)

def ordered_dither(image, levels=4):
    """
    Ordered dithering using Bayer matrix for superior halftone quality.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    h, w = image.shape
    threshold_max = 256 // levels
    
    bayer = bayer_dither_matrix(4) / 16.0  # Normalize 4x4 Bayer
    
    result = np.zeros_like(image)
    
    for y in range(h):
        for x in range(w):
            bayer_val = bayer[y % 4, x % 4] * 255
            pixel = image[y, x]
            if pixel > bayer_val:
                result[y, x] = 255
    
    return result

def adaptive_halftone(image, tone_levels=6):
    """
    Advanced halftone with adaptive tone levels for smooth gradient rendering.
    Produces dot patterns that vary in density to represent gray levels.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    h, w = image.shape
    image_float = image.astype(np.float32) / 255.0
    
    result = np.ones((h, w), dtype=np.uint8) * 255
    
    cell_size = 3
    dot_radius_max = cell_size // 2
    
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            # Get average tone in this cell
            cell = image_float[y:min(y+cell_size, h), x:min(x+cell_size, w)]
            avg_tone = np.mean(cell)
            
            # Map tone to dot radius (darker = bigger dot)
            dot_radius = int(dot_radius_max * (1.0 - avg_tone))
            
            if dot_radius > 0:
                cy = y + cell_size // 2
                cx = x + cell_size // 2
                cv2.circle(result, (cx, cy), dot_radius, 0, -1)
    
    return result

def filter_classic(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    Fine Linework: Crisp, tight edge tracing using adaptive thresholding and Canny detection.
    Mimics a perfect, thin 3RL lining pass with clean, isolated line work.
    """
    # Noise reduction via bilateral filtering
    spatial_sigma = max(1, int(noise_reduction * 8))
    color_sigma = max(1, int(noise_reduction * 20))
    filtered = cv2.bilateralFilter(img_gray, 9, color_sigma, spatial_sigma)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=contrast_boost * 4, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    
    # Canny edge detection for crisp lines
    low_thresh = max(50, int(50 / (contrast_boost + 0.5)))
    high_thresh = max(150, int(150 / (contrast_boost + 0.5)))
    edges = cv2.Canny(enhanced, low_thresh, high_thresh)
    
    # Dilate slightly to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Remove isolated noise (small components)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    clean_edges = np.zeros_like(edges)
    min_area = max(5, int(20 * (1 / (noise_reduction + 0.1))))
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(clean_edges, [contour], 0, 255, 1)
    
    return 255 - clean_edges

def filter_sketchy(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    PROFESSIONAL CROSS-HATCH SHADING: Advanced multi-scale gradient detection
    with ultra-sensitive mid-tone capture for portrait detail preservation.
    Maps 45° lines over light/mid shadows, 135° over deep shadows with variable density.
    """
    # Step 1: Multi-scale edge detection (Sobel + Laplacian hybrid)
    filtered = cv2.bilateralFilter(img_gray, 11, 25, 15)
    clahe = cv2.createCLAHE(clipLimit=contrast_boost * 3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    
    # Multi-scale gradient computation
    sobelx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Laplacian for secondary edge information
    laplacian = cv2.Laplacian(enhanced, cv2.CV_32F)
    laplacian_mag = np.abs(laplacian)
    
    # Combine gradients
    combined_grad = (sobel_mag + laplacian_mag * 0.5)
    grad_normalized = cv2.normalize(combined_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Step 2: ULTRA-SENSITIVE mid-tone shadow detection (key improvement)
    # Capture FULL range of grays, not just extremes
    inverted = 255 - enhanced
    
    # Create comprehensive shadow bands (captures all portrait tones)
    light_shadow = cv2.inRange(inverted, 15, 80)      # Very light grays (light shading)
    mid_shadow = cv2.inRange(inverted, 81, 150)       # Mid-tone grays (medium shading) - CRITICAL RANGE
    deep_shadow = cv2.inRange(inverted, 151, 220)     # Deep shadows
    
    # Step 3: Generate adaptive cross-hatch patterns
    height, width = img_gray.shape
    hatch_45_light = np.zeros_like(img_gray)
    hatch_45_mid = np.zeros_like(img_gray)
    hatch_135_deep = np.zeros_like(img_gray)
    
    # Adaptive line spacing based on shadow density and noise_reduction
    base_spacing = max(2, int(4 / (noise_reduction + 0.3)))
    light_spacing = base_spacing + 1  # Thinner in light areas
    mid_spacing = base_spacing       # Medium density
    deep_spacing = max(2, base_spacing - 1)  # Denser in deep shadows
    
    # 45-degree hatching (light and mid-tone areas)
    for i in range(0, height + width, light_spacing):
        cv2.line(hatch_45_light, (max(0, i - height), 0), (i, min(width, i)), 255, 1)
    
    for i in range(0, height + width, mid_spacing):
        cv2.line(hatch_45_mid, (max(0, i - height), 0), (i, min(width, i)), 255, 1)
    
    # 135-degree hatching (deep shadows)
    for i in range(-width, height, deep_spacing):
        cv2.line(hatch_135_deep, (0, i), (width, i + width), 255, 1)
    
    # Step 4: Apply hatching to adaptive shadow regions
    light_hatch = cv2.bitwise_and(hatch_45_light, hatch_45_light, mask=light_shadow)
    mid_hatch = cv2.bitwise_and(hatch_45_mid, hatch_45_mid, mask=mid_shadow)
    deep_hatch = cv2.bitwise_and(hatch_135_deep, hatch_135_deep, mask=deep_shadow)
    
    # Step 5: Extract clean contours
    _, contour_base = cv2.threshold(enhanced, int(180 * (1/contrast_boost)), 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(contour_base, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    clean_contours = np.zeros_like(contour_base)
    for contour in contours:
        if cv2.contourArea(contour) > max(10, int(30 * (1 / (noise_reduction + 0.2)))):
            cv2.drawContours(clean_contours, [contour], 0, 255, 1)
    
    # Step 6: Combine hatching with contours
    result = cv2.bitwise_or(light_hatch, mid_hatch)
    result = cv2.bitwise_or(result, deep_hatch)
    result = cv2.bitwise_or(result, clean_contours)
    
    # Step 7: Clean up with minimal morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return 255 - result

def filter_smooth(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    PROFESSIONAL STIPPLE/WHIP SHADING: Adaptive halftone with micro-dot texturing.
    Captures full gradient range with ordered dithering for superior mid-tone handling.
    Produces smooth, professional stipple patterns that mimic whip shading technique.
    """
    # Step 1: Bilateral filtering with edge preservation
    filtered = cv2.bilateralFilter(img_gray, 11, 25, 15)
    
    # Step 2: Enhanced contrast with adaptive CLAHE
    clahe = cv2.createCLAHE(clipLimit=contrast_boost * 4, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    
    # Step 3: Invert for stipple generation (darker areas get more dots)
    inverted = 255 - enhanced
    
    # Step 4: ULTRA-SENSITIVE ordered dithering for precise halftone
    # Bayer matrix ordered dithering captures all tone levels
    bayer_halftone = ordered_dither(inverted, levels=int(4 + noise_reduction * 2))
    
    # Step 5: Adaptive halftone with variable dot sizing
    # Creates larger dots in shadows, smaller in lights
    adaptive_stipple = adaptive_halftone(inverted, tone_levels=int(6 + noise_reduction))
    
    # Step 6: Blend ordered dither with adaptive halftone
    # This creates superior mid-tone sensitivity
    blended = cv2.addWeighted(bayer_halftone, 0.6, adaptive_stipple, 0.4, 0)
    
    # Step 7: Extract clean portrait contours
    _, contour_base = cv2.threshold(enhanced, int(170 * (1/contrast_boost)), 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(contour_base, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    clean_contours = np.zeros_like(contour_base)
    min_area = max(10, int(40 * (1 / (noise_reduction + 0.2))))
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(clean_contours, [contour], 0, 255, 1)
    
    # Step 8: Combine stipple texture with structural contours
    result = cv2.bitwise_or(blended, clean_contours)
    
    # Step 9: Light morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return result

def filter_thermal_ink(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    Classic Purple Thermal Burn: High-saturation deep purple lines with subtle heat-spread expansion.
    Mimics a traditional Spirit stencil fax copy with thermal bleed effect.
    """
    # Enhance edges
    clahe = cv2.createCLAHE(clipLimit=contrast_boost * 4, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    
    # Edge detection
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Dilate to create thermal spread effect
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(line_weight * 2 + 1), int(line_weight * 2 + 1)))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Bilateral filter for smoothness
    filtered = cv2.bilateralFilter(img_gray, 9, 20, 8)
    _, stencil = cv2.threshold(filtered, int(128 * contrast_boost), 255, cv2.THRESH_BINARY_INV)
    
    # Combine edges with base stencil
    result = cv2.bitwise_or(dilated, stencil)
    
    return 255 - result

def filter_carbon_transfer(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    Hand-Traced Carbon Paper: Textured, organic purple lines mimicking heavy physical hand-pressure.
    Adds organic texture variation to simulate carbon paper transfer inconsistency.
    """
    # Bilateral filter for edge preservation with texture
    filtered = cv2.bilateralFilter(img_gray, 11, 25, 15)
    
    # Create texture by adding subtle random perturbation
    texture_noise = np.random.normal(0, int(noise_reduction * 3), filtered.shape).astype(np.float32)
    textured = np.clip(filtered.astype(np.float32) + texture_noise, 0, 255).astype(np.uint8)
    
    # Adaptive thresholding for organic-looking edges
    block_size = int(11 + (noise_reduction * 4))
    if block_size % 2 == 0:
        block_size += 1
    stencil = cv2.adaptiveThreshold(
        textured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, int(contrast_boost * 3)
    )
    
    # Apply morphological smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(stencil, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return result

def filter_bold(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    Heavy Traditional Outlines: Thick, high-saturation black lines.
    Mimics bold traditional american 11RL/14RL line passes.
    """
    # Strong contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=contrast_boost * 5, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    
    # Aggressive thresholding
    _, thresh = cv2.threshold(enhanced, int(140 * (1 / contrast_boost)), 255, cv2.THRESH_BINARY_INV)
    
    # Heavy dilation for thick lines
    kernel_size = int(5 + line_weight * 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(thresh, kernel, iterations=int(line_weight))
    
    # Close small gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    return result

def filter_sharp(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    Neo-Traditional / Geometric Precision: Clean vector-style lines with zero anti-aliasing.
    Optimized for crisp script and mandalas.
    """
    # Strong bilateral filter for edge preservation
    filtered = cv2.bilateralFilter(img_gray, 13, 30, 20)
    
    # High-contrast threshold
    _, thresh = cv2.threshold(filtered, int(127 * (1 / contrast_boost)), 255, cv2.THRESH_BINARY_INV)
    
    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate line width slightly for vector appearance
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(line_weight + 1), int(line_weight + 1)))
    dilated = cv2.dilate(opened, kernel_dilate, iterations=1)
    
    # Apply Laplacian sharpening for crisp edges
    laplacian = cv2.Laplacian(dilated, cv2.CV_32F)
    sharpened = cv2.convertScaleAbs(laplacian)
    
    result = cv2.bitwise_or(dilated, sharpened)
    
    return result

def filter_duotone_purple(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    Two-Tone Stencil Map: Dark shadows to rich dark violet, mid-tones to softer lilac.
    Creates depth through color mapping for enhanced tattoo design visualization.
    """
    # Enhance contrast to separate tones
    clahe = cv2.createCLAHE(clipLimit=contrast_boost * 4, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    
    # Create shadow/midtone masks with full range sensitivity
    _, dark_mask = cv2.threshold(enhanced, int(100 * contrast_boost), 255, cv2.THRESH_BINARY)
    _, light_mask = cv2.threshold(enhanced, int(150 * contrast_boost), 255, cv2.THRESH_BINARY_INV)
    mid_mask = cv2.bitwise_and(cv2.bitwise_not(dark_mask), light_mask)
    
    # Create grayscale stencil
    _, stencil = cv2.threshold(enhanced, int(128 * contrast_boost), 255, cv2.THRESH_BINARY_INV)
    
    # Create output image
    height, width = img_gray.shape
    result = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Dark violet (75, 0, 130) for dark areas
    dark_violet = (75, 0, 130)
    result[dark_mask == 0] = dark_violet
    
    # Light lilac (200, 162, 200) for mid areas
    light_lilac = (200, 162, 200)
    result[mid_mask == 255] = light_lilac
    
    return result

def filter_high_contrast(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    Blackwork / High Saturation: Crisp pure black and pure white with zero mid-tones.
    Perfect for solid tribal or heavy black fields with maximum visual impact.
    """
    # Strong contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=contrast_boost * 6, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    
    # Aggressive binary threshold with Otsu's method
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Apply dilation based on line weight
    if line_weight > 1.0:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(line_weight * 2 + 1), int(line_weight * 2 + 1)))
        opened = cv2.dilate(opened, kernel_dilate, iterations=1)
    
    return opened

def process_stencil_core(img_gray, style, line_weight, noise_reduction, contrast_boost):
    """
    Main processing router for all stencil styles.
    Applies professional-grade advanced image processing algorithms based on selected style.
    """
    # Ensure parameters are within reasonable ranges
    noise_reduction = np.clip(noise_reduction, 0.1, 3.0)
    contrast_boost = np.clip(contrast_boost, 0.5, 3.0)
    line_weight = np.clip(line_weight, 0.5, 5.0)
    
    if style == "classic":
        return filter_classic(img_gray, line_weight, noise_reduction, contrast_boost)
    
    elif style == "sketchy":
        return filter_sketchy(img_gray, line_weight, noise_reduction, contrast_boost)
    
    elif style == "smooth":
        return filter_smooth(img_gray, line_weight, noise_reduction, contrast_boost)
    
    elif style == "thermal_ink":
        return filter_thermal_ink(img_gray, line_weight, noise_reduction, contrast_boost)
    
    elif style == "carbon_transfer":
        return filter_carbon_transfer(img_gray, line_weight, noise_reduction, contrast_boost)
    
    elif style == "bold":
        return filter_bold(img_gray, line_weight, noise_reduction, contrast_boost)
    
    elif style == "sharp":
        return filter_sharp(img_gray, line_weight, noise_reduction, contrast_boost)
    
    elif style == "duotone_purple":
        # Special case: returns RGB directly
        return filter_duotone_purple(img_gray, line_weight, noise_reduction, contrast_boost)
    
    elif style == "high_contrast":
        return filter_high_contrast(img_gray, line_weight, noise_reduction, contrast_boost)
    
    else:
        # Default: classic
        return filter_classic(img_gray, line_weight, noise_reduction, contrast_boost)

@app.post("/api/v1/generate-stencil/")
async def generate_stencil(
    file: UploadFile = File(...),
    style: str = Form("classic"),
    line_weight: float = Form(2.0),
    noise_reduction: float = Form(1.0),
    contrast_boost: float = Form(1.2)
):
    """
    Professional-grade tattoo stencil generation endpoint.
    Processes portrait images into print-ready stencil designs using advanced algorithms.
    
    Parameters:
    - style: One of ['classic', 'sketchy', 'smooth', 'thermal_ink', 'carbon_transfer', 'bold', 'sharp', 'duotone_purple', 'high_contrast']
    - line_weight: Controls line thickness (0.5 - 5.0)
    - noise_reduction: Controls smoothing intensity (0.1 - 3.0)
    - contrast_boost: Controls contrast enhancement (0.5 - 3.0)
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Corrupted or invalid image file.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Process through appropriate filter
        processed = process_stencil_core(gray, style, line_weight, noise_reduction, contrast_boost)
        
        # Handle different output types
        if style == "duotone_purple":
            # Already RGB from duotone filter
            output_rgb = processed
        else:
            # Convert grayscale/binary to RGB with white background
            if len(processed.shape) == 2:
                # Binary stencil - convert to RGB
                output_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                output_rgb = processed
        
        # Ensure white background and solid output
        final_output = apply_white_background(output_rgb)
        
        # Encode to PNG
        _, encoded_png = cv2.imencode('.png', final_output)
        
        return Response(
            content=encoded_png.tobytes(),
            media_type="image/png"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "engine": "Ultra Advanced Tattoo Stencil AI Engine - Professional Grade"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
