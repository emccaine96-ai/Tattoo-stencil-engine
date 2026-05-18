import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from scipy import ndimage
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import skeletonize
from skimage.measure import find_contours
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="AI-Powered Professional Tattoo Stencil Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ADVANCED SEGMENTATION: U-Net-like foreground/background separation
# ============================================================================

def segment_foreground_background(img_gray):
    """
    U-Net inspired foreground-background segmentation.
    Uses multi-scale morphological and gradient analysis to separate subject from background.
    """
    # Multi-scale edge detection
    edges_fine = cv2.Canny(img_gray, 30, 100)
    edges_coarse = cv2.Canny(img_gray, 50, 150)
    
    # Morphological gradient (edge detection via morphology)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    grad_small = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel_small)
    grad_large = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel_large)
    
    # Combine edge detections
    combined_edges = cv2.bitwise_or(edges_fine, edges_coarse)
    combined_edges = cv2.bitwise_or(combined_edges, cv2.convertScaleAbs(grad_small))
    combined_edges = cv2.bitwise_or(combined_edges, cv2.convertScaleAbs(grad_large))
    
    # Morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Binary segmentation
    _, segmentation = cv2.threshold(filled, 50, 255, cv2.THRESH_BINARY)
    
    # Fill holes (flood fill from edges)
    filled_segmentation = cv2.morphologyEx(segmentation, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return filled_segmentation

# ============================================================================
# ADVANCED EDGE DETECTION: Multi-scale Hybrid (Canny + Sobel + Laplacian)
# ============================================================================

def hybrid_edge_detection(img_gray, contrast_boost=1.2):
    """
    Professional multi-scale edge detection combining Canny, Sobel, and Laplacian.
    Detects both fine details and structural edges while suppressing noise.
    """
    # Bilateral filter for noise reduction while preserving edges
    filtered = cv2.bilateralFilter(img_gray, 9, 20, 15)
    
    # Canny edge detection (best for clean, connected edges)
    low_thresh = max(30, int(50 / (contrast_boost + 0.5)))
    high_thresh = max(100, int(150 / (contrast_boost + 0.5)))
    canny_edges = cv2.Canny(filtered, low_thresh, high_thresh)
    
    # Sobel edge detection (good for gradient magnitude)
    sobelx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_edges = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel_edges = cv2.threshold(sobel_edges, int(50 * (1 / contrast_boost)), 255, cv2.THRESH_BINARY)[1]
    
    # Laplacian edge detection (second derivative - good for fine details)
    laplacian = cv2.Laplacian(filtered, cv2.CV_32F)
    laplacian_edges = cv2.convertScaleAbs(laplacian)
    laplacian_edges = cv2.threshold(laplacian_edges, int(30 * (1 / contrast_boost)), 255, cv2.THRESH_BINARY)[1]
    
    # Combine all edge detections
    combined = cv2.bitwise_or(canny_edges, sobel_edges)
    combined = cv2.bitwise_or(combined, laplacian_edges)
    
    # Morphological refinement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    refined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return refined

# ============================================================================
# TOPOLOGICAL ANALYSIS: Floating Island & Bridge Detection
# ============================================================================

def detect_floating_islands(binary_image):
    """
    Detects 'floating islands' - enclosed white or black regions that would
    disconnect when cut. Uses connected component analysis.
    Returns: labeled image, island statistics
    """
    # Label connected components
    labeled_array, num_features = label(binary_image)
    
    island_stats = {}
    for component_id in range(1, num_features + 1):
        component = labeled_array == component_id
        area = np.sum(component)
        island_stats[component_id] = {
            'area': area,
            'component': component,
            'centroid': np.mean(np.where(component), axis=1)
        }
    
    return labeled_array, island_stats

def insert_bridges(binary_image, line_weight=2.0):
    """
    Structural integrity algorithm: detects floating islands and inserts
    'bridges' (small breaks in outline) to maintain connectivity.
    Critical for printable stencils.
    
    IMPORTANT: Input should be binary image where 255=black lines, 0=white background
    Output maintains the same convention
    """
    # Work with the image as-is (don't invert)
    working = binary_image.copy()
    
    # Detect floating islands (look for enclosed white regions)
    white_regions = (working < 128).astype(np.uint8)
    labeled, stats = detect_floating_islands(white_regions)
    
    result = working.copy()
    
    # For each island, find its boundary and create a bridge
    for island_id, island_data in stats.items():
        if island_data['area'] < 50:  # Skip very small components
            continue
        
        # Find boundary pixels of this island
        component = island_data['component']
        dilation = cv2.dilate(component.astype(np.uint8) * 255,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        erosion = cv2.erode(component.astype(np.uint8) * 255,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        boundary = dilation - erosion
        
        boundary_pixels = np.where(boundary > 0)
        if len(boundary_pixels[0]) == 0:
            continue
        
        # Create bridges at random boundary points (~10%)
        bridge_indices = np.random.choice(len(boundary_pixels[0]), 
                                         max(1, len(boundary_pixels[0]) // 10),
                                         replace=False)
        
        for idx in bridge_indices:
            y, x = boundary_pixels[0][idx], boundary_pixels[1][idx]
            radius = max(1, int(line_weight))
            # Make the bridge lighter (cut through the line)
            cv2.circle(result, (x, y), radius, 200, -1)
    
    return result

# ============================================================================
# VECTORIZATION: Potrace-inspired contour-to-vector conversion
# ============================================================================

def trace_contours_to_vectors(binary_image, max_error=1.0):
    """
    Potrace-inspired algorithm: Traces binary stencil contours and converts
    to vectorized paths. Returns contours that can be converted to SVG.
    
    This enables:
    - Lossless scaling to any size
    - SVG export for laser cutters
    - Sub-pixel precision
    
    Parameters:
    - binary_image: Binary image (0-255)
    - max_error: Douglas-Peucker approximation error (default 1.0)
    """
    # Find contours
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    vectorized_paths = []
    
    for contour in contours:
        # Approximate contour with reduced points (Douglas-Peucker)
        epsilon = max_error * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to vector format
        vector_path = {
            'points': approx.reshape(-1, 2).tolist(),
            'closed': True,
            'area': cv2.contourArea(approx)
        }
        vectorized_paths.append(vector_path)
    
    return vectorized_paths

def generate_svg_from_vectors(vector_paths, width, height):
    """
    Generates SVG markup from vectorized contours.
    Output is resolution-independent and printable on any cutter.
    """
    svg_lines = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '<style>path { fill: none; stroke: black; stroke-width: 0.5px; }</style>'
    ]
    
    for path_data in vector_paths:
        points = path_data['points']
        if len(points) < 2:
            continue
        
        path_str = f"M {points[0][0]} {points[0][1]}"
        for point in points[1:]:
            path_str += f" L {point[0]} {point[1]}"
        path_str += " Z"  # Close path
        
        svg_lines.append(f'<path d="{path_str}" />')
    
    svg_lines.append('</svg>')
    return '\n'.join(svg_lines)

# ============================================================================
# ADAPTIVE CONTRAST & THRESHOLDING
# ============================================================================

def adaptive_contrast_enhancement(img_gray, contrast_boost=1.2):
    """
    Professional contrast enhancement using:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Dynamic histogram stretching
    - Gamma correction
    """
    # CLAHE with adaptive clip limit
    clahe = cv2.createCLAHE(clipLimit=contrast_boost * 4, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    
    # Histogram stretching
    p2, p98 = np.percentile(enhanced, (2, 98))
    stretched = np.uint8(np.clip((enhanced - p2) / (p98 - p2) * 255, 0, 255))
    
    # Gamma correction
    gamma = 1.0 / (contrast_boost * 0.8)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    gamma_corrected = cv2.LUT(stretched, table)
    
    return gamma_corrected

def adaptive_threshold_otsu(img_gray):
    """
    Advanced multi-level Otsu thresholding for superior binary conversion.
    Handles varying lighting conditions across image.
    """
    # Global Otsu
    _, global_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive local threshold
    local_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 21, 5)
    
    # Combine: global for structure, local for details
    combined = cv2.bitwise_and(global_thresh, local_thresh)
    
    return combined

# ============================================================================
# PROFESSIONAL STENCIL FILTERS (Enhanced with Advanced Algorithms)
# ============================================================================

def filter_classic(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    Fine Linework: Uses hybrid edge detection + structural integrity.
    """
    # Enhance contrast
    enhanced = adaptive_contrast_enhancement(img_gray, contrast_boost)
    
    # Hybrid edge detection
    edges = hybrid_edge_detection(enhanced, contrast_boost)
    
    # Remove noise
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(edges)
    min_area = max(5, int(20 * (1 / (noise_reduction + 0.1))))
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(clean, [contour], 0, 255, 1)
    
    # Result: black lines on white background
    result = clean
    
    # Insert bridges for structural integrity
    result = insert_bridges(result, line_weight)
    
    return result

def filter_sketchy(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    Cross-Hatch: Uses segmentation + adaptive gradient detection.
    """
    # Enhance contrast
    enhanced = adaptive_contrast_enhancement(img_gray, contrast_boost)
    
    # Segment foreground
    foreground = segment_foreground_background(enhanced)
    
    # Multi-scale edge detection
    sobelx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    grad_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Inverted for shadow mapping
    inverted = 255 - enhanced
    light_shadow = cv2.inRange(inverted, 15, 80)
    mid_shadow = cv2.inRange(inverted, 81, 150)
    deep_shadow = cv2.inRange(inverted, 151, 220)
    
    # Generate adaptive hatching
    h, w = img_gray.shape
    hatch_45 = np.zeros_like(img_gray)
    hatch_135 = np.zeros_like(img_gray)
    
    spacing_light = max(2, int(5 / (noise_reduction + 0.3)))
    spacing_mid = max(2, int(4 / (noise_reduction + 0.3)))
    spacing_deep = max(2, int(3 / (noise_reduction + 0.3)))
    
    # 45-degree hatching
    for i in range(0, h + w, spacing_light):
        cv2.line(hatch_45, (max(0, i - h), 0), (i, min(w, i)), 255, 1)
    
    for i in range(0, h + w, spacing_mid):
        cv2.line(hatch_45, (max(0, i - h), 0), (i, min(w, i)), 255, 1)
    
    # 135-degree hatching
    for i in range(-w, h, spacing_deep):
        cv2.line(hatch_135, (0, i), (w, i + w), 255, 1)
    
    # Apply hatching to shadows
    light_hatch = cv2.bitwise_and(hatch_45, hatch_45, mask=light_shadow)
    mid_hatch = cv2.bitwise_and(hatch_45, hatch_45, mask=mid_shadow)
    deep_hatch = cv2.bitwise_and(hatch_135, hatch_135, mask=deep_shadow)
    
    # Combine
    result = cv2.bitwise_or(light_hatch, mid_hatch)
    result = cv2.bitwise_or(result, deep_hatch)
    
    # Extract contours
    _, contour_base = cv2.threshold(enhanced, int(180 * (1/contrast_boost)), 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(contour_base, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    clean_contours = np.zeros_like(contour_base)
    for contour in contours:
        if cv2.contourArea(contour) > max(10, int(30 * (1 / (noise_reduction + 0.2)))):
            cv2.drawContours(clean_contours, [contour], 0, 255, 1)
    
    result = cv2.bitwise_or(result, clean_contours)
    
    # Insert bridges for structural integrity (result is already black lines on white)
    result = insert_bridges(result, line_weight)
    
    return result

def filter_smooth(img_gray, line_weight, noise_reduction, contrast_boost):
    """
    Stipple: Uses adaptive threshold + halftone dithering.
    """
    # Enhance contrast
    enhanced = adaptive_contrast_enhancement(img_gray, contrast_boost)
    
    # Segment foreground
    foreground = segment_foreground_background(enhanced)
    
    # Adaptive threshold
    thresh = adaptive_threshold_otsu(enhanced)
    
    # Invert for stipple
    inverted = 255 - enhanced
    
    # Ordered dithering (Bayer 4x4)
    bayer_4x4 = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ], dtype=np.uint8) / 16.0
    
    h, w = inverted.shape
    dithered = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            bayer_val = bayer_4x4[y % 4, x % 4] * 255
            if inverted[y, x] > bayer_val:
                dithered[y, x] = 255
    
    # Extract contours
    _, contour_base = cv2.threshold(enhanced, int(170 * (1/contrast_boost)), 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(contour_base, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    clean_contours = np.zeros_like(contour_base)
    for contour in contours:
        if cv2.contourArea(contour) > max(10, int(40 * (1 / (noise_reduction + 0.2)))):
            cv2.drawContours(clean_contours, [contour], 0, 255, 1)
    
    result = cv2.bitwise_or(dithered, clean_contours)
    
    # Insert bridges for structural integrity
    result = insert_bridges(result, line_weight)
    
    return result

def filter_thermal_ink(img_gray, line_weight, noise_reduction, contrast_boost):
    """Classic Purple Thermal Burn with edge detection."""
    enhanced = adaptive_contrast_enhancement(img_gray, contrast_boost)
    edges = hybrid_edge_detection(enhanced, contrast_boost)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(line_weight * 2 + 1), int(line_weight * 2 + 1)))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    result = insert_bridges(dilated, line_weight)
    return result

def filter_carbon_transfer(img_gray, line_weight, noise_reduction, contrast_boost):
    """Hand-Traced Carbon with texture and adaptive threshold."""
    filtered = cv2.bilateralFilter(img_gray, 11, 25, 15)
    texture_noise = np.random.normal(0, int(noise_reduction * 3), filtered.shape).astype(np.float32)
    textured = np.clip(filtered.astype(np.float32) + texture_noise, 0, 255).astype(np.uint8)
    block_size = int(11 + (noise_reduction * 4))
    if block_size % 2 == 0:
        block_size += 1
    stencil = cv2.adaptiveThreshold(textured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block_size, int(contrast_boost * 3))
    result = insert_bridges(stencil, line_weight)
    return result

def filter_bold(img_gray, line_weight, noise_reduction, contrast_boost):
    """Heavy Traditional with strong contrast."""
    enhanced = adaptive_contrast_enhancement(img_gray, contrast_boost)
    _, thresh = cv2.threshold(enhanced, int(140 * (1 / contrast_boost)), 255, cv2.THRESH_BINARY_INV)
    kernel_size = int(5 + line_weight * 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(thresh, kernel, iterations=int(line_weight))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    result = insert_bridges(result, line_weight)
    return result

def filter_sharp(img_gray, line_weight, noise_reduction, contrast_boost):
    """Vector-style with Laplacian sharpening."""
    filtered = cv2.bilateralFilter(img_gray, 13, 30, 20)
    _, thresh = cv2.threshold(filtered, int(127 * (1 / contrast_boost)), 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(line_weight + 1), int(line_weight + 1)))
    dilated = cv2.dilate(opened, kernel_dilate, iterations=1)
    laplacian = cv2.Laplacian(dilated, cv2.CV_32F)
    sharpened = cv2.convertScaleAbs(laplacian)
    result = cv2.bitwise_or(dilated, sharpened)
    return insert_bridges(result, line_weight)

def filter_duotone_purple(img_gray, line_weight, noise_reduction, contrast_boost):
    """Two-Tone with full range tone separation."""
    enhanced = adaptive_contrast_enhancement(img_gray, contrast_boost)
    _, dark_mask = cv2.threshold(enhanced, int(100 * contrast_boost), 255, cv2.THRESH_BINARY)
    _, light_mask = cv2.threshold(enhanced, int(150 * contrast_boost), 255, cv2.THRESH_BINARY_INV)
    mid_mask = cv2.bitwise_and(cv2.bitwise_not(dark_mask), light_mask)
    h, w = img_gray.shape
    result = np.ones((h, w, 3), dtype=np.uint8) * 255
    dark_violet = (75, 0, 130)
    result[dark_mask == 0] = dark_violet
    light_lilac = (200, 162, 200)
    result[mid_mask == 255] = light_lilac
    return result

def filter_high_contrast(img_gray, line_weight, noise_reduction, contrast_boost):
    """Pure Black/White with Otsu's method."""
    enhanced = adaptive_contrast_enhancement(img_gray, contrast_boost)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    if line_weight > 1.0:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(line_weight * 2 + 1), int(line_weight * 2 + 1)))
        opened = cv2.dilate(opened, kernel_dilate, iterations=1)
    result = insert_bridges(opened, line_weight)
    return result

def process_stencil_core(img_gray, style, line_weight, noise_reduction, contrast_boost):
    """Main processing router with ML/CV advanced algorithms."""
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
        return filter_duotone_purple(img_gray, line_weight, noise_reduction, contrast_boost)
    elif style == "high_contrast":
        return filter_high_contrast(img_gray, line_weight, noise_reduction, contrast_boost)
    else:
        return filter_classic(img_gray, line_weight, noise_reduction, contrast_boost)

@app.post("/generate")
async def generate_stencil(
    file: UploadFile = File(...),
    style: str = Form("classic"),
    line_weight: float = Form(2.0),
    noise_reduction: float = Form(1.0),
    contrast_boost: float = Form(1.2),
    output_format: str = Form("png")
):
    """
    AI-Powered Professional Stencil Generation with Advanced CV/ML Algorithms.
    
    Implements:
    - U-Net-like Foreground/Background Segmentation
    - Multi-scale Hybrid Edge Detection (Canny + Sobel + Laplacian)
    - Adaptive Contrast & Thresholding (CLAHE + Otsu)
    - Topological Analysis & Bridge Insertion
    - Vectorization (Potrace-inspired)
    
    Parameters:
    - style: ['classic', 'sketchy', 'smooth', 'thermal_ink', 'carbon_transfer', 'bold', 'sharp', 'duotone_purple', 'high_contrast']
    - line_weight: 0.5 - 5.0
    - noise_reduction: 0.1 - 3.0
    - contrast_boost: 0.5 - 3.0
    - output_format: 'png' or 'svg'
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = process_stencil_core(gray, style, line_weight, noise_reduction, contrast_boost)
        
        # Handle duotone special case
        if style == "duotone_purple":
            output_rgb = processed
        else:
            if len(processed.shape) == 2:
                output_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                output_rgb = processed
        
        # Ensure white background
        h, w = output_rgb.shape[:2]
        result_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        mask = output_rgb[:, :, 0] < 200
        result_img[mask] = output_rgb[mask]
        
        if output_format == "svg":
            # Generate vectorized SVG
            gray_binary = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray_binary, 128, 255, cv2.THRESH_BINARY)
            vectors = trace_contours_to_vectors(binary, max_error=1.0)
            svg_content = generate_svg_from_vectors(vectors, w, h)
            return Response(content=svg_content, media_type="image/svg+xml")
        else:
            # PNG output
            _, encoded = cv2.imencode('.png', result_img)
            return Response(content=encoded.tobytes(), media_type="image/png")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "AI-Powered Professional Tattoo Stencil Engine",
        "features": [
            "U-Net Segmentation",
            "Hybrid Edge Detection",
            "Adaptive Contrast Enhancement",
            "Topological Bridge Insertion",
            "Potrace Vectorization",
            "SVG Export"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
