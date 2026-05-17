import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

app = FastAPI(title="Ultra Advanced Tattoo Stencil AI Engine")

# Enable CORS so your frontend apps can connect securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_stencil_core(img_gray, style, line_weight, noise_reduction, contrast_boost):
    """
    Advanced computer vision engine to extract pure stencil lines.
    """
    # 1. Bilateral Filtering: Removes skin grain/pores while keeping sharp edges crisp
    spatial_sigma = int(noise_reduction * 15)
    color_sigma = int(noise_reduction * 25)
    filtered = cv2.bilateralFilter(img_gray, 9, color_sigma, spatial_sigma)

    # 2. Contrast & Gamma Adjustment
    adjusted = cv2.convertScaleAbs(filtered, alpha=contrast_boost, beta=0)

    # 3. Core Processing Styles
    if style == "outline":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(adjusted, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.medianBlur(thresh, 3)

    elif style == "detailed":
        # Ridge Detection via Hessian Matrix eigenvalues (Isolates exact line structures)
        H_elems = hessian_matrix(adjusted, sigma=line_weight, order='rc')
        i1, i2 = hessian_matrix_eigvals(H_elems)
        
        ridges = np.uint8(cv2.normalize(i2, None, 0, 255, cv2.NORM_MINMAX))
        ridges = cv2.bitwise_not(ridges)
        
        block_size = 25
        stencil = cv2.adaptiveThreshold(
            ridges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, 4
        )
        return cv2.medianBlur(stencil, 3)
        
    else: 
        return cv2.adaptiveThreshold(
            adjusted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 15, 7
        )

def apply_stencil_filter(pil_image, filter_type):
    """
    Apply various stencil finish filters to match classic thermal stencil paper effects.
    Filters mimic the Stencil AI app from Play Store.
    """
    if filter_type == "classic":
        # Pure thermal paper look: crisp black lines on white
        return pil_image
    
    elif filter_type == "thermal_ink":
        # Slightly darkened lines for classic thermal printer effect
        enhancer = ImageEnhance.Brightness(pil_image)
        return enhancer.enhance(0.92)
    
    elif filter_type == "carbon_transfer":
        # Deep purple/burgundy lines (like carbon transfer paper)
        # Convert to RGB and apply color overlay
        img_array = np.array(pil_image)
        rgb_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        # Apply purple tint to dark lines
        purple_mask = img_array < 128
        rgb_array[purple_mask] = [128, 46, 74]  # BGR: Purple (#4A2E80)
        
        return Image.fromarray(cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB))
    
    elif filter_type == "bold":
        # Thicker, bolder stencil lines
        img_array = np.array(pil_image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(img_array, kernel, iterations=1)
        return Image.fromarray(dilated)
    
    elif filter_type == "smooth":
        # Smoothed edges for cleaner appearance
        return pil_image.filter(ImageFilter.SMOOTH)
    
    elif filter_type == "sharp":
        # Extra sharp lines for maximum detail
        return pil_image.filter(ImageFilter.SHARPEN)
    
    elif filter_type == "duotone_purple":
        # Two-tone purple effect (light and dark)
        img_array = np.array(pil_image)
        # Create a duotone effect
        duotone = np.where(
            img_array > 200,
            np.array([240, 240, 240]),  # Light gray for white areas
            np.array([100, 60, 100])     # Dark purple for lines
        )
        return Image.fromarray(duotone)
    
    elif filter_type == "sketchy":
        # Hand-drawn sketchy appearance
        img_array = np.array(pil_image)
        # Add slight noise and edge enhancement
        noise = np.random.randint(-15, 15, img_array.shape)
        sketchy = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(sketchy)
    
    elif filter_type == "high_contrast":
        # Maximum contrast for bold stencil effect
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Brightness(enhanced)
        return enhancer.enhance(0.95)
    
    else:
        return pil_image

@app.post("/api/v1/generate-stencil/")
async def generate_stencil(
    file: UploadFile = File(...),
    style: str = Form("detailed"),           
    line_weight: float = Form(2.0),          
    noise_reduction: float = Form(1.0),      
    contrast_boost: float = Form(1.2),
    filter_type: str = Form("classic")
):
    """
    Generate tattoo stencils with multiple filter options.
    
    Filter types:
    - classic: Pure thermal paper look (default)
    - thermal_ink: Slightly darkened thermal printer effect
    - carbon_transfer: Deep purple/burgundy lines
    - bold: Thicker, bolder lines
    - smooth: Smoothed edges
    - sharp: Extra sharp detail
    - duotone_purple: Two-tone purple effect
    - sketchy: Hand-drawn sketchy appearance
    - high_contrast: Maximum bold contrast
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Corrupted image file.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_gray = process_stencil_core(gray, style, line_weight, noise_reduction, contrast_boost)

        # Create white background image
        height, width = processed_gray.shape
        white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255  # Pure white in BGR
        
        # Create binary mask: 255 where stencil lines are (black), 0 where white
        stencil_mask = cv2.bitwise_not(processed_gray)
        
        # Place black lines on white background
        for i in range(3):  # Apply to all color channels
            white_bg[:, :, i] = cv2.bitwise_and(white_bg[:, :, i], stencil_mask)
        
        # Invert to get white background with black lines
        result = cv2.bitwise_not(white_bg)
        result = 255 - (255 - result)  # Ensure proper white background
        
        # Convert to grayscale for filter processing
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Convert to PIL Image for advanced filter processing
        pil_image = Image.fromarray(result_gray, mode='L')
        
        # Apply selected filter
        filtered_image = apply_stencil_filter(pil_image, filter_type)
        
        # Create final white background composite
        white_background = Image.new('RGB', filtered_image.size, (255, 255, 255))
        
        # If the image is in grayscale mode, paste it with a mask
        if filtered_image.mode == 'L':
            white_background.paste(filtered_image, (0, 0), filtered_image)
        else:
            # If RGB, compose them together
            white_background.paste(filtered_image, (0, 0))
        
        # Save to PNG bytes
        png_buffer = io.BytesIO()
        white_background.save(png_buffer, format='PNG')
        png_buffer.seek(0)
        
        return Response(content=png_buffer.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
