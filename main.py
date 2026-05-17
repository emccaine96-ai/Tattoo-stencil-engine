import io
import cv2
import numpy as np
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

@app.post("/api/v1/generate-stencil/")
async def generate_stencil(
    file: UploadFile = File(...),
    style: str = Form("detailed"),           
    line_weight: float = Form(2.0),          
    noise_reduction: float = Form(1.0),      
    contrast_boost: float = Form(1.2)        
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Corrupted image file.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_gray = process_stencil_core(gray, style, line_weight, noise_reduction, contrast_boost)

        # 4. Background Stripping Layer (Alpha Channel Masking)
        rgba = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGRA)
        
        # Turn white background pixels completely transparent
        white_mask = (rgba[:, :, 0] == 255) & (rgba[:, :, 1] == 255) & (rgba[:, :, 2] == 255)
        rgba[white_mask, 3] = 0  

        # Apply standard carbon-transfer purple (#4A2E80) to remaining lines
        line_mask = (rgba[:, :, 3] > 0)
        rgba[line_mask, 0] = 128  # Blue Channel
        rgba[line_mask, 1] = 46   # Green Channel
        rgba[line_mask, 2] = 74   # Red Channel

        _, encoded_png = cv2.imencode('.png', rgba)
        return Response(content=encoded_png.tobytes(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
