from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import fitz  # PyMuPDF
import io
import zipfile
from PIL import Image, ImageOps, ImageChops
import re
import numpy as np
import cv2

app = FastAPI()

def crop_borders(pil_img):
    # Convert to grayscale and threshold to find content bounding box
    gray = ImageOps.grayscale(pil_img)
    np_img = np.array(gray)
    _, thresh = cv2.threshold(np_img, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = pil_img.crop((x, y, x + w, y + h))
    return cropped

def deskew_image(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    coords = np.column_stack(np.where(gray < 255))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    rot_mat = cv2.getRotationMatrix2D((gray.shape[1] // 2, gray.shape[0] // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
    return Image.fromarray(rotated)

def adaptive_binarize(pil_img):
    gray = ImageOps.grayscale(pil_img)
    img_np = np.array(gray)
    binarized = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 15)
    return Image.fromarray(binarized)

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...), password: str = Form(...)):
    contents = await file.read()
    pdf_document = fitz.open(stream=contents, filetype="pdf")

    if pdf_document.needs_pass:
        if not pdf_document.authenticate(password):
            return {"error": "Incorrect password"}

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zip_file:
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            text = page.get_text("text")

            if page_number == 0 or re.search(r"lançamentos|compras", text, re.IGNORECASE):
                pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
                pil_img = Image.frombytes("L", [pix.width, pix.height], pix.samples)

                # Deskew
                pil_img = deskew_image(pil_img)

                # Crop whitespace
                pil_img = crop_borders(pil_img)

                # Adaptive thresholding → black text on white
                pil_img = adaptive_binarize(pil_img)

                # Optional: resize if too large
                if pil_img.width > 2000:
                    ratio = 2000 / pil_img.width
                    new_size = (2000, int(pil_img.height * ratio))
                    pil_img = pil_img.resize(new_size, Image.LANCZOS)

                # Save page as PNG into zip
                page_date = "2025-04-30"  # placeholder; replace with date detection if needed
                filename = f"{page_date}_page-{page_number + 1}.png"

                img_io = io.BytesIO()
                pil_img.save(img_io, format="PNG", optimize=True)
                img_io.seek(0)
                zip_file.writestr(filename, img_io.read())

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/x-zip-compressed", headers={
        "Content-Disposition": "attachment; filename=ai_ready_pages.zip"
    })

