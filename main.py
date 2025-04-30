from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import fitz, io, re, zipfile
from PIL import Image, ImageOps
import numpy as np
import cv2

app = FastAPI()

# -------------------------------------------------------------------- #
MARGIN            = 10      # keep this border everywhere
TOP_CROP_MAX      = 30      # never remove more than 30 px from top
SIDE_CROP_RATIO   = 0.45    # allow up to 45 % crop on left/right
BOTTOM_CROP_RATIO = 0.30    # allow up to 30 % crop at bottom
WHITE             = 245     # threshold for “white” pixels

def safe_crop(img: Image.Image) -> Image.Image:
    """Aggressive side/bottom crop, gentle on top."""
    gray = ImageOps.grayscale(img)
    arr  = np.array(gray)
    nz   = np.where(arr < WHITE)                # dark pixel coords
    if nz[0].size == 0:
        return img

    y_min, y_max = nz[0].min(), nz[0].max()
    x_min, x_max = nz[1].min(), nz[1].max()

    # margins
    x_min = max(0, x_min - MARGIN)
    y_min = max(0, y_min - MARGIN)
    x_max = min(arr.shape[1], x_max + MARGIN)
    y_max = min(arr.shape[0], y_max + MARGIN)

    W, H = img.size
    # --- guard top crop
    if y_min > TOP_CROP_MAX:
        y_min = 0

    # --- guard side / bottom crop %
    if x_min > W * SIDE_CROP_RATIO: x_min = 0
    if (W - x_max) > W * SIDE_CROP_RATIO: x_max = W
    if (H - y_max) > H * BOTTOM_CROP_RATIO: y_max = H

    return img.crop((x_min, y_min, x_max, y_max))

def mild_deskew(img: Image.Image) -> Image.Image:
    """Rotate only small skew angles (<3°) so we never flip 90°."""
    arr  = np.array(img.convert("L"))
    pts  = np.column_stack(np.where(arr < 255))
    if pts.size == 0:
        return img
    angle = cv2.minAreaRect(pts)[-1]
    if angle < -45: angle += 90
    if abs(angle) < 0.5 or abs(angle) > 3:
        return img                      # ignore tiny or huge angles
    h, w = arr.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rot = cv2.warpAffine(arr, M, (w, h),
                         flags=cv2.INTER_LINEAR, borderValue=255)
    return Image.fromarray(rot)

def binarise(img: Image.Image) -> Image.Image:
    g  = np.array(img.convert("L"))
    bw = cv2.adaptiveThreshold(g, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 25, 15)
    return Image.fromarray(bw)

# -------------------------------------------------------------------- #
@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...),
                      cpf:  str       = Form(...)):
    pdf = await file.read()
    doc = fitz.open(stream=pdf, filetype="pdf")

    # CPF-based password
    if doc.needs_pass:
        digits = re.sub(r"\D", "", cpf)
        for pw in (digits[:5], digits[:4], digits[:6], digits[:3]):
            if pw and doc.authenticate(pw):
                break
        else:
            return {"error": "CPF password failed"}

    buf_zip = io.BytesIO()
    with zipfile.ZipFile(buf_zip, "w",
                         zipfile.ZIP_DEFLATED,
                         compresslevel=4) as z:
        for i, pg in enumerate(doc):
            txt = pg.get_text("text")
            if i == 0 or re.search(r"lançamentos|compras", txt, re.I):

                # PyMuPDF gives upright orientation by default
                pix = pg.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
                img = Image.frombytes("L", (pix.width, pix.height),
                                      pix.samples)

                img = mild_deskew(img)   # tiny skew only
                img = safe_crop(img)     # keep header safe
                img = binarise(img)

                # downscale if very wide
                if img.width > 2000:
                    r = 2000 / img.width
                    img = img.resize((2000, int(img.height * r)),
                                     Image.LANCZOS)

                tmp = io.BytesIO()
                img.save(tmp, "PNG", optimize=True)
                tmp.seek(0)
                z.writestr(f"{i+1:02d}.png", tmp.read())

    buf_zip.seek(0)
    return StreamingResponse(buf_zip,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition":
                 "attachment; filename=ai_ready_pages.zip"})

