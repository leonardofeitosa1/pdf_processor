from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import fitz  # PyMuPDF
import io, re, zipfile
from PIL import Image, ImageOps
import numpy as np
import cv2

app = FastAPI()

# ---------------------------------------------------------------------------#
#                           image-processing helpers                         #
# ---------------------------------------------------------------------------#

WHITE              = 245   # pixels brighter than this are “white”
MARGIN             = 10    # leave 10-px border after crop
MAX_CROP_RATIO     = 0.30  # never crop >30 % of width/height

def smart_crop(img: Image.Image) -> Image.Image:
    """
    Crop only large outer whitespace.  Keeps a 10-px margin and aborts if
    cropping would remove more than 30 % of any dimension (to avoid
    cutting table cells or amounts).
    """
    gray = ImageOps.grayscale(img)
    arr  = np.array(gray)
    mask = arr < WHITE                  # True where content (dark)

    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return img                      # blank page

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # add margin back
    x0 = max(0,  x0 - MARGIN)
    y0 = max(0,  y0 - MARGIN)
    x1 = min(arr.shape[1], x1 + MARGIN)
    y1 = min(arr.shape[0], y1 + MARGIN)

    # safety-guard against excessive crop
    if (x0 > img.width  * MAX_CROP_RATIO or
        y0 > img.height * MAX_CROP_RATIO or
        (img.width  - x1) > img.width  * MAX_CROP_RATIO or
        (img.height - y1) > img.height * MAX_CROP_RATIO):
        return img

    return img.crop((x0, y0, x1, y1))

def deskew(img: Image.Image) -> Image.Image:
    """
    Auto-deskew using OpenCV.  Returns original if a deskew angle
    cannot be computed.
    """
    try:
        arr  = np.array(img.convert("L"))
        pts  = np.column_stack(np.where(arr < 255))
        if pts.size == 0:
            return img
        angle = cv2.minAreaRect(pts)[-1]
        if angle < -45:
            angle += 90
        (h, w) = arr.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rot = cv2.warpAffine(arr, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderValue=255)
        return Image.fromarray(rot)
    except Exception:
        return img

def binarise(img: Image.Image) -> Image.Image:
    """
    Adaptive threshold → crisp black/white, better OCR & compression.
    """
    try:
        g  = np.array(ImageOps.grayscale(img))
        bw = cv2.adaptiveThreshold(g, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   25, 15)
        return Image.fromarray(bw)
    except Exception:
        return ImageOps.grayscale(img)

def orient_to_portrait(page, img: Image.Image) -> Image.Image:
    """
    1) Undo the page’s native rotation flag.
    2) If still landscape, rotate −90° so text is upright portrait.
    """
    rot = page.rotation % 360            # 0, 90, 180, 270
    if rot:
        img = img.rotate(-rot, expand=True, fillcolor=255)
    if img.width > img.height:
        img = img.rotate(-90, expand=True, fillcolor=255)
    return img

# ---------------------------------------------------------------------------#
#                               main endpoint                                #
# ---------------------------------------------------------------------------#

@app.post("/process-pdf/")
async def process_pdf(
        file: UploadFile = File(...),
        cpf:  str        = Form(...)
    ):
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # ---------- CPF-based password trials ----------------------------------#
    if doc.needs_pass:
        digits = re.sub(r"\D", "", cpf)          # keep only numbers
        trials = [digits[:5], digits[:4],
                  digits[:6], digits[:3]]
        if not any(pw and doc.authenticate(pw) for pw in trials):
            return {"error": "CPF-based password failed"}

    # ---------- export -----------------------------------------------------#
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w",
                         compression=zipfile.ZIP_DEFLATED,
                         compresslevel=4) as zf:
        for i, page in enumerate(doc):
            txt = page.get_text("text")
            if i == 0 or re.search(r"lançamentos|compras", txt, re.I):

                # render @200 DPI grayscale
                pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
                img = Image.frombytes("L",
                                      (pix.width, pix.height),
                                      pix.samples)

                # processing pipeline
                img = orient_to_portrait(page, img)
                img = deskew(img)
                img = smart_crop(img)
                img = binarise(img)

                # downsize if >2000 px wide
                if img.width > 2000:
                    ratio = 2000 / img.width
                    img = img.resize((2000,
                                      int(img.height * ratio)),
                                     Image.LANCZOS)

                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                buf.seek(0)
                zf.writestr(f"{i+1:02d}.png", buf.read())

    zip_buf.seek(0)
    return StreamingResponse(
        zip_buf,
        media_type="application/x-zip-compressed",
        headers={
            "Content-Disposition":
            "attachment; filename=ai_ready_pages.zip"
        }
    )
