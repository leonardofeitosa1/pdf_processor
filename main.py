from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import fitz                             # PyMuPDF
import io, re, zipfile
from PIL import Image, ImageOps
import numpy as np
import cv2

app = FastAPI()

# ---------- helpers ------------------------------------------------------- #
def crop_borders(img: Image.Image) -> Image.Image:
    try:
        gray = ImageOps.grayscale(img)
        arr  = np.array(gray)
        _, t = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
        nz   = cv2.findNonZero(t)
        if nz is None:
            return img
        x, y, w, h = cv2.boundingRect(nz)
        return img.crop((x, y, x + w, y + h))
    except Exception:
        return img

def deskew(img: Image.Image) -> Image.Image:
    try:
        arr = np.array(img.convert("L"))
        pts = np.column_stack(np.where(arr < 255))
        if pts.size == 0:
            return img
        angle = cv2.minAreaRect(pts)[-1]
        if angle < -45:
            angle += 90
        (h, w) = arr.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(arr, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderValue=255)
        return Image.fromarray(rotated)
    except Exception:
        return img

def binarise(img: Image.Image) -> Image.Image:
    try:
        g   = np.array(ImageOps.grayscale(img))
        bw  = cv2.adaptiveThreshold(g, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,
                                    25, 15)
        return Image.fromarray(bw)
    except Exception:
        return ImageOps.grayscale(img)

# -------------------------------------------------------------------------- #
@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...),
                      cpf: str = Form(...)):
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # ---------- try passwords --------------------------------------------- #
    if doc.needs_pass:
        digits = re.sub(r"\D", "", cpf)            # only numbers
        trials = [digits[:5], digits[:4],
                  digits[:6], digits[:3]]
        auth_ok = False
        for pw in trials:
            if pw and doc.authenticate(pw):
                auth_ok = True
                break
        if not auth_ok:
            return {"error": "CPF-based password failed"}

    # ---------- export ---------------------------------------------------- #
    out_zip = io.BytesIO()
    with zipfile.ZipFile(out_zip, "w",
                         compression=zipfile.ZIP_DEFLATED,
                         compresslevel=4) as zf:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if i == 0 or re.search(r"lançamentos|compras", text, re.I):

                pix = page.get_pixmap(dpi=200,
                                      colorspace=fitz.csGRAY)
                img = Image.frombytes("L",
                                      (pix.width, pix.height),
                                      pix.samples)

                # processing pipeline
                img = deskew(img)
                # rotate to portrait if still landscape
                if img.width > img.height:
                    img = img.rotate(90, expand=True, fillcolor=255)
                img = crop_borders(img)
                img = binarise(img)

                if img.width > 2000:
                    ratio = 2000 / img.width
                    img = img.resize((2000,
                                      int(img.height * ratio)),
                                     Image.LANCZOS)

                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                buf.seek(0)
                zf.writestr(f"{i+1:02d}.png", buf.read())

    out_zip.seek(0)
    return StreamingResponse(out_zip,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition":
                 "attachment; filename=ai_ready_pages.zip"})

