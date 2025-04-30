from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import fitz, io, re, zipfile
from PIL import Image
import numpy as np
import cv2

app = FastAPI()

# ---------- helpers ------------------------------------------------------- #
MARGIN = 5  # just shave a thin white frame

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
        rot = cv2.warpAffine(arr, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderValue=255)
        return Image.fromarray(rot)
    except Exception:
        return img

def fixed_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((MARGIN, MARGIN, w - MARGIN, h - MARGIN))

def binarise(img: Image.Image) -> Image.Image:
    g  = np.array(img.convert("L"))
    bw = cv2.adaptiveThreshold(g, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 25, 15)
    return Image.fromarray(bw)

def orient_from_flag(page, img: Image.Image) -> Image.Image:
    rot = page.rotation % 360
    if rot == 90:
        img = img.rotate(-90, expand=True, fillcolor=255)
    elif rot == 270:
        img = img.rotate(-270, expand=True, fillcolor=255)
    elif rot == 180:
        img = img.rotate(180, expand=True, fillcolor=255)
    return img
# ------------------------------------------------------------------------- #

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...), cpf: str = Form(...)):
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # CPF-based password
    if doc.needs_pass:
        digits = re.sub(r"\D", "", cpf)
        for pw in (digits[:5], digits[:4], digits[:6], digits[:3]):
            if pw and doc.authenticate(pw):
                break
        else:
            return {"error": "CPF password failed"}

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for i, page in enumerate(doc):
            txt = page.get_text("text")
            if i == 0 or re.search(r"lançamentos|compras", txt, re.I):

                pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
                img = Image.frombytes("L", (pix.width, pix.height), pix.samples)

                img = orient_from_flag(page, img)  # only flag-based rotation
                img = deskew(img)
                img = fixed_crop(img)              # minimal crop
                img = binarise(img)

                if img.width > 2000:
                    ratio = 2000 / img.width
                    img = img.resize((2000, int(img.height * ratio)),
                                     Image.LANCZOS)

                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                buf.seek(0)
                z.writestr(f"{i+1:02d}.png", buf.read())

    out.seek(0)
    return StreamingResponse(out,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=ai_ready_pages.zip"})

