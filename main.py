from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import fitz, io, re, zipfile
from PIL import Image, ImageOps
import numpy as np
import cv2

app = FastAPI()

# -------------------------------------------------------------------- #
MARGIN            = 10
TOP_CROP_MAX      = 30
SIDE_CROP_RATIO   = 0.45
BOTTOM_CROP_RATIO = 0.30
WHITE             = 245

KEYWORDS = re.compile(
    r"(lançamentos|compras|despesas|transa[çc]ões|gastos|detalhe|movimenta)",
    re.I
)
DATE_RE   = re.compile(r"\b\d{2}[/-]\d{2}\b")
MONEY_RE  = re.compile(r"\d+\.\d{3},\d{2}|\d+,\d{2}")

# -------------------- image helpers ---------------------------------- #
def safe_crop(img: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(img)
    arr  = np.array(gray)
    nz   = np.where(arr < WHITE)
    if nz[0].size == 0:
        return img

    y_min, y_max = nz[0].min(), nz[0].max()
    x_min, x_max = nz[1].min(), nz[1].max()

    x_min = max(0,  x_min - MARGIN)
    y_min = max(0,  y_min - MARGIN)
    x_max = min(arr.shape[1], x_max + MARGIN)
    y_max = min(arr.shape[0], y_max + MARGIN)

    W, H = img.size
    if y_min > TOP_CROP_MAX:
        y_min = 0
    if x_min > W * SIDE_CROP_RATIO:
        x_min = 0
    if (W - x_max) > W * SIDE_CROP_RATIO:
        x_max = W
    if (H - y_max) > H * BOTTOM_CROP_RATIO:
        y_max = H

    return img.crop((x_min, y_min, x_max, y_max))

def mild_deskew(img: Image.Image) -> Image.Image:
    arr  = np.array(img.convert("L"))
    pts  = np.column_stack(np.where(arr < 255))
    if pts.size == 0:
        return img
    angle = cv2.minAreaRect(pts)[-1]
    if angle < -45: angle += 90
    if abs(angle) > 3:         # ignore large angles (would flip page)
        return img
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

def page_is_useful(text: str, idx: int) -> bool:
    if idx == 0:                       # always keep summary page
        return True
    if KEYWORDS.search(text):
        return True
    dates  = DATE_RE.findall(text)
    money  = MONEY_RE.findall(text)
    return len(dates) >= 2 and len(money) >= 1

# -------------------------------------------------------------------- #
@app.post("/process-pdf/")
async def process_pdf(
        file: UploadFile = File(...),
        cpf:  str        = Form(...)
    ):
    pdf = await file.read()
    doc = fitz.open(stream=pdf, filetype="pdf")

    # ---------- CPF password trials ---------------------------------- #
    if doc.needs_pass:
        digits = re.sub(r"\D", "", cpf)
        for pw in (digits[:5], digits[:4], digits[:6], digits[:3]):
            if pw and doc.authenticate(pw):
                break
        else:
            return {"error": "CPF password failed"}

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w",
                         zipfile.ZIP_DEFLATED, compresslevel=4) as z:

        for idx, page in enumerate(doc):
            text = page.get_text("text")
            if not page_is_useful(text, idx):
                continue                          # skip ads, simulators

            # render upright (PyMuPDF already honors rotation flag)
            pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
            img = Image.frombytes("L", (pix.width, pix.height),
                                  pix.samples)

            img = mild_deskew(img)
            img = safe_crop(img)
            img = binarise(img)

            if img.width > 2000:
                r = 2000 / img.width
                img = img.resize((2000, int(img.height * r)),
                                 Image.LANCZOS)

            tmp = io.BytesIO()
            img.save(tmp, "PNG", optimize=True)
            tmp.seek(0)
            z.writestr(f"{idx+1:02d}.png", tmp.read())

    zbuf.seek(0)
    return StreamingResponse(
        zbuf,
        media_type="application/x-zip-compressed",
        headers={
            "Content-Disposition":
            "attachment; filename=ai_ready_pages.zip"
        }
    )

