from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import fitz                         # PyMuPDF
import cv2, numpy as np
import io, zipfile, tempfile, os
from typing import List

app = FastAPI()

# ────────────────────────────────────────────────────────────────────────────────
# ★ helper functions ★
# ────────────────────────────────────────────────────────────────────────────────
def pdf_pass(pdf_bytes: bytes, pw_guesses: List[str]) -> fitz.Document | None:
    """
    Try the supplied password guesses (CPF-based substrings) and return
    an open, authenticated PyMuPDF document, or `None` if none work.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if not doc.needs_pass:
        return doc                                  # un-encrypted
    for guess in pw_guesses:
        if doc.authenticate(guess):
            return doc
    return None

def preprocess(pix: fitz.Pixmap) -> bytes:
    """
    Deskew -> crop white margins / barcodes ->
    convert to 8-bit gray -> adaptive threshold (bilevel) ->
    return PNG bytes.
    """
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:                                  # RGBA ➜ RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- auto-deskew -----------------------------------------------------------
    coords = np.column_stack(np.where(gray < 250))
    angle  = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

    # --- crop large white margins ---------------------------------------------
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    x, y, w, h = cv2.boundingRect(thresh)
    # be *less* aggressive on top margin → keep 10 px
    y = max(y - 10, 0)
    cropped = gray[y:y + h, x:x + w]

    # --- adaptive threshold → true black / white ------------------------------
    bw = cv2.adaptiveThreshold(cropped, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)

    # output ~200 DPI instead of ~150 DPI (was too soft)
    # ↑ *single* place where we raise resolution a bit
    encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    _, png_bytes = cv2.imencode('.png', bw, encode_param)
    return png_bytes.tobytes()

# ────────────────────────────────────────────────────────────────────────────────
# ★ main endpoint ★
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/process-pdf/")
async def process_pdf(
    file: UploadFile = File(...),
    cpf: str       = Form(...),                     # renamed field
):
    pdf_bytes = await file.read()

    pw_guesses = [cpf[:5], cpf[:4], cpf[:6], cpf[:3]]
    doc = pdf_pass(pdf_bytes, pw_guesses)
    if doc is None:
        return JSONResponse({"error": "Password (CPF) not accepted"}, status_code=401)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, page in enumerate(doc):
            # ➊ slightly higher DPI → 200  (was 150)
            pix = page.get_pixmap(dpi=200)
            img_data = preprocess(pix)
            zf.writestr(f"{i+1:02d}_{file.filename[:-4]}.png", img_data)

    zip_buf.seek(0)
    return StreamingResponse(zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=pages.zip"}
    )








from fastapi import Body

@app.get("/health")
def health():
    return {"ok": True}

