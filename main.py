from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import fitz                         # PyMuPDF
import cv2, numpy as np
import io, zipfile
from typing import List

app = FastAPI()

# ────────────────────────────────────────────────────────────────────────────────
# ★ helper functions ★
# ────────────────────────────────────────────────────────────────────────────────
def pdf_pass(pdf_bytes: bytes, pw_guesses: List[str]) -> fitz.Document | None:
    """
    Try the supplied password guesses (CPF-based substrings) and return
    an open, authenticated PyMuPDF document, or None if none work.
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
    Crop white margins, keep full 8-bit grayscale, lightly boost contrast,
    and return PNG bytes.
    """
    # Pixmap → NumPy
    img = np.frombuffer(pix.samples, dtype=np.uint8)\
            .reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:                                  # RGBA → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Crop large white margins
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    x, y, w, h = cv2.boundingRect(mask)
    y = max(y - 10, 0)                              # keep 10-px top margin
    gray = gray[y:y + h, x:x + w]

    # Light contrast enhancement (preserves shades)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Encode as 8-bit gray PNG with lower compression
    _, png_bytes = cv2.imencode(".png", gray,
                                [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return png_bytes.tobytes()

# ────────────────────────────────────────────────────────────────────────────────
# ★ main endpoint ★
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/process-pdf/")
async def process_pdf(
    file: UploadFile = File(...),
    cpf: str       = Form(...),
):
    pdf_bytes = await file.read()

    pw_guesses = [cpf[:5], cpf[:4], cpf[:6], cpf[:3]]
    doc = pdf_pass(pdf_bytes, pw_guesses)
    if doc is None:
        return JSONResponse({"error": "Password (CPF) not accepted"}, status_code=401)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        selected_pages = list(range(len(doc)))

        for i in selected_pages:
            page = doc[i]
            pix = page.get_pixmap()  # ← Removed dpi=250 to avoid lowering resolution
            img_data = preprocess(pix)
            zf.writestr(f"{i+1:02d}_{file.filename[:-4]}.png", img_data)

    zip_buf.seek(0)
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=pages.zip"}
    )

from fastapi import Body

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/decrypt-pdf/")
async def decrypt_pdf(
    file: UploadFile = File(...),
    cpf: str = Form(...),
):
    pdf_bytes = await file.read()

    # Try decrypting
    pw_guesses = [cpf[:5], cpf[:4], cpf[:6], cpf[:3]]
    doc = pdf_pass(pdf_bytes, pw_guesses)
    if doc is None:
        return JSONResponse({"error": "Password (CPF) not accepted"}, status_code=401)

    # Save as unencrypted PDF to a temp buffer
    output_buffer = io.BytesIO()
    doc.save(output_buffer, encryption=fitz.PDF_ENCRYPT_NONE)
    output_buffer.seek(0)

    return StreamingResponse(output_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=decrypted_{file.filename}"}
    )

