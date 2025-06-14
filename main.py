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
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if not doc.needs_pass:
        return doc
    for guess in pw_guesses:
        if doc.authenticate(guess):
            return doc
    return None

def preprocess(pix: fitz.Pixmap) -> bytes:
    img = np.frombuffer(pix.samples, dtype=np.uint8)\
            .reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    x, y, w, h = cv2.boundingRect(mask)
    y = max(y - 10, 0)
    gray = gray[y:y + h, x:x + w]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, png_bytes = cv2.imencode(".png", gray,
                                [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return png_bytes.tobytes()

# ────────────────────────────────────────────────────────────────────────────────
# ★ endpoints ★
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

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(doc.page_count):
            pix = doc[i].get_pixmap()
            img = preprocess(pix)
            zf.writestr(f"{i+1:02d}_{file.filename[:-4]}.png", img)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=pages.zip"}
    )

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/decrypt-pdf/")
async def decrypt_pdf(
    file: UploadFile = File(...),
    cpf: str       = Form(...),
):
    pdf_bytes = await file.read()
    pw_guesses = [cpf[:5], cpf[:4], cpf[:6], cpf[:3]]
    doc = pdf_pass(pdf_bytes, pw_guesses)
    if doc is None:
        return JSONResponse({"error": "Password (CPF) not accepted"}, status_code=401)

    new_doc = fitz.open()
    scale = 150 / 72
    mat = fitz.Matrix(scale, scale)
    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        new_page = new_doc.new_page(width=pix.width, height=pix.height)
        new_page.insert_image(fitz.Rect(0, 0, pix.width, pix.height), stream=img_bytes)

    out = io.BytesIO()
    new_doc.save(out)
    out.seek(0)

    return StreamingResponse(
        out,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=reprinted_{file.filename}"}
    )

