from fastapi import FastAPI, UploadFile, File, Form, JSONResponse
from fastapi.responses import StreamingResponse
import fitz                         # PyMuPDF
import cv2, numpy as np
import io, zipfile
from typing import List
import re
from collections import Counter

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
# ★ main endpoints ★
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
        for i in range(len(doc)):
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

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/decrypt-pdf/")
async def decrypt_pdf(
    file: UploadFile = File(...),
    cpf: str = Form(...),
):
    pdf_bytes = await file.read()
    pw_guesses = [cpf[:5], cpf[:4], cpf[:6], cpf[:3]]
    doc = pdf_pass(pdf_bytes, pw_guesses)
    if doc is None:
        return JSONResponse({"error": "Password (CPF) not accepted"}, status_code=401)

    output_buffer = io.BytesIO()
    doc.save(output_buffer, encryption=fitz.PDF_ENCRYPT_NONE)
    output_buffer.seek(0)

    return StreamingResponse(
        output_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=decrypted_{file.filename}"}
    )

@app.post("/reprint-pdf/")
async def reprint_pdf(
    file: UploadFile = File(...),
    cpf: str       = Form(...),
):
    pdf_bytes = await file.read()
    pw_guesses = [cpf[:5], cpf[:4], cpf[:6], cpf[:3]]
    doc = pdf_pass(pdf_bytes, pw_guesses)
    if doc is None:
        return JSONResponse({"error": "Password (CPF) not accepted"}, status_code=401)

    new_doc = fitz.open()  # empty PDF
    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)
        # render at 150 DPI (adjust if needed)
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")

        rect = fitz.Rect(0, 0, pix.width, pix.height)
        new_page = new_doc.new_page(width=pix.width, height=pix.height)
        new_page.insert_image(rect, stream=img_bytes)

    output_buffer = io.BytesIO()
    new_doc.save(output_buffer)
    output_buffer.seek(0)

    return StreamingResponse(
        output_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=reprinted_{file.filename}"}
    )

# ────────────────────────────────────────────────────────────────────────────────
# ★ colored-transaction extraction service ★
# ────────────────────────────────────────────────────────────────────────────────

money_re = re.compile(r'([+\-−–—]?)\s*\d{1,3}(?:\.\d{3})*,\d{2}(?![\d,])')
date_re  = re.compile(
    r'^\s*(?:'
    r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)'
    r'|'
    r'(\d{1,2})\s*'
    r'(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)'
    r'(?:[\s/](\d{2,4}))?'
    r')\b',
    re.I
)
month_map = {'jan':'01','fev':'02','mar':'03','abr':'04','mai':'05','jun':'06',
             'jul':'07','ago':'08','set':'09','out':'10','nov':'11','dez':'12'}

def norm_date(m: re.Match) -> str | None:
    if not m:
        return None
    if m.group(1):
        d, m_, *y = m.group(1).split('/')
        d = d.zfill(2); m_ = m_.zfill(2)
        y = y[0] if y else ''
        if y and len(y) == 2:
            y = '20' + y
        return f'{m_}/{d}/{y}' if y else f'{m_}/{d}'
    else:
        d = m.group(2).zfill(2)
        m_ = month_map[m.group(3).lower()]
        y = m.group(4) or ''
        if y and len(y) == 2:
            y = '20' + y
        return f'{m_}/{d}/{y}' if y else f'{m_}/{d}'

@app.post('/extract-colored-transactions/')
async def extract_colored_transactions(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')

    color_hist = Counter()
    lines = []
    for page in doc:
        for block in page.get_text('dict')['blocks']:
            if block['type'] != 0:
                continue
            for line in block['lines']:
                txts, cols = [], []
                for span in line['spans']:
                    txts.append(span['text'])
                    cols.append(span['color'])
                    color_hist[span['color']] += 1
                if txts:
                    lines.append((''.join(txts), txts, cols))

    dominant = color_hist.most_common(1)[0][0] if color_hist else None
    if dominant is None:
        return {'transactions': []}

    out = []
    for full, parts, colors in lines:
        dm = date_re.match(full)
        if not dm:
            continue
        purchase_date = norm_date(dm)
        desc_start = dm.end()
        pos = 0
        for part, c in zip(parts, colors):
            part_start = pos
            pos += len(part)
            if c == dominant:
                continue
            mval = money_re.search(part)
            if not mval:
                continue
            abs_val_start = part_start + mval.start()
            description = full[desc_start:abs_val_start].strip()
            neg = mval.group(0).strip().startswith(('-', '−', '–', '—'))
            num = re.sub(r'[+\-−–—\s]', '', mval.group(0)).replace('.', '').replace(',', '.')
            try:
                transaction_value = (-1 if neg else 1) * float(num)
            except ValueError:
                continue
            out.append({
                'purchase_date'    : purchase_date,
                'description'      : description,
                'transaction_value': round(transaction_value, 2)
            })
            break

    return {'transactions': out}

