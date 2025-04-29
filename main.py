from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import fitz  # PyMuPDF
import io
import zipfile

app = FastAPI()

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...), password: str = Form(...)):
    contents = await file.read()
    pdf_document = fitz.open(stream=contents, filetype="pdf")
    
    if pdf_document.needs_pass:
        if not pdf_document.authenticate(password):
            return {"error": "Incorrect password"}
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            zip_file.writestr(f"page_{page_number + 1}.png", img_data)
    
    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/x-zip-compressed", headers={
        "Content-Disposition": "attachment; filename=images.zip"
    })
