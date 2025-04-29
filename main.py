from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import fitz  # PyMuPDF
import io
from PIL import Image

app = FastAPI()

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...), password: str = Form(...)):
    contents = await file.read()
    pdf_document = fitz.open(stream=contents, filetype="pdf")
    
    if pdf_document.needs_pass:
        if not pdf_document.authenticate(password):
            return {"error": "Incorrect password"}

    images = []
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    # Combine images vertically
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)
    combined_image = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save to buffer
    output_buffer = io.BytesIO()
    combined_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    return StreamingResponse(output_buffer, media_type="image/png", headers={
        "Content-Disposition": "attachment; filename=combined.png"
    })

