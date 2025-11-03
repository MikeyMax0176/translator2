import io, re
import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from transformers import MarianMTModel, MarianTokenizer

st.set_page_config(page_title="Spanish → English PDF Translator", layout="wide")
st.title("📘 Spanish → English Translator")

@st.cache_resource
def load_model():
    name = "Helsinki-NLP/opus-mt-es-en"
    model = MarianMTModel.from_pretrained(name)
    tok = MarianTokenizer.from_pretrained(name)
    return model, tok

def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [p.get_text("text") for p in doc]
    doc.close()
    return "\n\n".join(pages)

def chunkify(text: str, max_chars: int = 900):
    bits = re.split(r'(?<=[\.\!\?\n])\s+', text.strip())
    chunks, cur = [], ""
    for b in bits:
        if len(cur) + len(b) + 1 <= max_chars:
            cur = (cur + " " + b).strip()
        else:
            if cur: chunks.append(cur)
            cur = b
    if cur: chunks.append(cur)
    return chunks

def translate_chunks(chunks, model, tok, progress=None):
    out, total = [], len(chunks) or 1
    for i, ch in enumerate(chunks, 1):
        batch = tok([ch], return_tensors="pt", padding=True, truncation=True)
        gen = model.generate(**batch, max_new_tokens=1024)
        out.append(tok.decode(gen[0], skip_special_tokens=True))
        if progress: progress.progress(i/total)
    return "\n\n".join(out)

# ---- Exports ----
def to_docx(translated_text: str) -> bytes:
    d = Document()
    for para in translated_text.split("\n\n"):
        d.add_paragraph(para)
    bio = io.BytesIO(); d.save(bio); bio.seek(0); return bio.getvalue()

def to_pdf_reflow(translated_text: str, page_w=595, page_h=842, margin=36, fontname="helv", fontsize=11) -> bytes:
    pdf = fitz.open()
    page = pdf.new_page(width=page_w, height=page_h)
    rect = fitz.Rect(margin, margin, page_w - margin, page_h - margin)
    text = translated_text
    while text:
        rest = page.insert_textbox(rect, text, fontname=fontname, fontsize=fontsize, color=(0,0,0), align=0)
        if rest:
            page = pdf.new_page(width=page_w, height=page_h)
            text = rest
        else:
            break
    bio = io.BytesIO(); pdf.save(bio); pdf.close(); bio.seek(0); return bio.getvalue()

def overlay_pdf(pdf_bytes: bytes, translate_fn, fontname="helv", fontsize=11, cover_bg=True, alpha=1.0) -> bytes:
    """Keep original graphics; overlay English on source text blocks."""
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = fitz.open()
    for i in range(len(src)):
        sp = src[i]
        dp = out.new_page(width=sp.rect.width, height=sp.rect.height)
        dp.show_pdf_page(dp.rect, src, i)  # draw original page
        for blk in sp.get_text("blocks"):
            if len(blk) < 5: continue
            x0,y0,x1,y1, txt = blk[:5]
            if not txt or not txt.strip(): continue
            en = translate_fn(txt.strip())
            if not en: continue
            rect = fitz.Rect(x0,y0,x1,y1)
            if cover_bg:
                sh = dp.new_shape(); sh.draw_rect(rect); sh.finish(fill=(1,1,1), color=None); sh.commit()
            dp.insert_textbox(rect, en, fontname=fontname, fontsize=fontsize, color=(0,0,0),
                              align=0, fill_opacity=alpha)
    bio = io.BytesIO(); out.save(bio); out.close(); src.close(); bio.seek(0); return bio.getvalue()

# ---- UI ----
uploaded = st.file_uploader("Upload a Spanish PDF", type=["pdf"])
layout_choice = st.selectbox("Output layout", ["Simple reflow (TXT/DOCX/PDF/HTML)", "Overlay English onto original (PDF)"])
fmt = []
if layout_choice == "Simple reflow (TXT/DOCX/PDF/HTML)":
    fmt = st.multiselect("Choose export formats", ["TXT","DOCX","PDF (reflowed)","HTML"], default=["TXT","DOCX","PDF (reflowed)"])
else:
    st.info("This mode keeps original graphics and overlays English text on top.")

if uploaded:
    raw = uploaded.read()
    model, tok = load_model()
    def tr_fn(s: str) -> str:
        b = tok([s], return_tensors="pt", padding=True, truncation=True)
        g = model.generate(**b, max_new_tokens=1024)
        return tok.decode(g[0], skip_special_tokens=True)

    if layout_choice == "Overlay English onto original (PDF)":
        st.info("Extracting and translating by blocks…")
        over = overlay_pdf(raw, tr_fn, fontname="helv", fontsize=11, cover_bg=True, alpha=1.0)
        st.success("Done!")
        st.download_button("⬇️ Download translated_overlay.pdf", over, file_name="translated_overlay.pdf", mime="application/pdf")
    else:
        st.info("Extracting text…")
        text = extract_text(raw)
        st.info("Translating…")
        bar = st.progress(0.0)
        translated = translate_chunks(chunkify(text), model, tok, progress=bar)
        st.success("Done!")

        if "TXT" in fmt:
            st.download_button("⬇️ TXT", translated.encode("utf-8"), "translation.txt")
        if "DOCX" in fmt:
            st.download_button("⬇️ DOCX", to_docx(translated),
                               "translation.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        if "PDF (reflowed)" in fmt:
            st.download_button("⬇️ PDF (reflowed)", to_pdf_reflow(translated),
                               "translation.pdf", mime="application/pdf")
        if "HTML" in fmt:
            html = f"""<!doctype html><meta charset="utf-8"><style>body{{font-family:system-ui,Arial;margin:2rem;line-height:1.5;}}</style>{''.join(f'<p>{p}</p>' for p in translated.split('\\n\\n'))}"""
            st.download_button("⬇️ HTML", html.encode("utf-8"), "translation.html", mime="text/html")
        st.text_area("Preview (first 8k chars)", translated[:8000], height=300)
