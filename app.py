import os
import tempfile

import fitz
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("Ask Questions About Your PDFs")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Set GEMINI_API_KEY in your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


def read_pdf(uploaded_file):
    chunks = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(uploaded_file.getvalue())
        path = temp.name

    try:
        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                start = 0

                while start < len(text):
                    end = min(start + 1000, len(text))
                    chunk = text[start:end].strip()

                    if chunk:
                        chunks.append({
                            "text": chunk,
                            "file": uploaded_file.name,
                            "page": page_number,
                        })

                    if end == len(text):
                        break

                    start = end - 150
    finally:
        os.remove(path)

    return chunks


def find_relevant(question, chunks):
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, normalize_embeddings=True)
    question_vector = model.encode([question], normalize_embeddings=True)
    scores = cosine_similarity(question_vector, vectors)[0]
    best_indices = scores.argsort()[::-1][:5]

    return [chunks[index] for index in best_indices]


files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
)

if files:
    with st.spinner("Reading PDFs..."):
        chunks = []
        for uploaded_file in files:
            chunks.extend(read_pdf(uploaded_file))

    if not chunks:
        st.warning("No readable text found. Scanned PDFs need OCR.")
        st.stop()

    st.success("PDFs are ready.")
    question = st.text_input("Type your question")

    if question:
        with st.spinner("Generating answer..."):
            passages = find_relevant(question, chunks)

            context = "\n\n".join(
                f"[{item['file']}, page {item['page']}]\n{item['text']}"
                for item in passages
            )

            prompt = f"""
Answer the question using only the PDF passages below.
If the passages do not contain the answer, say you could not find it.
Cite the file name and page number.

Question: {question}

PDF passages:
{context}
"""

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )

        st.subheader("Answer")
        st.write(response.text)

        with st.expander("Show source passages"):
            for item in passages:
                st.markdown(f"**{item['file']} — page {item['page']}**")
                st.write(item["text"])