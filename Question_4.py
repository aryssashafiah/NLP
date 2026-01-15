import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader

# Ensure punkt is available
nltk.download("punkt", quiet=True)

st.set_page_config(page_title="Q4 PDF Sentence Chunker (NLTK)", layout="wide")
st.title("Q4: PDF Sentence Chunker (NLTK)")

st.write(
    "Upload a PDF file, extract text, then split into sentences using NLTK "
    "and display indices **58 to 68**. Finally, apply sentence chunking on that sample."
)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        # Step 1 & 2: Read PDF and extract text
        reader = PdfReader(uploaded_file)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)

        full_text = " ".join(pages_text).strip()

        st.subheader("Basic info")
        st.write(f"Number of pages: **{len(reader.pages)}**")
        st.write(f"Total characters extracted: **{len(full_text)}**")

        if not full_text:
            st.warning("No text could be extracted from this PDF.")
            st.stop()

        # Step 3: Sentence tokenization on full text
        sentences = sent_tokenize(full_text)
        st.success(f"Number of detected sentences: {len(sentences)}")

        # Required fixed range (58 to 68 inclusive)
        start_idx, end_idx = 58, 68

        st.subheader("Step 3: Sample sentences (indices 58 to 68)")
        if len(sentences) <= start_idx:
            st.error(
                f"This PDF has only {len(sentences)} sentences, so indices 58–68 are not available. "
                "Please upload a PDF with more text."
            )
            st.stop()

        sample_sents = sentences[start_idx : min(end_idx + 1, len(sentences))]

        for i, s in enumerate(sample_sents, start=start_idx):
            st.markdown(f"**[{i}]** {s}")

        # Step 4: Apply sentence tokenizer chunking on Step 3 text
        st.subheader("Step 4: Semantic sentence chunking (NLTK) on the sample")
        sample_text = " ".join(sample_sents)

        # Re-tokenize the sample (chunking)
        chunks = sent_tokenize(sample_text)

        for j, c in enumerate(chunks, start=1):
            st.markdown(f"- **Chunk {j}:** {c}")

        with st.expander("Show raw extracted text (first 2000 characters)"):
            st.text(full_text[:2000])

    except Exception as e:
        st.error(f"Error reading PDF: {e}")

else:
    st.info("Please upload a PDF to begin.")
