import streamlit as st
import fitz 
import re
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return summarizer, qa_model, embeddings

summarizer, qa_model, embeddings = load_models()


def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"

    import re
    match = re.search(r'\n(REFERENCES|References|BIBLIOGRAPHY|Bibliography)\b', text)
    if match:
        text = text[:match.start()]
        
    return text


def generate_summary(text, detail_level, progress_bar):
    if detail_level == "Short (TL;DR)":
        max_words, min_words = 100, 30
    elif detail_level == "Medium":
        max_words, min_words = 200, 60
    else:
        max_words, min_words = 350, 100

    chunk_size = 3000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    combined_summary = []
    

    for i, chunk in enumerate(chunks):
        try:
            chunk_length = len(chunk.split())
            safe_max = min(max_words, max(10, int(chunk_length * 0.8)))
            safe_min = min(min_words, max(5, int(safe_max * 0.5)))

            result = summarizer(chunk, max_length=safe_max, min_length=safe_min, do_sample=False)
            combined_summary.append(result[0]['summary_text'])
        except Exception as e:
            print(f"Skipping chunk {i} due to error: {e}")
            
        progress_bar.progress((i + 1) / len(chunks))
            
    final_output = "\n\n".join([f"📌 {summary}" for summary in combined_summary])
    return final_output

@st.cache_resource
def build_vector_database(text):
    chunk_size = 1000 
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def ask_question(question, vectorstore):
    relevant_docs = vectorstore.similarity_search(question, k=5)
    context = " ".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""Based on the following context, answer the user's question.
    
    Rules:
    1. If asked for a definition or "What is...", explain it fully in a complete sentence.
    2. If asked for author names or a list of tools, provide a clean list without superscript numbers or affiliations.
    3. For all other questions, provide a detailed, multi-sentence explanation.
    4. If the answer is not in the context, say 'The paper does not mention this.'

    Context: {context}

    Question: {question}

    Answer:"""
    
    result = qa_model(prompt, max_length=200, min_length=5, do_sample=False)
    
    return result[0]['generated_text']

st.set_page_config(page_title="AI Paper Summarizer", layout="centered")

st.title("📄 AI Research Paper Summarizer")
st.write("Upload an academic paper (PDF) to get an abstractive summary using BART.")

detail_level = st.select_slider(
    "Select Summary Detail Level:",
    options=["Short (TL;DR)", "Medium", "Detailed (Full Breakdown)"],
    value="Medium"
)

uploaded_pdf = st.file_uploader("Upload your PDF here", type=["pdf"])

if uploaded_pdf is not None:
    with st.spinner("Extracting and cleaning text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_pdf)
        st.session_state.raw_text = raw_text
        
    st.success("Text extracted successfully!")
    
    with st.expander("View Extracted Raw Text"):
        st.write(raw_text[:1000] + "... [Text Truncated for View]")

    if st.button("Generate Summary"):
        progress_text = "AI is reading and summarizing the full document..."
        st.write(progress_text)
        my_bar = st.progress(0)
        
        summary = generate_summary(raw_text, detail_level, my_bar)
        st.session_state.summary = summary
            
    if 'summary' in st.session_state:
        st.subheader("💡 Comprehensive AI Summary")
        st.write(st.session_state.summary)
        st.divider()
        st.download_button(
            label="📥 Download Summary as .TXT",
            data=st.session_state.summary,
            file_name="Detailed_Paper_Summary.txt",
            mime="text/plain"
        )
    if 'raw_text' in st.session_state:
        st.divider()
        st.subheader("💬 Chat with this Paper (RAG)")
        st.write("Ask specific questions about the methodology, datasets, or conclusions.")
        
        with st.spinner("Indexing document for Q&A..."):
            if 'vectorstore' not in st.session_state:
                st.session_state.vectorstore = build_vector_database(st.session_state.raw_text)
        
        user_question = st.text_input("What would you like to know?")
        
        if user_question:
            with st.spinner("Searching document and generating answer..."):
                answer = ask_question(user_question, st.session_state.vectorstore)
                st.info(f"**Answer:** {answer}")