import streamlit as st
from pdf_utils import extract_text_from_pdf, clean_text, chunk_text
from db_utils import setup_collection, insert_documents, search
from query_utils import ask_llama

st.set_page_config(page_title="Knowledge Assistant", layout="wide")
st.title("Knowledge Assistant")

# --- Layout with Two Columns ---
col1, col2 = st.columns([3, 1])  # 3 parts left, 1 part right (for file upload)

# --- Left Column for Chat ---
with col1:
    st.subheader("Ask Questions About the Document")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    query = st.chat_input("Type your question here...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Generating response..."):
            relevant_chunks = search(query)
            context = "\n\n".join(relevant_chunks)
            answer = ask_llama(context, query)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Right Column for Document Upload ---
with col2:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file:
        with st.spinner("Extracting and processing content..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            clean = clean_text(raw_text)
            chunks = chunk_text(clean)

            setup_collection()
            insert_documents(chunks)

        st.success("PDF processed and indexed successfully.")
