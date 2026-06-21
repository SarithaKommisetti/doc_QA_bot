import streamlit as st
from src.query import query_rag

st.set_page_config(page_title="Document Q&A Bot")

st.title("📄 RAG Document Q&A Bot")

question = st.text_input("Ask a question from your documents")

if st.button("Get Answer"):
    if question.strip():
        result = query_rag(question)
        st.subheader("Answer")
        st.write(result) 