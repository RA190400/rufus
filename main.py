# main.py

import streamlit as st
from rufus import Rufus
import json
# Initialize the Rufus class
rag_system = Rufus()

# Streamlit UI
st.title("Rufus: Web Data Extraction and Retrieval System")
st.subheader("Extract relevant information from web pages based on your questions.")

# UI layout with columns for better organization
col1, col2 = st.columns(2)
with col1:
    target_url = st.text_input("🔗 Enter the URL of the website to scrape")
with col2:
    file_format = st.selectbox("📄 Select file format for saving answers", ["json", "docx"])

question = st.text_area("❓ Enter your question/prompt")

# Button to start the RAG system
if st.button("Run Retrieval"):
    if target_url and question:
        with st.spinner("Running retrieval..."):
            # Run the RAG system and capture results
            rag_system.run(target_url, question, file_format)

            # Load results from the file generated
            results = []
            if file_format == "json":
                with open("rag_results.json", "r") as f:
                    results = json.load(f)
            elif file_format == "docx":
                # Streamlit does not natively support DOCX rendering; outputting a link for download only
                results = "Results saved to DOCX. Use the download link below."

            # Display results directly in Streamlit
            if isinstance(results, list) and results:
                for entry in results:
                    st.write("### Question:")
                    st.write(entry['question'])
                    st.write("### Answer:")
                    st.write(entry['answer'])
                    st.markdown("---")  # Divider for readability
            else:
                st.write("No results found.")

            # Display download link
            if file_format == "json":
                st.download_button(
                    "Download Results as JSON",
                    data=open("rag_results.json", "rb"),
                    file_name="rag_results.json",
                    mime="application/json",
                )
            elif file_format == "docx":
                st.download_button(
                    "Download Results as DOCX",
                    data=open("rag_results.docx", "rb"),
                    file_name="rag_results.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
    else:
        st.warning("Please provide both the URL and a question to proceed.")
