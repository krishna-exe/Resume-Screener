"""
This module contains functions for performing mathematical operations.
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_pdf_names(pdf_docs):
    total_pdfs = len(pdf_docs)
    pdf_names = []
    for i in range(total_pdfs):
        pdf_names.append((i, pdf_docs[i].name))
    return pdf_names


def get_pdf_text(pdf_docs):
    pdf_text = []
    for pdf in pdf_docs:
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        pdf_text.append(text)
    return pdf_text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100000,
        chunk_overlap=0,
        length_function=len
    )
    chunks = []
    for txt in text:
        chunk = text_splitter.split_text(txt)
        chunks.extend(chunk)
    return chunks


def get_vectorstore(text_chunks, job_description):
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="my_collection")
    #metadata = ["mysource" for i in range(len(text_chunks))]
    collection.add(
        documents=[f"{text_chunks[i]}" for i in range(len(text_chunks))],
        ids=[f"{k}" for k in range(len(text_chunks))]
    )
    query1 = job_description
    results = collection.query(
        query_texts=query1,
        n_results=len(text_chunks)
    )
    return results


def percentage(match):
    res = (1/(1+match))*100
    return f'{res:.2f}%'


def get_rank_table(vectorstore, pdf_names):
    rank_table = {"Rank": [], 
                  "Name": [], 
                  "Match": []
                  }
    for i in range(len((vectorstore["ids"])[0])):
        l2_dist = (vectorstore["distances"][0])[i]
        id = int((vectorstore["ids"][0])[i])
        rank_table["Rank"].append(f'{i+1}')
        rank_table["Name"].append(f'{pdf_names[id][1]}')
        rank_table["Match"].append(f'{percentage(l2_dist)}')
    return rank_table

def create_rank_table(rank_table):
    rank_frame = rank_table
    st.dataframe(rank_frame, use_container_width=True)
    return rank_frame

def download_rank_table(rank_table):
    rank_table = pd.DataFrame(rank_table)
    csv_button = st.download_button(
                    label='Download CSV',
                    data=rank_table.to_csv(index=False),
                    file_name='data.csv',
                    mime='text/csv'
)


def normalize_rank_table(rank_table):
    # use this as an extra feature button
    pass

def main():
   
    st.set_page_config(page_title="Resume Ranker", page_icon=":books:")

    if "rank_table" not in st.session_state:
        st.session_state.rank_table = None

    st.header("Resume Ranker :books:")
    job_description = st.text_input("Enter Job Description: ")

    with st.sidebar:
        st.subheader("Analyze Resume")
        pdf_docs = st.file_uploader(
            "Input Resume and click Analyze", accept_multiple_files=True)

    if st.button("Analyze"):
        with st.spinner("Processing"):
            # get pdf names
            pdf_names = get_pdf_names(pdf_docs)

            # get pdf text
            raw_text = get_pdf_text(pdf_docs)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            # st.write(text_chunks)

            # create vector store
            if job_description:
                vectorstore = get_vectorstore(text_chunks, job_description)

            rank = get_rank_table(vectorstore, pdf_names)
            
            # create the table for display
            rank_frame = create_rank_table(rank)

            download_rank_table(rank_frame)

            


if __name__ == "__main__":
    main()