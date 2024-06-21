from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import GPT4AllEmbeddings
from langchain_community.llms import GPT4All
from langchain.vectorstores.chroma import Chroma
import argparse
from dataclasses import dataclass
#from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import shutil

# Option 1: Set the environment variable
#import os

#os.environ["OPENAI_API_KEY"] = "sk-proj-mxyP5veNzTJiHOi18EyET3BlbkFJn0wUJBwvyKy8nxRleIDs"

# Create the GPT4AllEmbedding object

embeddings = GPT4AllEmbeddings()
#vector = embeddings.embed_query("Hello world")
#print(vector)
#print(len(vector))

CHROMA_PATH = 'chroma'
DATA_PATH = 'data\Books'

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Create a new DB from the documents
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()