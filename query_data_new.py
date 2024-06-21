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

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

embeddings = GPT4AllEmbeddings()
CHROMA_PATH = 'chroma'
DATA_PATH = 'data\Books'


def main():
    query_text = input("Enter query text : ")

    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0: #or results[0][1] < 0.7:
        print(len(results))
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Instantiate the model. Callbacks support token-wise streaming
    model = GPT4All(model="./models/mistral-7B-v0.1.tar", n_threads=8)
    response = model.invoke("Once upon a time, ")

   
    response_text = model.predict(prompt)
    

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()