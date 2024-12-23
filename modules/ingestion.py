import os
import glob
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

from .constants import CHROMA_SETTINGS, CHUNK_SIZE, CHUNK_OVERLAP
from .utils import load_documents
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings

def process_documents(source_folder: str, embedding_model=None, ignored_files=None) -> List[Document]:
    print(f"Loading documents from {source_folder}")
    documents = load_documents(source_folder, ignored_files)
    if not documents:
        print("No new documents to load")
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    # Optionally do semantic chunking
    if embedding_model:
        text_splitter_semantic = SemanticChunker(embedding_model)
        documents = text_splitter_semantic.split_documents(documents)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {CHUNK_SIZE} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and \
           os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            if len(list_index_files) > 3:
                return True
    return False

def ingest(db_folder: str, source_folder: str):
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
    if does_vectorstore_exist(db_folder):
        print(f"Appending to existing vectorstore at {db_folder}")
        db = Chroma(persist_directory=db_folder, embedding_function=embedding_model, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents(
            source_folder, 
            embedding_model=embedding_model, 
            ignored_files=[metadata['source'] for metadata in collection['metadatas']]
        )
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        print("Creating new vectorstore")
        texts = process_documents(source_folder, embedding_model=embedding_model)
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embedding_model, persist_directory=db_folder)
    db.persist()
    db = None
