import os
import re
import aiofiles
from datetime import datetime
import glob
from typing import List
from langchain.docstore.document import Document

from . import constants
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

async def write_file(file_content: bytes, file_path: str):
    async with aiofiles.open(file_path, 'wb') as out_file:
        await out_file.write(file_content)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s\'\".,?@:/]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text)
    for date in dates:
        standardized_date = datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d')
        text = text.replace(date, standardized_date)
    return text

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        docs = loader.load()
        if ext in ['.pdf', '.html', '.txt']:
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
        return docs
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files=None) -> List[Document]:
    if ignored_files is None:
        ignored_files = []
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    documents = []
    for file_path in all_files:
        if file_path in ignored_files:
            continue
        documents.extend(load_single_document(file_path))
        print(f"Loaded document {file_path}")
    return documents
