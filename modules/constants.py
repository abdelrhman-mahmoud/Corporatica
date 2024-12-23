import os

PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'db')
CHROMA_SETTINGS = {
    "persist_directory": PERSIST_DIRECTORY,
    "anonymized_telemetry": False
}

SOURCE_DIRECTORY = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# CSV file that tracks all agents
OVERVIEW_FILEPATH = os.path.join('data', 'overview.csv')
