import os
import uuid
import shutil

class Bot:
    def __init__(self) -> None:
        self.bot_id = str(uuid.uuid4())
        self.source_dir = os.path.join('data', 'source_documents', self.bot_id)
        self.db_dir = os.path.join('data', 'vector_db', self.bot_id)
        self.create_dir(self.source_dir)
        self.create_dir(self.db_dir)

    @staticmethod
    def create_dir(directory: str):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

def get_db_dir(bot_id: str) -> str:
    return os.path.join('data', 'vector_db', bot_id)
