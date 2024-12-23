# RAG Agent Streamlit App

A demonstration of a Retrieval-Augmented Generation (RAG) chatbot built with [LangChain](https://github.com/hwchase17/langchain), [ollama](https://github.com/jmorganca/ollama), and [Streamlit](https://github.com/streamlit/streamlit).

## Features
1. **Agent Management**  
   Easily create or load agents with a custom name, description, and opening message.

2. **Document Upload & Ingestion**  
   Upload files in multiple formats (.pdf, .txt, .docx, .md, etc.). Automatically process and store them in a vector database (Chroma).

3. **Contextual Q&A**  
   Asks questions and retrieves relevant chunks from your documents to craft the final response. Provides clarifications or direct answers when necessary.

4. **Memory & History**  
   Maintains short-term conversation history for contextualizing user queries.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/abdelrhman-mahmoud/Corporatica.git
   cd Corporatica
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install ollama**

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

4. **Pull Required Models**

   ```bash
   # Pull the main LLaMA-based model
   !ollama pull llama3

   # Pull the embedding model
   !ollama pull mxbai-embed-large
   ```

## Usage

1. **Start the Streamlit App**

   ```bash
   streamlit run app.py
   ```

2. **Open Your Browser**

   Go to [http://localhost:8501/](http://localhost:8501/) (Streamlit's default).  
   In the sidebar, configure your agent name, description, and starter message, then click **Create / Load Agent**.

3. **Upload Documents**

   Still in the sidebar, upload files to index, then click **Ingest Documents** to parse, split, and embed them into Chroma DB.

4. **Ask Questions**

   Type your query in the main page's text input. Click **Ask** to get an answer from the agent, optionally with sources shown in an expandable section.

## File Descriptions

```
my_streamlit_app/
├── app.py
├── requirements.txt
└── modules/
    ├── __init__.py
    ├── constants.py
    ├── templates.py
    ├── bot.py
    ├── utils.py
    ├── ingestion.py
    └── agent.py
```

- **`app.py`**  
  The main Streamlit entry point. Handles UI elements such as the sidebar for file upload and agent config, plus the chat input/output.

- **`requirements.txt`**  
  List of Python dependencies needed for this project (Streamlit, langchain, sentence-transformers, etc.).

- **`modules/`**  
  A Python package containing:
  1. **`constants.py`**  
     Centralizes paths, chunking parameters, and references to `OVERVIEW_FILEPATH`.
  2. **`templates.py`**  
     Contains Jinja2 templates and prompt strings for conversation logic.
  3. **`bot.py`**  
     A `Bot` class that manages per-bot source & database directories.
  4. **`utils.py`**  
     Utility functions for loading, cleaning, and reading/writing files.
  5. **`ingestion.py`**  
     Logic to parse and embed documents, storing them in a Chroma DB.
  6. **`agent.py`**  
     The core agent logic for question-answering and retrieval.

## Tips

1. **GPU Memory**  
   - If you have limited GPU memory (<8GB), you may run out of VRAM during re-ranking or inference. You can force re-ranker or LLM to run on CPU, or use smaller models.
2. **Empty Retrieval**  
   - If no documents are uploaded or no relevant results are found, the agent falls back to direct answers or clarifications.
3. **System Requirements**  
   - Tested on Python 3.12. Make sure your environment meets [ollama's prerequisites](https://github.com/jmorganca/ollama).

## License

You may include a license of your choice here, like MIT License or Apache 2.0.

---

**Happy Chatting!**
