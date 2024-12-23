import os
import csv
import asyncio
from typing import Dict, List

from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from .bot import get_db_dir
from .constants import OVERVIEW_FILEPATH
from .templates import Prompts, CustomTemplates
from . import ingestion  # for potential references
from . import constants

# We keep a session cache in memory
session_cache = {}

class Agent(BaseModel):
    bot_id: str
    name: str
    description: str
    starter: str
    model: str

def load_model(model_name: str) -> OllamaLLM:
    return OllamaLLM(
        model=model_name,
        device='cpu',
        callbacks=[StreamingStdOutCallbackHandler()],
    )

embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

def load_retriever(bot_id: str):
    return Chroma(
        persist_directory=get_db_dir(bot_id),
        embedding_function=embedding_model
    ).as_retriever(search_kwargs={"k": 10})

def rerank_docs(reranker_model: CrossEncoder, query: str, retrieved_docs: List[Document]) -> List[tuple]:
    """
    Re-rank retrieved_docs based on CrossEncoder scores.

    :param reranker_model: CrossEncoder instance
    :param query: the query string
    :param retrieved_docs: list of Documents to be re-ranked
    :return: list of (Document, float) sorted by score descending
    """
    if not retrieved_docs:
        # Return empty if there are no documents
        return []
    query_and_docs = [(query, r.page_content) for r in retrieved_docs]
    scores = reranker_model.predict(query_and_docs)
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)

async def load_agent(agent: Agent):
    session_cache[agent.bot_id] = {
        'name': agent.name,
        'description': agent.description,
        'starter': agent.starter,
        'llm': load_model(agent.model),
        'retriever': load_retriever(agent.bot_id),
        'reranker': CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1",device='cpu'),
        'history': [],
        'prompts': Prompts(agent.model),
        'jinja_templates': CustomTemplates(agent.model)
    }

def find_bot_by_id(bot_id: str) -> Agent:
    if not os.path.exists(OVERVIEW_FILEPATH):
        return None
    with open(OVERVIEW_FILEPATH, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['bot_id'] == bot_id:
                return Agent(**row)
    return None

async def create_agent(agent: Agent):
    # Load agent into session_cache
    await load_agent(agent)
    # Append agent details to overview.csv
    if not os.path.exists('data'):
        os.makedirs('data')
    with open(OVERVIEW_FILEPATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        # If file is empty, write header
        if file.tell() == 0:
            writer.writerow(['bot_id', 'name', 'description', 'starter', 'model'])
        writer.writerow([agent.bot_id, agent.name, agent.description, agent.starter, agent.model])

def preview_agent(agent: Agent) -> str:
    if agent.bot_id not in session_cache:
        asyncio.run(load_agent(agent))
    # Clear out old chat history so each "preview" starts fresh
    session_cache[agent.bot_id]['history'] = []
    return session_cache[agent.bot_id]['starter']

async def on_message(bot_id: str, content: str) -> Dict[str, List[str]]:
    """Route the user message, retrieve relevant docs, answer from them or clarify."""
    print('\nBEGIN PROCESS\n')
    
    # Load from CSV if not in memory
    if bot_id not in session_cache:
        agent = find_bot_by_id(bot_id)
        if agent:
            await load_agent(agent)
        else:
            raise ValueError(f"Bot with id {bot_id} does not exist")

    bot_info = session_cache[bot_id]
    llm = bot_info['llm']
    retriever = bot_info["retriever"]
    reranker = bot_info["reranker"]
    history = bot_info['history']
    prompts = bot_info['prompts']
    jinja_templates = bot_info['jinja_templates']

    # Contextualize if there's prior chat
    if len(history) > 0:
        rendered_history = jinja_templates.render_template(history)
        content = llm.invoke(
            prompts.contextualize_in_history.format(chat_history=rendered_history, query=content),
            stop=['<|eot_id|>']
        )
        print('\nDONE CONTEXTUALIZING\n')

    source_documents = None
    route = llm.invoke(prompts.route_query.format(query=content), stop=['<|eot_id|>'])
    print('\nDONE ROUTING\n')

    if route.strip().startswith('DOCS'):
        # Retrieve from Chroma
        retrieved_docs = retriever.get_relevant_documents(
            f"Represent this sentence for searching relevant passages: {content}"
        )
        print('\nDONE RETRIEVED\n')

        # Check if no docs were retrieved
        if not retrieved_docs:
            answer = (
                "No relevant documents were found. "
                "Please ensure you've ingested documents or re-check your question."
            )
            history.append({"role": "user", "content": content})
            history.append({"role": "assistant", "content": answer})
            return {"response": answer, "sources": []}

        # Re-rank
        reranked_docs = rerank_docs(reranker, content, retrieved_docs)
        print('\nDONE RERANKING\n')
        
        if not reranked_docs:
            answer = (
                "No relevant documents after re-ranking. "
                "Please try rephrasing your question or upload additional documents."
            )
            history.append({"role": "user", "content": content})
            history.append({"role": "assistant", "content": answer})
            return {"response": answer, "sources": []}

        source_documents = [doc[0] for doc in reranked_docs[:4]]
        docs_content = '\n'.join([doc.page_content for doc in source_documents])

        # Check relevancy
        relevancy = llm.invoke(
            prompts.sort_relevancy.format(context=docs_content, query=content),
            stop=['<|eot_id|>']
        )
        print('\nDONE RELEVANCY\n')

        if relevancy.strip().startswith('YES'):
            answer = llm.invoke(prompts.qa_from_docs.format(context=docs_content, query=content), stop=['<|eot_id|>'])
        else:
            # Clarify
            answer = llm.invoke(prompts.clarify.format(context=docs_content, query=content), stop=['<|eot_id|>'])
    else:
        # DEFAULT route—just answer directly
        answer = llm.invoke(content, stop=['<|eot_id|>'])

    # Track conversation
    history.append({"role": "user", "content": content})
    history.append({"role": "assistant", "content": answer})
    bot_info['history'] = history[-8:]  # keep last 8 messages

    sources = [doc.page_content for doc in source_documents] if source_documents else []
    return {"response": answer, "sources": sources}

def overview() -> List[Agent]:
    agents = []
    if not os.path.exists(OVERVIEW_FILEPATH):
        return agents
    with open(OVERVIEW_FILEPATH, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            agents.append(Agent(**row))
    return agents
