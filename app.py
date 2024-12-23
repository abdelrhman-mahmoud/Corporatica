import streamlit as st
import os
import asyncio

# Local module imports
from modules.bot import Bot
from modules.agent import Agent, create_agent, preview_agent, on_message, overview
from modules.ingestion import ingest
from modules.constants import OVERVIEW_FILEPATH
from modules.utils import load_single_document

def upload_documents(bot: Bot, file_paths):
    """Copy files to bot.source_dir and call ingest."""
    import shutil
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(bot.source_dir, filename)
        shutil.copy(file_path, dest_path)
    print("Uploading Complete")
    ingest(bot.db_dir, bot.source_dir)
    print("Ingestion Complete")

def main():
    st.title("RAG Agent Demo")

    if "bot" not in st.session_state:
        st.session_state["bot"] = Bot()

    # Sidebar
    with st.sidebar:
        st.header("Configure Agent")
        agent_name = st.text_input("Agent Name", value="Assistant Bot")
        agent_description = st.text_input("Agent Description", value="RAG agent for Corpotatica.")
        agent_starter = 'Hello! How can I assist you today?'

        model_name = st.selectbox("Model Name", options=["llama3"], index=0)

        create_button = st.button("Create / Load Agent")

        st.write("---")
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, 
                                          type=["pdf","txt","doc","docx","md","csv","ppt","pptx","html","odt"])
        ingest_button = st.button("Ingest Documents")

    if create_button:
        new_agent = Agent(
            bot_id=st.session_state["bot"].bot_id,
            name=agent_name,
            description=agent_description,
            starter=agent_starter,
            model=model_name
        )
        asyncio.run(create_agent(new_agent))
        st.success(f"Agent {new_agent.name} created with ID: {new_agent.bot_id}")
        starter_msg = preview_agent(new_agent)
        st.session_state["agent"] = new_agent
        st.info(starter_msg)

    if "agent" not in st.session_state:
        st.warning("Create or load an agent first.")
        return

    if ingest_button and uploaded_files:
        local_file_paths = []
        temp_dir = "temp_documents"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            local_file_paths.append(file_path)
        upload_documents(st.session_state["bot"], local_file_paths)
        st.success("Files ingested successfully!")

    st.write("---")
    st.header("Chat")
    user_question = st.text_input("Enter your question", value="")

    if st.button("Ask"):
        if user_question.strip():
            response_data = asyncio.run(on_message(st.session_state["bot"].bot_id, user_question))
            st.write("### Bot Response")
            st.write(response_data["response"])
            if response_data["sources"]:
                with st.expander("Sources"):
                    for idx, src in enumerate(response_data["sources"], start=1):
                        st.write(f"**Source {idx}:**\n\n{src[:400]} ...")
        else:
            st.warning("Please enter a question before asking.")

if __name__ == "__main__":
    main()
