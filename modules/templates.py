from jinja2 import Template

# Template for user/assistant conversation
for_llama3_history = """
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if message['role'] == 'user' %}
        USER: {{ message['content'] }}
    {% elif message['role'] == 'assistant' %}
        MACHINE: {{ message['content'] }}
    {% else %}
        {{ raise_exception('Only user and assistant roles are supported!') }}
    {% endif %}
{% endfor %}
"""

for_llama3_route_query = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You help triage user request, \
    do NOT answer the question itself, do NOT explain yourself. \
    Given a text input, output either DOCS or DEFAULT, according to these definitions:

    DOCS: if user is asking an ambiguous question that seems to require knowledge from some external documentation
    DEFAULT: if user is just chit-chatting

    Output ONE word answer: DOCS or DEFAULT, WITHOUT explanation. <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Ok, what is your question <|eot_id|><|start_header_id|>user<|end_header_id|>
    {query} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

for_llama3_sort_relevancy = """<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>\
    You help classify the relevancy of the context info to the question, \
    do NOT answer the question, do NOT explain yourself. \
    Given some documents, and a question from human, output either YES or NO, by the following definitions:

    YES: if the documents CAN help you derive an answer for the question
    NO: if the documents is NOT relevant to the question

    Output ONE word answer: YES, or NO, WITHOUT explanation. <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Ok, give me the documents <|eot_id|><|start_header_id|>user<|end_header_id|>
    DOCUMENTS: {context} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Ok, what is your question <|eot_id|><|start_header_id|>user<|end_header_id|>
    {query} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

for_llama3_qa_from_docs = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You help answer humans's questions, only based on the given documents. \
    Given some documents, and a human question, output your answer. Keep it short. <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Ok, give me the documents <|eot_id|><|start_header_id|>user<|end_header_id|>
    DOCUMENTS: {context} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Ok, what is your question <|eot_id|><|start_header_id|>user<|end_header_id|>
    {query} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

for_llama3_contextualize_in_history = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Given a chat history and the most recent question, you help formulate the question into \
    a standalone question that can be understood without the chat history. \
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is. \
    Output only that formulated question. <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Ok, tell me the chat history. Use USER:, and MACHINE: to show whose turn it is in the history <|eot_id|><|start_header_id|>user<|end_header_id|>
    CHAT HISTORY: {chat_history} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Ok, what is your most recent question that needs formulated? <|eot_id|><|start_header_id|>user<|end_header_id|>
    {query} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

for_llama3_clarify = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You help gather more information. \
    Given some documents, and a human question, which is not directly relevant to the documents, \
    do NOT answer the question, but ask a clarification question, for users to elaborate on question question. \
    Tell the user that the question needs elaboration by ouputing one single clarification question. \
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Ok, give me the documents <|eot_id|><|start_header_id|>user<|end_header_id|>
    DOCUMENTS: {context} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Ok, what is your question <|eot_id|><|start_header_id|>user<|end_header_id|>
    {query} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


class CustomTemplates:
    """Handles Jinja2 template rendering for user/assistant history."""
    def __init__(self, model: str) -> None:
        models = ['llama3']
        if model not in models:
            raise RuntimeError('No Models Found')
        self.active = models.index(model)
        if self.active == 0:
            self.history = Template(for_llama3_history)

    def render_template(self, content):
        if self.active == 0:
            return self.history.render({"messages": content})
        return ""  # Extend for more models if needed


class Prompts:
    """Holds all prompt strings (Jinja not needed for these)"""
    def __init__(self, model: str) -> None:
        models = ['llama3']
        if model not in models:
            raise RuntimeError('No Models Found')
        self.active = models.index(model)
        if self.active == 0:
            self.route_query = for_llama3_route_query
            self.sort_relevancy = for_llama3_sort_relevancy
            self.qa_from_docs = for_llama3_qa_from_docs
            self.contextualize_in_history = for_llama3_contextualize_in_history
            self.clarify = for_llama3_clarify
