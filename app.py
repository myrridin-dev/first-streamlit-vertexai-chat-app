# started from https://github.com/bijukunjummen/genai-chat/blob/main/app.py
# then changed OpenAI to VertexAI
# then merged it with https://python.langchain.com/docs/integrations/memory/streamlit_chat_message_history/
# then merged it with https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py

from typing import List

import os
import streamlit as st
import vertexai

from langchain import globals

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import StreamlitChatMessageHistory, ChatMessageHistory
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings

PROJECT_ID='first-vertexai-streamlit-app'
REGION_ID='us-central1'

TEXT_EMBEDDING_MODEL = 'textembedding-gecko'
LLM_MODEL = "gemini-1.0-pro"

INDEX_PATH = './index/'
DB_PATH = '/tmp/'

ROLE_AI="ai"
ROLE_HUMAN="human"
ROLE_SYSTEM="system"

CHAT_HISTORY_KEY="chat_history"

DEBUG = True

# chat histories can be stored by session_id
def get_session_id():
    # TODO change to something that is unique to the calling user
    return "any"

def get_chat_history_by_session_id(session_id) -> BaseChatMessageHistory:
    # TODO RedisChatMessageHistory to persist/retrieve history by session_id, conversation_id
    # Eg. session_manager = RedisChatMessageHistory(str(session_id), redis_url, key_prefix="chat_history", ttl=3600)
    # return session_manager.get_session_history
    return msgs

# Clear Chat
def clear_chat_history():
    msgs.messages.clear()

st.set_page_config(page_title="Medicare in 2024", page_icon="🦈")
st.title("📖 Medicare in 2024")

def get_split_documents(index_path: str) -> List[str]:
    chunk_size=1024
    chunk_overlap=128

    split_docs = []

    for file_name in os.listdir(index_path):
        if file_name.endswith(".pdf"):
            loader = UnstructuredPDFLoader(index_path + file_name)
        else:
            loader = TextLoader(index_path + file_name)

        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs.extend(text_splitter.split_documents(loader.load()))

    return split_docs


def create_vector_db():
    embeddings = VertexAIEmbeddings(
        model_name=TEXT_EMBEDDING_MODEL, batch_size=5
    )
    # Load documents, generate vectors and store in Vector database
    split_docs = get_split_documents(INDEX_PATH)
    faissdb = FAISS.from_documents(split_docs, embeddings)
    faissdb_name = DB_PATH + '/faiss.db'
    faissdb.save_local(faissdb_name)

    if DEBUG:
        faissdb_size = os.path.getsize(faissdb_name)
        print(f"Size of FAISS db on disk: {faissdb_size} bytes")

    return faissdb

@st.cache_resource
def get_llm() -> VertexAI:
    return VertexAI(
        model='gemini-1.0-pro',
        max_output_tokens=8192,
        temperature=0.2,
        top_p=0.8,
        top_k=1,
        verbose=DEBUG,
    )

@st.cache_resource
def get_embeddings() -> VertexAIEmbeddings:
    return VertexAIEmbeddings(
        model_name='textembedding-gecko',
        batch_size=5
    )

# main
globals.set_debug(DEBUG)

vertexai.init(project=PROJECT_ID, location=REGION_ID)

llm = VertexAI(
    model=LLM_MODEL,
    max_output_tokens=8192,
    temperature=0.2,
    top_p=0.8,
    top_k=1,
    verbose=DEBUG,
)

faissdb = create_vector_db()
retriever = faissdb.as_retriever()

### Sub chain for contextualizing the question
# Goal: takes historical messages and the latest user question, and reformulates the question if it makes reference to any information in the historical information.
contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history, \
    formulate a standalone question which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (ROLE_SYSTEM, contextualize_q_system_prompt),
        MessagesPlaceholder(CHAT_HISTORY_KEY),
        (ROLE_HUMAN, "{input}"),
    ]
)

# See https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/history_aware_retriever.py#L57
# If there is no chat_history in invoke(), chain is input | retriever | llm
# If there is chat_history in invoke(), chain is:
#   input + chat_history | llm using contextualize_q_system_prompt to get updated_input
#   updated_input | retriever | llm
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Main chain for chat with history
qa_system_prompt = """
    You are a helpful AI assistant and an expert at Medicare laws in the US. You're tasked to answer the question given below, \
    but only based on the context provided. If you cannot find an answer ask the user to rephrase the question. Use three sentences \
    maximum and keep the answer concise.

    Examples:
    Q: Will I get Part A and Part B automatically?
    A: If you’re already getting benefits from Social Security or the Railroad Retirement Board (RRB), you’ll automatically get \
    Part A and Part B starting the first day of the month you turn 65. 

    Q: Will I have to sign up for Part A and/or Part B?
    A: If you’re close to 65, but NOT getting Social Security or RRB benefits, you’ll need to sign up for Medicare.

    Q: Do I have to pay for Part A?
    A: You usually don’t pay a monthly premium for Part A coverage if you or your spouse paid Medicare taxes while working \
    for a certain amount of time. 

    {context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (ROLE_SYSTEM, qa_system_prompt),
        MessagesPlaceholder(CHAT_HISTORY_KEY),
        (ROLE_HUMAN, "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages") # The key to use in Streamlit session state for storing messages.
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_chat_history_by_session_id, # This is a function that returns history given a session_id
    input_messages_key="input",
    output_messages_key="answer",   # This is what the response key in the vertexai response is
    history_messages_key=CHAT_HISTORY_KEY, # This is specifying the key where Langchain automatically stores new messages as history?
)

view_messages = st.expander("View the message contents in session state")

# Populate sidebar
with st.sidebar:
    st.title('Chat settings')
    streaming_on = st.toggle('Streaming')

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if question := st.chat_input():
    session_id = get_session_id()
    chat_history = get_chat_history_by_session_id(session_id)

    st.chat_message(ROLE_HUMAN).write(question)

    # we specify the corresponding chat history via a configuration parameter tied to session_id
    config = {"configurable": {"session_id": session_id}}

    # Chain - Stream vs. Full
    if streaming_on:
        placeholder = st.empty()
        full_response = ''
        for chunk in rag_chain_with_history.stream({"input": question}, config):
            # before request is made chunk will only be {'input': 'question from human'}
            # so need to check if 'answer' has come back in response
            if "answer" in chunk.keys():
                if DEBUG:
                    print(f"[PARTIAL] ai response: {chunk['answer']}")
                full_response += chunk["answer"]
                placeholder.chat_message(ROLE_AI).write(full_response)
            placeholder.chat_message(ROLE_AI).write(full_response)
    else:
        response = rag_chain_with_history.invoke({"input": question}, config)
        if DEBUG:
            print(f"[FULL] ai response: {response}")
        st.chat_message(ROLE_AI).write(response["answer"])

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)


# Populate sidebar
with st.sidebar:
    st.button('Clear history', on_click=clear_chat_history)
    st.divider()
    st.write("History")
    st.write([m.content for m in msgs.messages if type(m) == HumanMessage])