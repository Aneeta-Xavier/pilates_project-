import os
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import chainlit as cl

# Load and prepare data from local JSON
with open("combined_data.json", "r") as f:
    raw_data = json.load(f)

all_docs = [
    Document(page_content=entry["content"], metadata=entry["metadata"])
    for entry in raw_data
]

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
chunked_docs = splitter.split_documents(all_docs)

# Set up FAISS vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunked_docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Set up LLM and QA chain
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# On chat start
@cl.on_chat_start
async def start():
    await cl.Message("Ready! Ask me anything about Reformer Pilates.").send()
    cl.user_session.set("qa_chain", qa_chain)

# On message
@cl.on_message
async def handle_message(message: cl.Message):
    chain = cl.user_session.get("qa_chain")
    if chain:
        response = chain.run(message.content)
        await cl.Message(response).send()
