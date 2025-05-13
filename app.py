import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import chainlit as cl

with open("combined_data.json", "r") as f:
    raw_data = json.load(f)

all_docs = [Document(page_content=entry["content"], metadata=entry["metadata"]) for entry in raw_data]

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
chunked_docs = splitter.split_documents(all_docs)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunked_docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@cl.on_message
async def main(message):
    response = rag_chain.run(message.content)
    await cl.Message(content=response).send()
