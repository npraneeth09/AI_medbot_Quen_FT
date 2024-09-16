from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from src.prompt import *
from src.helper import download_hugging_face_embedding
import os
import json
import gradio as gr


local_llm = r"C:\Users\irath\Documents\docs\Projects\End_to_End_Medical_Chatbot_using_Langchain\End-to-End-Medical-Chatbot-using-Langchain\Model\qwen2-1.5b-medical_qa-finetune-q4_k_m.gguf"
llm = LlamaCpp(model_path = local_llm, temperature = 0.1, max_tokens=2048, top_p=1, n_ctx=2048)

embeddings = download_hugging_face_embedding()

url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

retriever = db.as_retriever(search_kwargs={"k":1})

chat_history = []
if llm is not None and db is not None:
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
else:
    print("LLM or Vector Database not initialized")
    
def predict(message, history):
    history_langchain_format = []
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "message"])
    
    response = chain({"question": message, "chat_history": chat_history})
    answer = response['answer']
    
    chat_history.append((message, answer))
    
    temp = []
    for input_question, bot_answer in history:
        temp.append(input_question)
        temp.append(bot_answer)
        history_langchain_format.append(temp)
    temp.clear()
    temp.append(message)
    temp.append(answer)
    history_langchain_format.append(temp)
    
    return answer

gr.ChatInterface(predict).launch()