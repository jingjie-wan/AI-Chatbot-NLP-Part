from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
import psutil
import os
from pymongo import MongoClient
from datetime import datetime


# # RAG Systems

# ## Initializing the RAGPipeline Class

# In[10]:


class RAGPipeline:
    def __init__(self, model_name: str = "deepseek-r1:8b", max_memory_gb: float = 3.0):
        self.setup_logging()
        self.check_system_memory(max_memory_gb)
        self.llm = OllamaLLM(model=model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.conversation_history = []  # Store conversation history

        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        You are an HR assistant. Answer the question based only on the following employee records. Be concise.
        If you cannot find the answer in the context, say \"I cannot answer this based on the provided context.\"

        Today's Date: {current_date}

        Context: {context}
        Question: {question}
        Answer: """)
        
# Memory Management and Logging
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_system_memory(self, max_memory_gb: float):
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < max_memory_gb:
            self.logger.warning("Memory is below recommended threshold.")
            
# Loading and Splitting Documents
    def load_and_split_documents(self) -> List[Document]:
            # Connect to MongoDB
            client = MongoClient("mongodb+srv://iriswan0202:720716@cluster0.qwm1e.mongodb.net/")
            db = client['HRWIKI']

            # Get all collections in the database
            #collections = db.list_collection_names()
            collections = ['EmploymentAgreement', 'Medical plan summary - Price Details 2025', '1000 PLAN SBC - ITLIZE GLOBAL', '2500 PLAN SBC - ITLIZE GLOBAL', 'Delta Dental Benefit Summary','Itlize Global LLC - DELTA Buy-Up Plan - PPO Plus Premier - Non Par MAC Benefit Summary', 'Delta Vision Benefit Summary', 'Enrollment Application form UHC Global', 'WorkAuthDetails', 'Employee', 'EmployeeTerminationDocument']
            # Retrieve all documents from all collections
            with open("temp_data.txt", "w", encoding="utf-8") as f:
                for collection_name in collections:
                    collection = db[collection_name]
                    data = list(collection.find())

                    for entry in data:
                        formatted_entry = '\n'.join(f"{key}: {value}" for key, value in entry.items())
                        f.write(f"Collection: {collection_name}\n{formatted_entry}\n\n")

            # Load and split text
            loader = TextLoader("temp_data.txt", encoding="utf-8")
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                add_start_index=True,
            )
            splits = text_splitter.split_documents(documents)
            self.logger.info(f"Created {len(splits)} document chunks from {len(collections)} collections.")
            return splits
    
# Creating a Vector Store with FAISS
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        batch_size = 32
        vectorstore = FAISS.from_documents(documents[:batch_size], self.embeddings)

        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectorstore.add_documents(batch)
            self.logger.info(f"Processed batch {i//batch_size + 1} with {len(batch)} documents")

        self.logger.info(f"Total documents in vectorstore: {len(vectorstore.index_to_docstore_id)}")
        return vectorstore

    def setup_rag_chain(self, vectorstore: FAISS):
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10, "fetch_k": 20})
    
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
    
        def add_history_and_question(question):
            history_text = "\n".join([f"User: {q}\nAI: {a}" for q, a in self.conversation_history])
            combined_question = f"{history_text}\n\nCurrent Question: {question}" if history_text else question
    
            return {
                "context": format_docs(retriever.get_relevant_documents(question)),
                "current_date": datetime.now().strftime('%Y-%m-%d'),
                "question": combined_question
            }
    
        rag_chain = (
            add_history_and_question
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

# Querying the Model with Memory Monitoring
    def query(self, chain, question: str) -> str:
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.logger.info(f"Memory usage: {memory_usage:.1f} MB")
        response = chain.invoke(question)

        # Record conversation history
        self.conversation_history.append((question, response))

        return response


# In[14]:


rag_pipeline = RAGPipeline(model_name="deepseek-r1:8b", max_memory_gb=3.0)
documents = rag_pipeline.load_and_split_documents()
vectorstore = rag_pipeline.create_vectorstore(documents)
chain = rag_pipeline.setup_rag_chain(vectorstore)


# In[21]:


from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict
import logging
import json
from fastapi.middleware.cors import CORSMiddleware
import re

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


class QueryRequest(BaseModel):
    userId: str
    message: str


@app.post("/ask", response_model=Dict[str, str])
async def ask_ai(request: QueryRequest):
    try:

        request_json = request.dict()
        print(f"\n🔹 Received Request:\n{json.dumps(request_json, indent=2)}")

        user_id = request.userId
        question = request.message + "Only output the answer."

        if not question:
            raise HTTPException(status_code=400, detail="❌ Error: Missing 'message' field in request.")

        def clean_response(response):
            # Only output response after 'Answer:'
            response_answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response  # Extract answer or return full response if not found
            # Only output response between or after 'think'
            response_think = response_answer.split("</think>")[-1].strip()
            cleaned_response = re.sub(r"<think>", "", response_think).strip()
            return cleaned_response
        
        response = rag_pipeline.query(chain, question)
        cleaned_response = clean_response(response)

        print(f"\n🔹 AI Response (not cleaned):\n{response}")

        return {"userId": user_id, "response": cleaned_response}

    except Exception as e:
        logging.error(f"❌ Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.options("/ask")
async def options_handler(request: Request):
    return {}





