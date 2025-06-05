# Introduction
This is the NLP part of a HR AI chatbot. 

It utilizes the open-source **DeepSeek model (deepseek-r1:8b)** and employs a **Retrieval-Augmented Generation (RAG)** approach, with **MongoDB** serving as the knowledge base.
# How to run:
You can test the .py file locally by run:
```
uvicorn rag_ai_agent:app --host 0.0.0.0 --port 8082 --reload
```
on command line. Then open http://127.0.0.1:8082/docs on your brower.

# Prerequisite
Download and Install [Ollama](https://ollama.com/)

Install DeepSeek (right now we use the 8b version due to restriction of our local computers)
```
ollama pull deepseek-r1:8b
```

Check if you have installed DeepSeek successfully on Ollama
```
ollama list
```

Install necessary Python libraries
```
pip install pandas faiss-cpu scikit-learn ollama
pip install pymongo langchain faiss-cpu sentence-transformers psutil
pip install langchain langchain-community langchain-core langchain-huggingface langchain-ollama faiss-cpu sentence-transformers pymongo psutil
pip install fastapi uvicorn
```
# Data
Dummy HR data (including employee basic information, visa status, medical insurance, etc.) in MongoDB database. 24 Collections in JSON format.

# Input and Output
## Input format
```
{
    "userId": "0079",
    "message": "How many vacation days do I have left?"
}
```
### Output format
```
{
    "userId": "0079",
    "response":  "You have 14 days of vacation left"
}
```
