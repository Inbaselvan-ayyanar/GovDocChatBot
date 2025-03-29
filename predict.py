import chromadb
import fitz  
from flask import jsonify
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import CrossEncoder
import json

CHROMA_DB_DIR = "chroma_db"  
cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  

def retrieve_relevant_chunks(query, k=5):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text") 
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)
    results = vector_store.similarity_search(query, k=k)  
    return results  

def rerank_chunks(query, chunks):
    if not chunks:
        return ""

    query_doc_pairs = [(query, doc.page_content) for doc in chunks]
    scores = cross_encoder_model.predict(query_doc_pairs)
    ranked_chunks = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    return ranked_chunks[0][1].page_content  

def ask_ollama(query):
    retrieved_chunks = retrieve_relevant_chunks(query)
    best_chunk = rerank_chunks(query, retrieved_chunks)

    response = ollama.chat(model="mistral", messages=[
        {"role": "system", "content": "You are an AI assistant answering questions based on provided context."},
        {"role": "user", "content": f"Context: {best_chunk} \n\nAnswer the question: {query}"}
    ])
    return response['message']['content']

def main(query):
    answer = ask_ollama(query)
    print("\n AI Answer:\n", answer)
    return answer

def run(question):
    return main(question)

