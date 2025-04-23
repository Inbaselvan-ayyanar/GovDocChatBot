GovDocChatBot

📄 AI-Powered PDF Chatbot for Government Document Access4

Overview

GovDocChatBot is a smart chatbot solution designed to process government documents (PDFs) and enable users to ask natural language questions and get accurate answers based only on the content of those documents. It enhances public access, data transparency, and efficiency in retrieving government information.

It uses OCR, vector embeddings, LLMs, and retrieval-augmented generation (RAG) techniques to ensure fast and context-aware responses.

🔍 Features

✅ PDF text and image (OCR) extraction

✅ Recursive character splitting for chunk management

✅ Vector embedding storage using ChromaDB

✅ AI-powered Q&A via LLM (Mistral through Ollama)

✅ Context-aware search with cross-encoder re-ranking

✅ Seamless retrieval from large datasets of government documents


📦 Prerequisites

Python 3.10 or later

Ollama installed with Mistral and embedding model

Required Python libraries:

chromadb, pymupdf, ollama, langchain, sentence_transformers, flask

⚙️ Installation

Clone the repository  git clone https://github.com/Inbaselvan-ayyanar/GovDocChatBot.git

cd GovDocChatBot

Install dependencies  pip install -r requirements.txt

Run Ollama and download models

ollama pull mistral  

ollama pull nomic-embed-text  


🚀 Usage

Prepare PDF documents

Place your PDF files in the input directory or specify their path.

Train the system (Optional if already trained)

python Train.py

Run the chatbot: python flask4.py

Ask questions through the web interface or API

The chatbot will return responses based on the document content.

⚙️ Configuration

Edit constants in the scripts:

CHROMA_DB_DIR: Location for vector database storage.

model: Ollama model (default is "mistral")

embedding_function: Uses "nomic-embed-text" by default.

🔧 Troubleshooting

❗ Ensure that Ollama is running and both models are pulled.

❗ Check that documents are clean and readable for proper OCR.

❗ If answers seem off, re-train with fresh document ingestion using Train.py.

📬 Contact

For queries or support, contact: a.inbaselvan@gmail.com
