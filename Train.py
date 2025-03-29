import fitz  # PyMuPDF for PDF extraction
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import cv2
import numpy as np
from paddleocr import PaddleOCR
CHROMA_DB_DIR = "chroma_db" 
ocr = PaddleOCR()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    paragraphs = []
    current_paragraph = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        if text:
            lines = text.split("\n")
            for line in lines:
                line = line.strip()
                if line:  
                    current_paragraph += line + " "  # Append lines together
                else:
                    if current_paragraph:  # If we hit an empty line, store the paragraph
                        paragraphs.append(current_paragraph.strip())
                        current_paragraph = ""

            if current_paragraph:  # Store last paragraph if it exists
                paragraphs.append(current_paragraph.strip())  

        # Extract images and perform OCR
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Convert image to grayscale for better OCR results
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Perform OCR
            result = ocr.ocr(gray, cls=True)

            if not result:  # Handle None or empty results
                continue  

            image_text = []
            for line in result:
                if line is None:  # Check for None before iterating
                    continue
                try:
                    para = "".join([word[1][0] for word in line if word])  # Handle unexpected formats
                    image_text.append(para)
                except Exception as e:
                    print(f"Error processing OCR line: {e}")

            if image_text:
                paragraphs.extend(image_text)  # Store OCR text as paragraphs

    return paragraphs

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],)
    return text_splitter.split_text(text)

def store_in_chromadb(text_chunks):
    if not text_chunks:
        return
    
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")  # Ollama embedding model
    vector_store = Chroma.from_texts(text_chunks, embedding_model, persist_directory=CHROMA_DB_DIR)
    vector_store.persist()
    print("✅ Data stored in ChromaDB")


def main(pdf_path):
    print("🔹 Extracting and storing document...")
    pdf_text = extract_text_from_pdf(pdf_path)
    #print(pdf_text)
    
    for text in pdf_text:
        text_chunks = split_text(text)
        store_in_chromadb(text_chunks)


def run(filename):
    
    pdf_path = f"C:/Users/ainba/Downloads/ccp2/uploads/{filename}"  
    main(pdf_path)
    return 1