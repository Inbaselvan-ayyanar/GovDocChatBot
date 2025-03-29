import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import predict
import Train
from docx2pdf import convert  # For DOCX to PDF conversion
from PIL import Image  # For image to PDF conversion
import pythoncom
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flash messages

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """ Check if file has an allowed extension """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_pdf(file_path):
    """ Convert DOCX and image files to PDF """
    file_ext = file_path.rsplit('.', 1)[1].lower()
    pdf_path = f"{file_path.rsplit('.', 1)[0]}.pdf"

    if file_ext == "docx":
        try:
            pythoncom.CoInitialize()  # Initialize COM in this thread
            convert(file_path, pdf_path)
        finally:
            pythoncom.CoUninitialize()  # Ensure cleanup
    elif file_ext in {"png", "jpg", "jpeg"}:
        img = Image.open(file_path)
        img.convert("RGB").save(pdf_path)
    else:
        return file_path  # Already a PDF

    os.remove(file_path)  # Remove original file after conversion
    return pdf_path

@app.route('/')
def home():
    return render_template('home.html')

# Train Route
@app.route('/train')
def train_page():
    return render_template('Train_index.html')

# File Upload Route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('train_page'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('train_page'))
    
    if not allowed_file(file.filename):
        flash('Invalid file format. Allowed: PDF, DOCX, PNG, JPG', 'error')
        return redirect(url_for('train_page'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    name=filename.split(".")
    # Convert to PDF if necessary
    pdf_path = convert_to_pdf(file_path)
    
    if Train.run(name[0]+".pdf"):
        print(filename)
        flash(f'File {os.path.basename(pdf_path)} uploaded successfully!', 'success')
    
    return redirect(url_for('train_page'))

# Predict Route (Chatbot)
@app.route('/predict')
def predict_page():
    return render_template('predict_index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "No message received"}), 400

    bot_response = predict.run(user_message)
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=False)  # Set to True only during development

