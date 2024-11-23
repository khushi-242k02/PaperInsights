from flask import Flask, request, render_template, send_file
import fitz  # PyMuPDF
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch
import tempfile
import time
import PyPDF2
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



# Initialize Flask app
app = Flask(__name__)

# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Use smaller model for speed
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)  # Using smaller model



# Function to summarize text into a paragraph
def summarize_text_paragraph(text, max_length=300, min_length=100):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=2, early_stopping=True)  # Reduced beams
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Improved chunking function to ensure efficient tokenization
def chunk_text(text, max_tokens=512):
    sentences = text.split('. ')  # Split the text into sentences
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding the sentence exceeds max token limit
        if len(tokenizer.encode(current_chunk + sentence)) <= max_tokens:
            current_chunk += sentence + ". "
        else:
            if current_chunk:  # Avoid adding empty chunks
                chunks.append(current_chunk)
            current_chunk = sentence + ". "  # Start a new chunk

    if current_chunk:  # Add the last chunk if it isn't empty
        chunks.append(current_chunk)
    
    return chunks

# Function to extract text by page from the PDF
def extract_text_by_page(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    pages = [page.extract_text() for page in reader.pages]
    return pages

# Batch summarization function
def summarize_text_batch(texts, max_length=300, min_length=100):
    inputs = tokenizer.batch_encode_plus(
        ["summarize: " + text for text in texts],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=1024
    ).to(device)
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=2,  # Reduced beams
        early_stopping=True
    )
    
    return [tokenizer.decode(s, skip_special_tokens=True) for s in summary_ids]


# # Initialize the T5 tokenizer/model for summarization
# print("Loading T5 model...")
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Initialize the NER pipeline
print("Loading NER model...")
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

# Function to extract text by page using PyMuPDF
def extract_text_by_page_pymupdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    pages = [pdf_document[page_num].get_text() for page_num in range(pdf_document.page_count)]
    pdf_document.close()
    return pages

# Function to perform NER on text
 # Function to perform NER on text
def ner_on_text(text):
    entities = ner(text)
    unique_entities = {}  # Dictionary to store unique entities and the highest score

    for entity in entities:
        # Create a tuple of (word, entity_group) as the key
        entity_key = (entity['word'], entity['entity_group'])

        # If the entity is not in the dictionary or if we find a higher score, update it
        if entity_key not in unique_entities or unique_entities[entity_key] < entity['score']:
            unique_entities[entity_key] = entity['score']

    # Format the entities into the desired output format
    formatted_entities = [
        f"{word} ({entity_group}): {score:.2f}"
        for (word, entity_group), score in unique_entities.items()
    ]
    
    return formatted_entities



# # Function to split text into manageable chunks
# def chunk_text(text, max_length=512):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), max_length):
#         chunks.append(' '.join(words[i:i + max_length]))
#     return chunks

# # Function to summarize text in batches
# def summarize_text_batch(chunks):
#     summaries = []
#     for chunk in chunks:
#         inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=1024, truncation=True).to(device)
#         summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
#         summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
#     return summaries

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

# @app.route('/highlights')
# def highlights():
#     return render_template('highlights.html')

@app.route('/extract', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        pdf_file = request.files['pdf']
        if pdf_file:
            # Save the uploaded PDF file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                pdf_file.save(temp_file.name)
                pages = extract_text_by_page_pymupdf(temp_file.name)
                all_entities = []
                for page_num, page_text in enumerate(pages):
                    if page_text.strip():
                        page_entities = ner_on_text(page_text)
                        all_entities.append((f"Page {page_num + 1}", page_entities))
                    else:
                        all_entities.append((f"Page {page_num + 1}", ["No text available"]))
            return render_template('extract.html', entities=all_entities)
    return render_template('extract.html')

# @app.route('/summarize', methods=['GET', 'POST'])
# def summarize():
#     if request.method == 'POST':
#         pdf_file = request.files['pdf']
#         if pdf_file:
#             start_time = time.time()
#             # Save the uploaded PDF file to a temporary location
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#                 pdf_file.save(temp_file.name)
#                 pages = extract_text_by_page_pymupdf(temp_file.name)
#                 summaries = []
#                 for page in pages:
#                     if page:
#                         chunks = chunk_text(page)
#                         page_summary = summarize_text_batch(chunks)
#                         summaries.append(' '.join(page_summary))
#             end_time = time.time()
#             print(f"Processing Time: {end_time - start_time:.2f} seconds")
#             return render_template('summarize.html', summaries=summaries)
#     return render_template('summarize.html')

@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        pdf_file = request.files['pdf']
        if pdf_file:
            # Extract text from each page of the uploaded PDF
            start_time = time.time()  # Track time for performance testing
            pages = extract_text_by_page(pdf_file)
            summaries = []

            # Process each page
            for page in pages:
                if page:
                    # First, chunk the page if the content is too large
                    chunks = chunk_text(page)
                    # Summarize multiple chunks at once
                    page_summary = summarize_text_batch(chunks)
                    
                    summaries.append(' '.join(page_summary))

            end_time = time.time()
            print(f"Processing Time: {end_time - start_time} seconds")

            return render_template('summarize.html', summaries=summaries)
    return render_template('summarize.html')



# Word Cloud 
# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Function to extract text by page using PyMuPDF
def extract_text_by_page(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    pages = [page.extract_text() for page in reader.pages]
    return pages

# Function to generate word cloud
def generate_word_cloud(text):
    # Tokenize and clean the text (remove stopwords)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    cleaned_text = ' '.join(filtered_words)

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(cleaned_text)
    
    # Convert word cloud to image and then to base64 string for embedding in HTML
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    img_b64 = base64.b64encode(img.read()).decode('utf-8')
    return img_b64


def extract_keywords(text, num_keywords=15):
    # Initialize TfidfVectorizer to extract keywords
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([text])  # Fit and transform the text
    
    # Get the feature names (words) and their corresponding tf-idf scores
    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = tfidf_matrix.toarray().flatten()  # Get the scores as a flat array
    
    # Sort the words by their tf-idf score in descending order
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    
    # Get the top N keywords
    top_keywords = feature_names[sorted_indices][:num_keywords]
    return top_keywords


@app.route('/highlights', methods=['GET', 'POST'])
def highlights():
    wordcloud_img = None  # To store the base64 word cloud image
    if request.method == 'POST':
        pdf_file = request.files['pdf']
        if pdf_file:
            pages = extract_text_by_page(pdf_file)
            text = ' '.join(pages)  # Combine text from all pages
            wordcloud_img = generate_word_cloud(text)
    return render_template('highlights.html', wordcloud_img=wordcloud_img)




if __name__ == '__main__':
    app.run(debug=True)
