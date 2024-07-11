import openai
import streamlit as st
import requests
import os
from io import BytesIO
from PIL import Image
import textwrap
import tempfile
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up your OpenAI API key
openai_api_key = os.environ['OPENAI_API_KEY']


def summarize_document(document_path):
    
    # Load document
    loader = PyMuPDFLoader(document_path)
    documents = loader.load()
    print(f"Loaded document: {document_path}")
    
    # Split the document into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Split document into {len(texts)} chunks.")

    # Combine the document chunks into a single context
    context = " ".join([text.page_content for text in texts])
    
    # Define a prompt template for summarization
    prompt = f"Summarize the following document: {context}"
    
    # Use OpenAI's GPT model for summarization
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    summary = response.choices[0].message.content.strip()
    print(f"Generated summary: {summary}")
    
    return summary

def generate_dalle_prompt(summary):
    # Use OpenAI's GPT-3 to generate a DALL-E prompt based on the summary
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Create a detailed prompt which is less than 1000 characters for DALL-E to generate an image based on this summary: {summary}."}
        ]
    )
    
    prompt = response.choices[0].message.content
    if len(prompt) > 1000:
        prompt = textwrap.shorten(prompt, width=1000, placeholder="...")
        
    return prompt

def generate_dalle_image(prompt):
    # Use OpenAI's DALL-E to generate an image based on the prompt
    response = openai.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    
    # Get the URL of the generated image
    image_url = response.data[0].url
    
    # Download the image
    image_response = requests.get(image_url)
    image = Image.open(BytesIO(image_response.content))
    
    return image

def main():
    st.title("Document Summarization and DALL-E Image Generation")

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_pdf_path = temp_file.name

        st.write("Summarizing the document...")
        summary = summarize_document(temp_pdf_path)
        st.write("Summary of the document:")
        st.write(summary)
        
        st.write("Generating DALL-E prompt...")
        dalle_prompt = generate_dalle_prompt(summary)
        st.write("DALL-E prompt:")
        st.write(dalle_prompt)
        
        st.write("Generating image with DALL-E...")
        image = generate_dalle_image(dalle_prompt)
        
        st.image(image, caption='Generated Image', use_column_width=True)

        # Clean up the temporary file
        os.remove(temp_pdf_path)
        print(f"Removed temporary file: {temp_pdf_path}")

if __name__ == "__main__":
    main()
