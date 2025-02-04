import openai
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import streamlit as st
from datetime import datetime
from transformers import pipeline
import shutil

# Set OpenAI API Key
openai.api_key = "your_openai_api_key_here"

# Setup Huggingface pipeline for summarization
summarizer = pipeline("summarization")

# Function to read and extract text from PDF
def extract_pdf_text(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Function to summarize the research paper
def summarize_paper(text):
    summary = summarizer(text, max_length=500, min_length=100, do_sample=False)
    return summary[0]['summary_text']

# Function to analyze research paper using GPT-4
def analyze_research_paper(text):
    prompt = f"""
    I have provided a research paper below. Please read and understand the content, then summarize the key findings and research directions mentioned in the paper. Use the insights to find relevant research papers in the AI domain. Here's the paper content:
    
    {text}
    
    Please provide a summary and some related research areas.
    """
    
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=1500,
        n=1,
        temperature=0.7,
    )
    
    return response.choices[0].text.strip()

# Function to search for related papers using Google Scholar (web scraping)
def search_related_papers(query):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    search_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}"
    
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    papers = []
    for item in soup.find_all("h3", {"class": "gs_rt"}):
        title = item.get_text()
        link = item.a["href"] if item.a else "No link available"
        papers.append({"title": title, "link": link})
    
    return papers

# Function to assist with code generation and optimization
def generate_code_from_research(topic):
    prompt = f"""
    I need help writing code related to the following topic from a research paper: {topic}
    Please generate a Python code snippet that can help implement the concept discussed in the paper.
    """
    
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        temperature=0.7,
    )
    
    return response.choices[0].text.strip()

# Function to generate and explain code
def explain_and_optimize_code(code):
    prompt = f"""
    Please explain the following code and suggest any improvements for optimization, efficiency, or clarity:
    
    {code}
    """
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Function for handling and visualizing data (e.g., CSVs)
def plot_data(file_path):
    df = pd.read_csv(file_path)
    df.plot(kind='bar')
    plt.show()

# Function to handle task deadlines and reminders
def set_deadline(task, deadline):
    current_time = datetime.now()
    if current_time < deadline:
        st.write(f"Task '{task}' is due on {deadline}.")
    else:
        st.write(f"Task '{task}' is overdue!")

# Streamlit web interface
def research_assistant():
    st.title("AI Research Assistant")
    st.write("Upload your research paper (PDF format) and let the assistant help you analyze, summarize, and find related research.")

    # Upload PDF research paper
    uploaded_file = st.file_uploader("Upload Research Paper", type=["pdf"])

    if uploaded_file is not None:
        pdf_text = extract_pdf_text(uploaded_file)
        if st.button("Summarize Paper"):
            summary = summarize_paper(pdf_text)
            st.subheader("Summary of the Paper:")
            st.write(summary)

        if st.button("Analyze Paper with GPT-4"):
            analysis = analyze_research_paper(pdf_text)
            st.subheader("GPT-4 Analysis:")
            st.write(analysis)

        # Fetch Related Papers
        related_papers = search_related_papers(analysis)
        st.subheader("Related Research Papers:")
        for i, paper in enumerate(related_papers, start=1):
            st.write(f"{i}. {paper['title']} - {paper['link']}")

        # Code Assistance
        coding_topic = st.text_input("Enter topic for code assistance:")
        if coding_topic:
            generated_code = generate_code_from_research(coding_topic)
            st.subheader("Generated Code:")
            st.code(generated_code)

            if st.button("Explain and Optimize Code"):
                explanation = explain_and_optimize_code(generated_code)
                st.subheader("Code Explanation and Optimization:")
                st.write(explanation)

        # Data Handling (CSV)
        data_file = st.file_uploader("Upload Dataset (CSV format)", type=["csv"])
        if data_file:
            st.write("Visualizing the data...")
            plot_data(data_file)

        # Task Management
        task_name = st.text_input("Enter task name:")
        deadline_date = st.date_input("Set deadline date:")
        if task_name and deadline_date:
            deadline = datetime(deadline_date.year, deadline_date.month, deadline_date.day)
            set_deadline(task_name, deadline)

# Run Streamlit app
if __name__ == "__main__":
    research_assistant()
