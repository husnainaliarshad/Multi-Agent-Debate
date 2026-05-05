import pandas as pd
import pymupdf
import sys
import os

def read_pdf(file_path):
    try:
        doc = pymupdf.open(file_path)
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    except Exception as e:
        return f"Error reading PDF {file_path}: {e}"

def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df.to_string()
    except Exception as e:
        return f"Error reading Excel {file_path}: {e}"

docs = [
    "Fall26_GenAI_Final_Project.pdf",
    "Project Rubrics.xlsx",
    "Springer_Template_FAST.pdf"
]

for doc in docs:
    print(f"--- CONTENT OF {doc} ---")
    if doc.endswith('.pdf'):
        print(read_pdf(doc))
    elif doc.endswith('.xlsx'):
        print(read_excel(doc))
    print("\n" + "="*50 + "\n")
