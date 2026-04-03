# ============================================================
# 2_document_loaders.py
# Document Loaders + PDF Loaders
# ============================================================

import os
import pandas as pd
from config import DATA_PATH, TEXT_COLUMNS, JOB_COLUMNS


def print_header(title):
    print("\n" + "=" * 60)
    print("   " + title)
    print("=" * 60)


# ============================================================
# LOADER 1: CSV Loader (Manual)
# ============================================================

def load_csv_manual():
    """Load CSV using pandas (manual method)."""
    print_header("LOADER 1: CSV (Manual Pandas)")
    
    df = pd.read_csv("/Users/swethaharidas/Downloads/python/final_chatbot/data.csv")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)[:8]}...")
    
    # Create documents
    documents = []
    for idx, row in df.head(5).iterrows():
        text = str(row.get('Text', ''))[:200]
        documents.append({
            'content': text,
            'metadata': {
                'source': 'csv',
                'row': idx,
                'skills': str(row.get('Skills', ''))[:100]
            }
        })
    
    print(f"\n   Sample document:")
    print(f"   Content: {documents[0]['content'][:150]}...")
    print(f"   Metadata: {documents[0]['metadata']}")
    
    return df, documents


# ============================================================
# LOADER 2: CSV Loader (LangChain)
# ============================================================

def load_csv_langchain():
    """Load CSV using LangChain CSVLoader."""
    print_header("LOADER 2: CSV (LangChain)")
    
    try:
        from langchain_community.document_loaders import CSVLoader
        
        loader = CSVLoader(
            file_path=DATA_PATH,
            csv_args={'delimiter': ','},
            encoding='utf-8'
        )
        
        docs = loader.load()
        print(f"   Documents loaded: {len(docs)}")
        print(f"   Type: {type(docs[0])}")
        print(f"\n   Sample document:")
        print(f"   Content: {docs[0].page_content[:200]}...")
        print(f"   Metadata: {docs[0].metadata}")
        
        return docs
    except Exception as e:
        print(f"   Error: {e}")
        return []


# ============================================================
# LOADER 3: Text Loader (LangChain)
# ============================================================

def load_text_langchain():
    """Load plain text files."""
    print_header("LOADER 3: Text File (LangChain)")
    
    # Create a sample text file
    sample_text = """
    Python Developer Resume
    Skills: Python, Django, Flask, SQL, AWS
    Experience: 5 years backend development
    Education: Masters in Computer Science
    """
    
    with open("sample_resume.txt", "w") as f:
        f.write(sample_text)
    
    try:
        from langchain_community.document_loaders import TextLoader
        
        loader = TextLoader("sample_resume.txt")
        docs = loader.load()
        
        print(f"   Documents: {len(docs)}")
        print(f"   Content: {docs[0].page_content[:200]}")
        print(f"   Metadata: {docs[0].metadata}")
        
        os.remove("sample_resume.txt")
        return docs
    except Exception as e:
        print(f"   Error: {e}")
        os.remove("sample_resume.txt")
        return []


# ============================================================
# LOADER 4: PDF Loader
# ============================================================

def load_pdf():
    """Load PDF files."""
    print_header("LOADER 4: PDF Loader")
    
    # Check if sample PDF exists
    if not os.path.exists("sample.pdf"):
        print("   No sample.pdf found.")
        print("   Creating a demo with text instead...\n")
        
        # Simulate PDF content
        pdf_content = {
            'page_content': 'John Smith - Python Developer. Skills: Python, Django, SQL. Experience: 5 years at Google.',
            'metadata': {'source': 'sample.pdf', 'page': 0}
        }
        print(f"   Simulated PDF content: {pdf_content['page_content']}")
        print(f"   Metadata: {pdf_content['metadata']}")
        
        # Show how real PDF loading works
        print("\n   HOW TO LOAD REAL PDFs:")
        print("   " + "-" * 45)
        print("   pip install pypdf")
        print()
        print("   from langchain_community.document_loaders import PyPDFLoader")
        print("   loader = PyPDFLoader('resume.pdf')")
        print("   pages = loader.load_and_split()")
        print("   # Each page becomes a document")
        print()
        print("   OTHER PDF LOADERS:")
        print("   - PyPDFLoader      → Basic, page by page")
        print("   - PDFMinerLoader   → Better text extraction")
        print("   - UnstructuredPDFLoader → Handles complex layouts")
        print("   - PyMuPDFLoader    → Fast, preserves formatting")
        
        return []
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        
        loader = PyPDFLoader("sample.pdf")
        pages = loader.load_and_split()
        
        print(f"   Pages loaded: {len(pages)}")
        for i, page in enumerate(pages[:3]):
            print(f"\n   Page {i+1}:")
            print(f"   Content: {page.page_content[:150]}...")
            print(f"   Metadata: {page.metadata}")
        
        return pages
    except ImportError:
        print("   Install: pip install pypdf")
        return []
    except Exception as e:
        print(f"   Error: {e}")
        return []


# ============================================================
# LOADER 5: Directory Loader
# ============================================================

def load_directory():
    """Load all files from a directory."""
    print_header("LOADER 5: Directory Loader")
    
    print("   HOW TO LOAD ENTIRE DIRECTORIES:")
    print("   " + "-" * 45)
    print()
    print("   from langchain_community.document_loaders import DirectoryLoader")
    print()
    print("   # Load all text files")
    print("   loader = DirectoryLoader('docs/', glob='*.txt')")
    print()
    print("   # Load all PDFs")
    print("   loader = DirectoryLoader('docs/', glob='*.pdf',")
    print("       loader_cls=PyPDFLoader)")
    print()
    print("   # Load all CSVs")
    print("   loader = DirectoryLoader('data/', glob='*.csv',")
    print("       loader_cls=CSVLoader)")
    print()
    print("   docs = loader.load()")


# ============================================================
# COMPARISON
# ============================================================

def compare_loaders():
    print_header("DOCUMENT LOADERS COMPARISON")
    
    print(f"\n   {'Loader':<25} {'Best For':<25} {'Package':<20}")
    print("   " + "-" * 70)
    print(f"   {'CSVLoader':<25} {'CSV/Excel data':<25} {'langchain':<20}")
    print(f"   {'TextLoader':<25} {'Plain text files':<25} {'langchain':<20}")
    print(f"   {'PyPDFLoader':<25} {'PDF documents':<25} {'pypdf':<20}")
    print(f"   {'PDFMinerLoader':<25} {'Complex PDFs':<25} {'pdfminer':<20}")
    print(f"   {'UnstructuredLoader':<25} {'HTML/Word/etc':<25} {'unstructured':<20}")
    print(f"   {'DirectoryLoader':<25} {'Multiple files':<25} {'langchain':<20}")
    print(f"   {'WebBaseLoader':<25} {'Websites':<25} {'beautifulsoup4':<20}")
    print(f"   {'JSONLoader':<25} {'JSON data':<25} {'langchain':<20}")


def main():
    print_header("DOCUMENT & PDF LOADERS DEMO")
    
    load_csv_manual()
    load_csv_langchain()
    load_text_langchain()
    load_pdf()
    load_directory()
    compare_loaders()


if __name__ == "__main__":
    main()