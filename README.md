# End-to-End Medical Chatbot using BioMistral LLM

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain and BioMistral-7B for answering medical and heart health-related queries.

## 🎯 Project Overview

This project implements a medical AI assistant specializing in heart health using:
- **BioMistral-7B**: An open-source medical language model
- **RAG Architecture**: Retrieval-Augmented Generation for accurate responses
- **Vector Database**: ChromaDB for efficient similarity search
- **Medical Embeddings**: PubMedBERT for domain-specific embeddings

## 🛠️ Technologies Used

- **LangChain**: Framework for LLM applications
- **Llama.cpp**: Efficient LLM inference
- **ChromaDB**: Vector database
- **SentenceTransformers**: Embedding generation
- **PyMuPDF (fitz)**: PDF processing
- **BioMistral-7B**: Medical domain LLM

## 📋 Prerequisites

```bash
Python 3.8+
Google Colab (recommended) or local GPU
Hugging Face account (for model access)
```

## 🚀 Installation

```bash
# Install required packages
pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf PyMuPDF
```

## 📂 Project Structure

```
├── healthyheart.pdf          # Medical document (heart health)
├── EndToEndMedicalChatBot.ipynb  # Main notebook
└── README.md                 # Project documentation
```

## 🔧 Setup & Configuration

### 1. Import Dependencies

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import fitz  # PyMuPDF
```

### 2. Set Hugging Face Token

```python
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "your_token_here"
```

### 3. Load and Process PDF

```python
doc = fitz.open("/content/healthyheart.pdf")

# Convert to LangChain Documents
from langchain.schema import Document
documents = []
for i, page in enumerate(doc):
    documents.append(Document(
        page_content=page.get_text(), 
        metadata={"page": i + 1}
    ))
```

### 4. Text Chunking

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
```

### 5. Create Embeddings

```python
embeddings = SentenceTransformerEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)
```

### 6. Build Vector Store

```python
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

### 7. Load BioMistral LLM

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="itlwas/BioMistral-7B-Q4_K_M-GGUF",
    filename="biomistral-7b-q4_k_m.gguf",
    temperature=0.2,
    max_tokens=2048,
    top_p=1
)
```

### 8. Create RAG Chain

```python
template = """
<|context|>
You are a medical AI assistant specializing in heart health.
Use the provided context to give accurate and helpful general advice
about health, diet, and lifestyle.
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)

reg_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | RunnableLambda(lambda x: x.to_string())
    | llm
    | RunnableLambda(lambda output: output["choices"][0]["text"])
    | StrOutputParser()
)
```

## 💡 Usage

```python
# Ask a question
query = "Who is at risk of heart disease?"
response = reg_chain.invoke(query)
print(response)
```

## 🔍 Example Queries

- "Who is at risk of heart disease?"
- "What foods are good for heart health?"
- "How does exercise affect heart health?"
- "What are the symptoms of heart disease?"

## ⚙️ Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| chunk_size | 300 | Size of text chunks |
| chunk_overlap | 50 | Overlap between chunks |
| temperature | 0.2 | LLM creativity (lower = more factual) |
| max_tokens | 2048 | Maximum response length |
| k | 5 | Number of retrieved documents |

## 🐛 Known Issues & Solutions

See the "Code Issues and Fixes" section below for common problems and their solutions.

## 📊 Performance

- **Response Time**: ~5-10 seconds per query (GPU)
- **Accuracy**: High for heart health queries
- **Context Window**: Up to 2048 tokens

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with:
- BioMistral model license
- Medical information usage guidelines

## 🙏 Acknowledgments

- BioMistral team for the medical LLM
- LangChain community
- PubMedBERT developers

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**⚠️ Disclaimer**: This chatbot is for educational purposes only. Always consult healthcare professionals for medical advice.
