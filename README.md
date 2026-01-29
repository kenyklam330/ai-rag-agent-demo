🤖 Currys AI RAG Agent
Developed by Ken Lam | 🔗 LinkedIn | 📧 Email
📖 Project Overview
This is a high-performance Retrieval-Augmented Generation (RAG) application built to assist retail specialists in navigating complex product catalogs. It allows users to upload technical PDFs and receive accurate, context-aware answers.

✨ Key Features
Hybrid Provider Selection: Toggle seamlessly between Ollama (Gemma 3) for 100% local data privacy and OpenAI (GPT-4o-mini) for cloud-based reasoning.

Dual-Model Strategy: Utilizes nomic-embed-text for optimized document retrieval and Gemma 3:4b for human-like response generation.

Transparent Attribution: Every answer includes a "Source Documents" expander, showing the exact PDF snippets and page numbers used to verify the information.

Modern 2026 Tech Stack: Built using the latest LangChain v1.x modular partner packages (langchain-ollama, langchain-openai).

🛠️ Technical Stack
Frontend: Streamlit

Orchestration: LangChain (Core & Community)

Vector Store: FAISS (Facebook AI Similarity Search)

Embeddings: Nomic Embed Text / OpenAI Text Embedding 3

LLMs: Google Gemma 3:4b / OpenAI GPT-4o-mini

🚀 Installation & Setup
1. Prerequisites
Python 3.10+

Ollama (for local execution)

Required models:

Bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Environment Variables
If using OpenAI, set your API key in your environment or via the UI:

Bash
export OPENAI_API_KEY="your_api_key_here"
4. Run the Application
Bash
streamlit run app.py
📂 Project Structure
Plaintext
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
└── temp/               # (Auto-created) Temporary PDF storage
🤝 Contact & Collaboration
I am an aspiring AI Engineer passionate about retail automation. Feel free to reach out for collaboration or professional inquiries!

Ken Lam

LinkedIn: ken-yiu-kei-lam

Email: kenyklam330@gmail.com
