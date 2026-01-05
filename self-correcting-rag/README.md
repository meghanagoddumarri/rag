# Self-Correcting Agentic RAG with Web Fallback

An autonomous AI Agent built with **LangGraph** and **Llama-3-8B** that intelligently routes queries between a local knowledge base and the live web.

## 🧠 The Architecture
Unlike standard RAG, this system implements a **Self-Correction loop**:
1. **Retrieve**: Pulls context from a local Vector Store.
2. **Grade**: An LLM "Grader" evaluates the relevance of the retrieved data.
3. **Decision**: 
   - If **Relevant**: Generates an answer using local data.
   - If **Irrelevant**: Triggers a **Tavily Web Search** to find real-time info.



##  Tech Stack
- **Orchestration:** LangGraph
- **LLM:** Meta-Llama-3-8B-Instruct (via Hugging Face)
- **Vector DB:** SKLearnVectorStore
- **Embeddings:** Sentence-Transformers (Local)
- **Web Tools:** Tavily Search API
- **Deployment:** Docker & FastAPI

##  Quick Start

### 1. Setup Environment
Create a `.env` file and add your keys:
```text
HF_TOKEN=your_huggingface_token
TAVILY_API_KEY=your_tavily_key
GOOGLE_API_KEY=your_google_key