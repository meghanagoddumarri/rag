import os
from typing import List
from typing_extensions import TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app_api = FastAPI(title="Corrective RAG Agent API")

# --- 1. CONFIGURATION & MODELS ---
# These will be pulled from your Docker environment or .env file
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation", 
    max_new_tokens=512,
    temperature=0.1,
)
llm = ChatHuggingFace(llm=llm_endpoint)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 2. VECTOR STORE SETUP ---
raw_docs = [
    Document(page_content="Mistral-7B-Instruct is an advanced open-source model optimized for chat."),
    Document(page_content="Agentic RAG uses a reasoning loop to determine document relevance."),
    Document(page_content="Hugging Face Embeddings run locally, bypassing API rate limits."),
    Document(page_content="Vector databases store meaning as mathematical vectors.")
]
vectorstore = SKLearnVectorStore.from_documents(documents=raw_docs, embedding=embeddings)
retriever = vectorstore.as_retriever(k=2)

# --- 3. LANGGRAPH STATE & NODES ---
class GraphState(TypedDict):
    question: str
    generation: str
    relevance: str
    documents: List[Document]

def grade_node(state):
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader. If the document is relevant to the question, reply ONLY with 'yes'. If not, reply ONLY with 'no'."),
        ("human", "Document: {document} \n Question: {question}")
    ])
    grader_chain = grader_prompt | llm | StrOutputParser()
    doc_txt = state["documents"][0].page_content
    response = grader_chain.invoke({"question": state["question"], "document": doc_txt})
    binary_score = "yes" if "yes" in response.lower() else "no"
    return {"relevance": binary_score}

def web_search_node(state):
    search = TavilySearchResults(k=3)
    results = search.invoke({"query": state["question"]})
    content = "\n".join([r["content"] for r in results])
    return {"documents": [Document(page_content=content)]}

def generate_node(state):
    prompt = ChatPromptTemplate.from_template("Answer using context: {documents}\nQuestion: {question}")
    gen_chain = prompt | llm | StrOutputParser()
    generation = gen_chain.invoke({"documents": state["documents"], "question": state["question"]})
    return {"generation": generation}

# --- 4. BUILD THE GRAPH ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", lambda state: {"documents": retriever.invoke(state["question"]), "question": state["question"]})
workflow.add_node("grade_documents", grade_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    lambda state: "generate" if state["relevance"] == "yes" else "web_search",
    {"generate": "generate", "web_search": "web_search"}
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
agent_app = workflow.compile()

# --- 5. API ENDPOINTS ---
class UserRequest(BaseModel):
    question: str

@app_api.post("/ask")
async def ask_ai(request: UserRequest):
    try:
        inputs = {"question": request.question}
        result = agent_app.invoke(inputs)
        return {
            "question": request.question,
            "answer": result["generation"],
            "source": "Local Docs" if result.get("relevance") == "yes" else "Web Search"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_api.get("/")
async def root():
    return {"status": "Agent is online"}