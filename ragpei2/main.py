# main.py
import os
import json
from dotenv import load_dotenv
from typing import List, TypedDict

# --- Modelos (LLM e Embeddings) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# --- Loader (PyPDF) ---
from langchain_community.document_loaders import PyPDFLoader

# --- Text Splitter ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Vector Store (ChromaDB) ---
from langchain_community.vectorstores import Chroma

# --- RAG ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- A ESTRELA: LangGraph ---

from langgraph.graph import StateGraph, END

# --- Configuração ---
print("Carregando configurações...")
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY não encontrada no arquivo .env")

ARQUIVO_PDF = "./ragpei2/docs/edital_mestrado_ppgi_2025_2.pdf"

# --- 1. Inicializar Modelos Gemini ---
print("Inicializando modelos...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# Usar modelo de embedding local e gratuito
print("Carregando modelo de embeddings local...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# --- 2. LOAD, SPLIT e STORE (Executado apenas uma vez) ---
# Em um app real, isso seria feito offline. Aqui, fazemos no início.
print(f"Processando PDF...")
loader = PyPDFLoader(ARQUIVO_PDF)
documents = loader.load()

# Dividir o texto em chunks menores
print("Dividindo texto em chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
splits = text_splitter.split_documents(documents)


# Diretório para persistência do banco de dados
CHROMA_DB_DIR = "./chroma_db"

if os.path.exists(CHROMA_DB_DIR):
    print("Carregando banco de dados ChromaDB persistido...")
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
else:
    print("Criando embeddings e armazenando no ChromaDB...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    vectorstore.persist()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Vector Store pronto.")


# --- 3. Definir o Estado do Grafo (LangGraph) ---
# O "Estado" é o objeto que flui entre os nós.
class GraphState(TypedDict):
    pergunta: str      # A pergunta original do usuário
    documentos: List  # Os documentos encontrados no Chroma (objetos Document)
    resposta: str      # A resposta final do LLM

# --- 4. Definir os "Nós" do Grafo ---

# NÓ 1: BUSCAR DOCUMENTOS
def retrieve_docs(state: GraphState) -> GraphState:
    print("--- NÓ: RETRIEVE_DOCS ---")
    pergunta = state['pergunta']
    # O retriever do ChromaDB (definido acima)
    documentos_obj = retriever.invoke(pergunta)
    # Mantém os objetos Document para o RAG chain
    print(f"Documentos encontrados: {len(documentos_obj)}")
    return {"documentos": documentos_obj, "pergunta": pergunta}

# NÓ 2: GERAR RESPOSTA (RAG)
def generate_answer(state: GraphState) -> GraphState:
    print("--- NÓ: GENERATE_ANSWER (RAG) ---")
    pergunta = state['pergunta']
    documentos = state['documentos']
    
    # Cria a chain "Stuff" (igual ao código anterior)
    prompt_template = """
    Você é um assistente especializado em responder perguntas sobre documentos acadêmicos e editais.
    Use o contexto fornecido abaixo para responder à pergunta de forma detalhada e completa.
    Seja específico, cite informações relevantes do contexto e forneça explicações claras.
    
    Contexto: {context}
    Pergunta: {input}
    
    Resposta detalhada:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    rag_chain = create_stuff_documents_chain(llm, prompt)
    
    # Invoca a chain com os documentos
    resposta = rag_chain.invoke({"input": pergunta, "context": documentos})
    
    return {"resposta": resposta}

# NÓ 3: FALLBACK (Se os documentos não forem bons)
def fallback(state: GraphState) -> GraphState:
    print("--- NÓ: FALLBACK (Sem RAG) ---")
    pergunta = state['pergunta']
    
    # Responde sem contexto, apenas com o conhecimento geral do LLM
    prompt = ChatPromptTemplate.from_template(
        "Responda a pergunta a seguir: {input}"
    )
    chain = prompt | llm | StrOutputParser()
    resposta = chain.invoke({"input": pergunta})
    
    return {"resposta": resposta}

# NÓ 4 (CONDICIONAL): AVALIAR RELEVÂNCIA
# Este nó usa o LLM para decidir se os documentos são úteis
def grade_documents(state: GraphState) -> GraphState:
    print("--- NÓ: GRADE_DOCUMENTS (Avaliação) ---")
    pergunta = state['pergunta']
    documentos = state['documentos']

    # Prompt para o LLM "avaliador"
    prompt = ChatPromptTemplate.from_template(
        """
        Analise os documentos e a pergunta.
        A pergunta é: {pergunta}
        Os documentos recuperados são:
        <documentos>
        {documentos}
        </documentos>
        
        A informação nos documentos é suficiente para responder à pergunta?
        Responda APENAS com "sim" ou "nao".
        """
    )
    
    # Usamos o Gemini 1.5 Flash (rápido) para a avaliação
    eval_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    eval_chain = prompt | eval_llm | StrOutputParser()
    
    resultado = eval_chain.invoke({
        "pergunta": pergunta,
        "documentos": "\n---\n".join([doc.page_content for doc in documentos])
    })
    
    print(f"Resultado da avaliação: {resultado}")
    # Retorna o "caminho" (edge) que o grafo deve seguir
    if "sim" in resultado.lower():
        return "relevante"
    else:
        return "irrelevante"

# --- 5. Construir o Grafo (LangGraph) ---
print("Construindo o grafo...")

workflow = StateGraph(GraphState)

# Adiciona os nós ao grafo
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("generate_rag", generate_answer)
workflow.add_node("fallback", fallback)

# Define o ponto de entrada
workflow.set_entry_point("retrieve")

# Adiciona as arestas (edges) condicionais
workflow.add_conditional_edges(
    # O nó de partida é o "retrieve"
    "retrieve",
    # A função que decide o próximo passo é "grade_documents"
    grade_documents,
    # O mapeamento das saídas da função para os próximos nós
    {
        "relevante": "generate_rag", # Se for "relevante", vai para o RAG
        "irrelevante": "fallback"    # Se for "irrelevante", vai para o Fallback
    }
)

# Adiciona as arestas finais
# Se o grafo chegar em "generate_rag" ou "fallback", ele termina.
workflow.add_edge("generate_rag", END)
workflow.add_edge("fallback", END)

# Compila o grafo
app = workflow.compile()
print("Grafo compilado. Pronto.")
print("-" * 50)


# --- 6. Executar o Grafo em modo Chat ---
print("=" * 50)
print("💬 CHAT RAG - Digite 'sair' para encerrar")
print("=" * 50)

while True:
    # Solicita a pergunta do usuário
    pergunta_usuario = input("\n🤔 Você: ").strip()
    
    # Verifica se o usuário quer sair
    if pergunta_usuario.lower() in ['sair', 'exit', 'quit']:
        print("\n👋 Encerrando o chat. Até logo!")
        break
    
    # Ignora entradas vazias
    if not pergunta_usuario:
        continue
    
    # O input deve ser um dicionário com as chaves do estado inicial
    inputs = {"pergunta": pergunta_usuario}
    
    print("\n🔍 Processando...")
    
    # O 'stream' mostra o fluxo passo a passo
    for output in app.stream(inputs):
        # 'output' é um dicionário onde a chave é o nome do nó
        # e o valor é o estado (GraphState) após esse nó.
        node_name = list(output.keys())[0]
        node_state = output[node_name]
    
    # A resposta final estará no último estado
    resposta_final = node_state.get('resposta')
    print(f"\n🤖 Assistente: {resposta_final}")
    print("-" * 50)