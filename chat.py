"""Chat simplificado reutilizando o banco vetorial persistido."""

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


def carregar_componentes():
	"""Inicializa LLM, embeddings e carrega o vetor store persistido."""
	load_dotenv()
	api_key = os.getenv("GOOGLE_API_KEY")
	if not api_key:
		raise ValueError("GOOGLE_API_KEY não encontrada no arquivo .env")

	llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
	embeddings = HuggingFaceEmbeddings(
		model_name="sentence-transformers/all-MiniLM-L6-v2",
		model_kwargs={"device": "cpu"},
	)

	vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
	retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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

	fallback_chain = (
		ChatPromptTemplate.from_template("Responda com base no seu conhecimento geral: {input}")
		| llm
		| StrOutputParser()
	)

	return retriever, rag_chain, fallback_chain


def executar_chat():
	retriever, rag_chain, fallback_chain = carregar_componentes()

	print("=" * 50)
	print("💬 CHAT RAG - Digite 'sair' para encerrar")
	print("=" * 50)

	while True:
		pergunta = input("\n🤔 Você: ").strip()
		if not pergunta:
			continue
		if pergunta.lower() in {"sair", "exit", "quit"}:
			print("\n👋 Encerrando o chat. Até logo!")
			break

		print("\n🔍 Processando...")
		documentos = retriever.invoke(pergunta)

		if documentos:
			resposta = rag_chain.invoke({"input": pergunta, "context": documentos})
		else:
			resposta = fallback_chain.invoke({"input": pergunta})

		print(f"\n🤖 Assistente: {resposta}")
		print("-" * 50)


if __name__ == "__main__":
	executar_chat()
