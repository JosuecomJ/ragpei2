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
	"""
 	Inicializa e retorna os principais componentes para um sistema RAG (Retrieval-Augmented Generation):

	- Carrega a chave de API do Google Gemini a partir do arquivo .env.
	- Inicializa o modelo de linguagem (LLM) ChatGoogleGenerativeAI.
	- Carrega o vetor store persistido via Chroma e cria um retriever para busca de contexto.
	- Define o template de prompt para respostas detalhadas baseadas em contexto.
	- Cria a cadeia RAG para geração de respostas com base em documentos.
	- Cria uma cadeia de fallback para respostas baseadas em conhecimento geral.
	- Cria um expansor de perguntas para gerar reformulações da consulta do usuário.

	Returns:
		tuple: (retriever, query_expander, rag_chain, fallback_chain)
	"""
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
	retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

	query_expander = (
		ChatPromptTemplate.from_template(
			"""
			Gere até três reformulações concisas e relevantes para a seguinte pergunta:
			"{pergunta}"
			Liste cada reformulação em uma linha separada, sem enumeração e sem explicações adicionais.
			"""
		)
		| llm
		| StrOutputParser()
	)

	return retriever, query_expander, rag_chain, fallback_chain


def expandir_consultas(query_expander, pergunta: str) -> list[str]:
	"""
	Expande uma consulta original gerando variações usando um modelo de linguagem (LLM) e remove duplicatas.

	Args:
		query_expander: Um objeto capaz de gerar variações da consulta via método `invoke`.
		pergunta (str): Consulta original a ser expandida.

	Returns:
		list[str]: Lista de consultas únicas, incluindo a original e suas variações geradas.
	"""
	"""Gera variações da consulta original usando o LLM e remove duplicatas."""
	variacoes_texto = query_expander.invoke({"pergunta": pergunta})
	variacoes = [pergunta]
	for linha in variacoes_texto.splitlines():
		consulta = linha.strip().lstrip("-•0123456789. ").strip()
		if consulta:
			variacoes.append(consulta)

	consultas_unicas: list[str] = []
	for consulta in variacoes:
		if consulta not in consultas_unicas:
			consultas_unicas.append(consulta)
	return consultas_unicas


def executar_chat():
	"""
	Inicia um chat interativo utilizando um sistema RAG (Retrieval-Augmented Generation).

	O usuário pode digitar perguntas, que são processadas por um pipeline de expansão de consultas,
	recuperação de documentos e geração de respostas. Caso nenhum documento relevante seja encontrado,
	uma cadeia de fallback é utilizada para gerar a resposta.

	O chat permanece ativo até que o usuário digite 'sair', 'exit' ou 'quit'.

	Fluxo principal:
		1. Carrega os componentes necessários (retriever, query_expander, rag_chain, fallback_chain).
		2. Exibe instruções iniciais do chat.
		3. Recebe perguntas do usuário em loop.
		4. Expande a consulta e recupera documentos relevantes.
		5. Remove documentos duplicados.
		6. Gera resposta baseada nos documentos ou utiliza fallback.
		7. Exibe a resposta ao usuário.

	Não recebe parâmetros e não retorna valores.
	"""
	retriever, query_expander, rag_chain, fallback_chain = carregar_componentes()

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
		consultas = expandir_consultas(query_expander, pergunta)
		documentos_coletados = []
		for consulta in consultas:
			documentos_coletados.extend(retriever.invoke(consulta))

		visto = set()
		documentos = []
		for doc in documentos_coletados:
			chave = doc.page_content
			if chave not in visto:
				visto.add(chave)
				documentos.append(doc)

		if documentos:
			resposta = rag_chain.invoke({"input": pergunta, "context": documentos})
		else:
			resposta = fallback_chain.invoke({"input": pergunta})

		print(f"\n🤖 Assistente: {resposta}")
		print("-" * 50)


if __name__ == "__main__":
	executar_chat()
