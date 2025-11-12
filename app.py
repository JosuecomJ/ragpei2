import streamlit as st
from chat import carregar_componentes, expandir_consultas

# --- Configuração inicial da página ---
st.set_page_config(page_title="Chat RAG Edital", layout="centered")

# --- Título principal da aplicação (Passo 2.2) ---
st.title("💬 Chat RAG - Edital Acadêmico")

# --- PASSO 1.1: Função para carregar componentes RAG com cache ---
@st.cache_resource
def get_rag_components():
    """
    Carrega e armazena em cache os componentes RAG (retriever, query_expander,
    rag_chain, fallback_chain). Esta função será executada apenas uma vez na
    inicialização da aplicação Streamlit.
    """
    return carregar_componentes()

# --- PASSO 1.1: Chama a função cacheada uma vez no início do script ---
retriever, query_expander, rag_chain, fallback_chain = get_rag_components()

# --- PASSO 2.1: CRIAÇÃO DA SIDEBAR E MOVENDO O BOTÃO DE LIMPAR CONVERSA ---
st.sidebar.title("⚙️ Configurações") # Título da sidebar
st.sidebar.markdown("---") # Linha divisória

# Botão "Limpar Conversa" movido para a sidebar (Passo 1.2 movido para 2.1)
if st.sidebar.button("🗑️ Limpar Conversa", help="Reinicia o histórico da conversa."):
    st.session_state.history = [] # Limpa o histórico
    # Adiciona uma mensagem inicial para não começar completamente vazio
    st.session_state.history.append({
        "pergunta": "...", # A pergunta do usuário não é relevante para a primeira mensagem
        "resposta": "Olá! Sou seu assistente para o edital acadêmico. Como posso ajudar hoje?",
        "consultas": []
    })
    st.rerun() # Força o Streamlit a reexecutar o script para mostrar o chat limpo

st.sidebar.markdown("---") # Outra linha divisória
# --- FIM DO PASSO 2.1 ---


# Inicializa histórico na sessão
if "history" not in st.session_state:
    st.session_state.history = []
    # Adiciona a mensagem inicial aqui também, caso a sessão seja nova e o botão de limpar não tenha sido clicado
    st.session_state.history.append({
        "pergunta": "...",
        "resposta": "Olá! Sou seu assistente para o edital acadêmico. Como posso ajudar hoje?",
        "consultas": []
    })


# Exibe histórico de mensagens
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["pergunta"])
    with st.chat_message("assistant"):
        st.write(item["resposta"])
        with st.expander("Consultas geradas"):
            st.write(" | ".join(item["consultas"]))

# Input do usuário
pergunta = st.chat_input("Digite sua pergunta sobre o edital:")

if pergunta:
    # Adiciona mensagem do usuário ao histórico
    st.session_state.history.append({
        "pergunta": pergunta,
        "resposta": "", # A resposta será preenchida abaixo
        "consultas": []  # As consultas serão preenchidas abaixo
    })

    # Exibe mensagem do usuário imediatamente
    with st.chat_message("user"):
        st.write(pergunta)

    # Processa resposta
    with st.chat_message("assistant"):
        # --- PASSO 1.3: MELHORIA DA MENSAGEM DO SPINNER ---
        with st.spinner("Buscando informações e gerando resposta..."):
            # Os componentes RAG (retriever, query_expander, rag_chain, fallback_chain)
            # já foram carregados e cacheados no início do script, então os usamos diretamente.

            consultas = expandir_consultas(query_expander, pergunta)

            documentos_coletados = []
            for consulta in consultas:
                documentos_coletados.extend(retriever.invoke(consulta))

            # Remove duplicados
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

        st.write(resposta)
        with st.expander("Consultas geradas"):
            st.write(" | ".join(consultas))

    # Atualiza histórico com a resposta e as consultas geradas
    st.session_state.history[-1]["resposta"] = resposta
    st.session_state.history[-1]["consultas"] = consultas

    st.rerun()