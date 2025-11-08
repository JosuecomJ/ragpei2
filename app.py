
import streamlit as st
from chat import carregar_componentes, expandir_consultas

st.set_page_config(page_title="Chat RAG Edital", layout="centered")
st.title("💬 Chat RAG - Edital Acadêmico")

# Inicializa histórico na sessão
if "history" not in st.session_state:
    st.session_state.history = []

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
        "resposta": "",
        "consultas": []
    })
    
    # Exibe mensagem do usuário
    with st.chat_message("user"):
        st.write(pergunta)
    
    # Processa resposta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            retriever, query_expander, rag_chain, fallback_chain = carregar_componentes()
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
    
    # Atualiza histórico com resposta
    st.session_state.history[-1]["resposta"] = resposta
    st.session_state.history[-1]["consultas"] = consultas
    
    # Rerun para atualizar a tela
    st.rerun()