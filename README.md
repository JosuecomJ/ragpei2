# RAGPEI2 - Sistema de Chat RAG para Editais Acadêmicos

Um sistema inteligente de chat baseado em Retrieval-Augmented Generation (RAG) que permite fazer perguntas sobre editais acadêmicos e receber respostas contextualizadas baseadas no conteúdo dos documentos.

## 🚀 Funcionalidades

- **Processamento de PDFs**: Carrega e processa documentos PDF de editais acadêmicos
- **Vetorização Local**: Usa embeddings gratuitos e locais (HuggingFace) para evitar limites de API
- **Banco de Dados Persistente**: Armazena embeddings no ChromaDB para reutilização
- **Chat Interativo**: Interface de chat no terminal para perguntas e respostas
- **Avaliação Inteligente**: Sistema que decide se usar contexto do documento ou conhecimento geral
- **LangGraph**: Fluxo de trabalho estruturado com nós condicionais para processamento inteligente

## 🛠️ Tecnologias Utilizadas

- **Python 3.13**
- **LangChain**: Framework para aplicações LLM
- **LangGraph**: Orquestração de fluxos de trabalho
- **ChromaDB**: Banco de dados vetorial
- **Google Gemini**: Modelos de linguagem para geração de respostas
- **HuggingFace Transformers**: Embeddings locais gratuitos
- **Poetry**: Gerenciamento de dependências

## 📋 Pré-requisitos

- Python 3.13+
- Poetry (gerenciador de dependências)
- Chave da API do Google Gemini (GOOGLE_API_KEY)

## 🔧 Instalação

1. **Clone o repositório**:

```bash
git clone https://github.com/seu-usuario/ragpei2.git
cd ragpei2
```

2. **Instale as dependências**:

```bash
poetry install
```

3. **Configure o ambiente**:

   - Crie um arquivo `.env` na raiz do projeto
   - Adicione sua chave da API do Google Gemini:

   ```
   GOOGLE_API_KEY=sua_chave_aqui
   ```

4. **Prepare o documento**:
   - Coloque o PDF do edital em `ragpei2/docs/edital_mestrado_ppgi_2025_2.pdf`
   - Ou ajuste o caminho no código em `main.py`

## 📦 Exportando requirements.txt (opcional)

Se precisar de um arquivo `requirements.txt` para deploy em ambientes que não usam Poetry, gere com:

```bash
poetry export -f requirements.txt --output requirements.txt
```

Isso criará um arquivo `requirements.txt` com todas as dependências do projeto.
com isso sera sera possivel instalar as dependencias usando ` pip install -r requirements.txt`

## 🚀 Como Executar

1. **Ative o ambiente virtual**:

```bash
poetry shell
```

2. **Execute o chat**:

```bash
poetry run python ragpei2/main.py
```

3. **Interaja com o chat**:
   - Digite suas perguntas sobre o edital
   - Digite "sair" para encerrar

## 💡 Como Funciona

O sistema segue este fluxo inteligente:

1. **Carregamento**: Processa o PDF e divide em chunks
2. **Vetorização**: Cria embeddings usando modelo local (sentence-transformers)
3. **Armazenamento**: Salva no ChromaDB (persistente)
4. **Chat Loop**:
   - Recebe pergunta do usuário
   - Busca documentos relevantes no banco vetorial
   - Avalia se os documentos são suficientes para responder
   - Gera resposta usando contexto (RAG) ou conhecimento geral (fallback)

## 📁 Estrutura do Projeto

```
ragpei2/
├── pyproject.toml          # Configuração do Poetry
├── README.md              # Esta documentação
├── ragpei2/
│   ├── __init__.py
│   ├── main.py            # Código principal do chat RAG
│   └── docs/              # Diretório para documentos
│       └── edital_mestrado_ppgi_2025_2.pdf
├── tests/
│   └── __init__.py
└── chroma_db/             # Banco vetorial (criado automaticamente)
```

## 🔍 Exemplo de Uso

```
💬 CHAT RAG - Digite 'sair' para encerrar
==================================================

🤔 Você: Qual é o período de inscrição?

🔍 Processando...
--- NÓ: RETRIEVE_DOCS ---
Documentos encontrados: 3
--- NÓ: GRADE_DOCUMENTS (Avaliação) ---
Resultado da avaliação: sim
--- NÓ: GENERATE_ANSWER (RAG) ---

🤖 Assistente: As inscrições para o Mestrado em Informática do PPGI 2025/2 estarão abertas
no período de 01 de dezembro de 2024 a 31 de janeiro de 2025. Os candidatos devem enviar
toda a documentação exigida para o email ppgi@ufma.br até as 23h59 do dia 31 de janeiro de 2025.
--------------------------------------------------
```

## ⚙️ Configuração Avançada

### Alterar Modelo de Embedding

Para usar um modelo diferente de embedding, modifique em `main.py`:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="outro-modelo-huggingface",
    model_kwargs={'device': 'cpu'}
)
```

### Ajustar Parâmetros de Busca

Modifique o número de documentos recuperados:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Busca top-5
```

### Personalizar Chunk Size

Ajuste o tamanho dos chunks de texto:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # Tamanho do chunk
    chunk_overlap=300,    # Sobreposição entre chunks
    length_function=len,
)
```

## 💬 Interface Web com Streamlit

Também é possível interagir com o sistema via uma interface de chat moderna na web usando Streamlit.

1. **Execute o frontend Streamlit**:

```bash
poetry run streamlit run app.py
```

2. **Acesse no navegador**:
   - O Streamlit exibirá um link (geralmente http://localhost:8501)
   - Faça perguntas diretamente na interface web, com histórico de chat e visual moderno

> **Obs:** O backend não será reprocessado a cada execução do Streamlit, apenas consultas ao banco vetorial já criado.

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 📞 Suporte

Para dúvidas ou sugestões, abra uma issue no GitHub ou entre em contato com os mantenedores do projeto.

---

**Desenvolvido com ❤️ para facilitar o acesso a informações acadêmicas**
