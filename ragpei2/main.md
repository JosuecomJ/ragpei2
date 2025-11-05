# main.md

## 🧠 Projeto RAGPEI2 - Chat RAG para Editais Acadêmicos

Este projeto implementa um sistema de perguntas e respostas sobre editais acadêmicos utilizando técnicas de RAG (Retrieval-Augmented Generation) com LangChain, LangGraph, ChromaDB e modelos Gemini.

---

### Funcionalidades

- **Processamento de PDF**: Carrega e divide o edital em partes menores (chunks) para facilitar a busca.
- **Vetorização**: Gera embeddings dos textos usando modelo local HuggingFace (`all-MiniLM-L6-v2`).
- **Banco de dados vetorial**: Armazena os embeddings no ChromaDB, permitindo buscas semânticas rápidas.
- **Chat inteligente**: Usuário faz perguntas e recebe respostas detalhadas baseadas no conteúdo do edital.
- **Avaliação de relevância**: O modelo Gemini avalia se os documentos recuperados são suficientes para responder à pergunta.
- **Fallback**: Se não houver contexto suficiente, o Gemini responde com conhecimento geral.

---

### Fluxo de Execução

1. **Carregamento do PDF**: O arquivo do edital é lido e dividido em chunks.
2. **Criação/Carregamento do banco vetorial**: Se o banco já existe, é carregado; caso contrário, é criado e persistido.
3. **Chat**: O usuário faz perguntas, que são processadas pelo grafo LangGraph:
   - Busca documentos relevantes no ChromaDB.
   - Avalia se os documentos são suficientes.
   - Gera resposta detalhada ou usa fallback.

---

### Principais Tecnologias

- **LangChain**: Framework para aplicações de IA com LLMs.
- **LangGraph**: Orquestração de fluxos de decisão com grafos.
- **ChromaDB**: Banco de dados vetorial para busca semântica.
- **Gemini (Google Generative AI)**: Geração de respostas e avaliação de relevância.
- **HuggingFace Embeddings**: Vetorização local e gratuita dos textos.

---

### Como Executar

1. Instale as dependências com Poetry:
   ```bash
   poetry install
   ```
2. Configure o arquivo `.env` com sua chave `GOOGLE_API_KEY`.
3. Execute o script principal:
   ```bash
   poetry run python ragpei2/main.py
   ```
4. Interaja com o chat no terminal.

---



---

### Estrutura do Projeto

```
ragpei2/
├── main.py
├── chat.py
├── docs/
│   └── edital_mestrado_ppgi_2025_2.pdf
├── chroma_db/
├── .env
├── .gitignore
└── README.md
```

---

### Licença

Este projeto é distribuído sob a licença MIT.
