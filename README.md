# Desafio MBA Engenharia de Software com IA - Full Cycle

> RAG pipeline that ingests PDF documents into a vector database (PostgreSQL + pgVector) and enables semantic search via CLI chat, powered by LangChain and Google Gemini.

## Ingestão e Busca Semântica com LangChain e Postgres

Pipeline RAG (Retrieval-Augmented Generation) que realiza ingestão de documentos PDF em banco vetorial (PostgreSQL + pgVector) e permite buscas semânticas via chat CLI usando LangChain e Google Gemini.

O sistema carrega um PDF, divide em chunks, gera embeddings com Google Gemini e armazena no PostgreSQL com pgVector. Na busca, recupera os trechos mais relevantes e usa uma LLM para gerar respostas fundamentadas exclusivamente no conteúdo do documento.

### Arquitetura

```
PDF ──▶ PyPDFLoader ──▶ TextSplitter ──▶ Embeddings (Gemini) ──▶ PostgreSQL + pgVector
                                                                          │
Pergunta ──▶ Embedding ──▶ Similarity Search ──▶ Top 10 chunks ──▶ LLM ──▶ Resposta
```

## Pré-requisitos

- Python 3.12+
- Docker e Docker Compose
- API Key do Google (Google AI Studio)

## Configuração

1. Clone o repositório e acesse a pasta do projeto:

```bash
git clone https://github.com/Leticiasg/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

2. Crie o arquivo `.env` a partir do template:

```bash
cp .env.example .env
```

3. Preencha o `.env` com suas credenciais:

```
GOOGLE_API_KEY='sua-chave-aqui'
GOOGLE_EMBEDDING_MODEL='models/gemini-embedding-001'
GOOGLE_LLM_MODEL='gemini-2.5-flash-lite'
DATABASE_URL='postgresql+psycopg://postgres:postgres@localhost:5432/rag'
PG_VECTOR_COLLECTION_NAME='rag_documents'
PDF_PATH='document.pdf'
INGEST_BATCH_SIZE=10
INGEST_BATCH_DELAY=60
```

A ingestão envia os chunks para a API de embeddings em lotes. `INGEST_BATCH_SIZE` define quantos chunks são enviados por lote e `INGEST_BATCH_DELAY` define o intervalo em segundos entre cada lote, para não exceder o rate limit da API do Google.

4. Crie e ative o ambiente virtual:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

5. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução

1. Suba o banco de dados:

```bash
docker compose up -d
```

2. Execute a ingestão do PDF:

```bash
python src/ingest.py
```

3. Rode o chat:

```bash
python src/chat.py
```

## Tecnologias

- Python 3.12
- LangChain
- Google Gemini (embeddings: gemini-embedding-001, LLM: gemini-2.5-flash-lite)
- PostgreSQL + pgVector
- Docker & Docker Compose
