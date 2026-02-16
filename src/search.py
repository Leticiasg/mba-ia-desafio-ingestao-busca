import logging
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")
LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_prompt(question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

        store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
        )
    except Exception as e:
        logger.error("Falha ao conectar ao banco vetorial: %s", e)
        return "Erro: não foi possível conectar ao banco de dados. Verifique se o PostgreSQL está rodando."

    try:
        results = store.similarity_search_with_score(question, k=10)
    except Exception as e:
        logger.error("Falha na busca semântica: %s", e)
        return "Erro: falha ao realizar a busca semântica."

    contexto = "\n".join([doc.page_content for doc, _ in results])

    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)
        response = llm.invoke(prompt)
    except Exception as e:
        logger.error("Falha ao chamar a LLM: %s", e)
        return "Erro: falha ao gerar resposta. Verifique sua GOOGLE_API_KEY."

    return response.content
