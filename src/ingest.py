import logging
import math
import os
import sys
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "10"))
BATCH_DELAY = int(os.getenv("INGEST_BATCH_DELAY", "60"))


def ingest_pdf():
    if not PDF_PATH or not os.path.isfile(PDF_PATH):
        logger.error("PDF não encontrado: %s", PDF_PATH)
        sys.exit(1)

    if not DATABASE_URL:
        logger.error("DATABASE_URL não configurada")
        sys.exit(1)

    logger.info("Carregando PDF: %s", PDF_PATH)
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    logger.info("PDF carregado: %d páginas", len(documents))

    logger.info("Dividindo em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_documents(documents)
    total_batches = math.ceil(len(chunks) / BATCH_SIZE)
    logger.info("Total de chunks: %d (%d lotes)", len(chunks), total_batches)

    logger.info("Gerando embeddings e salvando no banco de dados...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    first_batch = chunks[:BATCH_SIZE]
    store = PGVector.from_documents(
        documents=first_batch,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        pre_delete_collection=True,
    )
    logger.info("Lote 1/%d: %d chunks salvos", total_batches, len(first_batch))

    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        logger.info("Aguardando %ds para respeitar o rate limit...", BATCH_DELAY)
        time.sleep(BATCH_DELAY)

        store.add_documents(batch)
        logger.info("Lote %d/%d: %d chunks salvos", batch_num, total_batches, len(batch))

    logger.info("Ingestão concluída com sucesso!")


if __name__ == "__main__":
    ingest_pdf()
