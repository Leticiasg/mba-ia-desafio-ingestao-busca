import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search import search_prompt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Chat com o PDF - Digite 'sair' para encerrar")

    while True:
        try:
            pergunta = input("\nPERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            logger.info("Encerrando chat. Até logo!")
            break

        if pergunta.lower() == "sair":
            logger.info("Encerrando chat. Até logo!")
            break

        if not pergunta:
            continue

        resposta = search_prompt(pergunta)
        print(f"\nRESPOSTA: {resposta}")


if __name__ == "__main__":
    main()
