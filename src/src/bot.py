import logging
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters, ContextTypes,
)

from src.model import init_model, get_model
from src.db import db

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

system_prompt = (
    """Ты — ассистент, отвечающий на вопросы пользователей **ИСКЛЮЧИТЕЛЬНО** на основе предоставленного контекста:
     {context}
     
     Твои действия:
1. Внимательно анализируй вставленный контекст из сообщения пользователя
2. Отвечай ТОЛЬКО если информация есть в контексте
3. Если ответа в контексте нет — честно говори «В предоставленной информации нет ответа на этот вопрос»
4. Никогда не используй внешние знания или предположения

**Формат ответа:**
- Четкий ответ по существу
- Без пояснений о своей работе
- Без упоминания «контекста» в ответе

**Важно:**
- НЕ дополняй информацию, даже если тема тебе знакома
- НЕЛЬЗЯ выходить за рамки контекста ни при каких условиях"""
)

retriever = db.as_retriever()

init_model(temperature=0.1,
           max_tokens=1024,
           n_ctx=32768,
           n_gpu_layers=40,
           n_batch=512,
           verbose=False)

llm = get_model()

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', f"<|im_start|><|system|>{system_prompt}<|im_end|>"),
        ('user', '<|im_start|><|user|>{input}<|im_end|><|endoftext|>')
    ]
)

qa_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, qa_chain)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    CURRENT_STATE = 1

    await update.message.reply_text(
        f"""Привет, чем я могу помочь?""")

    return CURRENT_STATE


async def conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    message = update.message.text
    logger.info("Message of %s: %s", user.first_name, message)
    res = chain.invoke({"input": message})
    logger.info("Model answer: %s", res)

    await update.message.reply_text(res['answer'])
    return 1


def main() -> None:
    application = Application.builder().token(os.getenv('TG_TOKEN')).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={1: [MessageHandler(filters.TEXT, conversation)], },
        fallbacks=[],
    )

    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
