import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from telegram import Update
from telegram.ext import ContextTypes

from db import collection
from model import model

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

try:
    with open('prompts/system_prompt.txt', 'r', encoding='utf-8') as file:
        system_prompt = file.read()
except Exception as e:
    print('Using default prompt, error fetching file', e)
    system_prompt = """Ты — ассистент. Отвечающий на вопросы пользователей **ИСКЛЮЧИТЕЛЬНО** на основе предоставленного контекста. Если ответа нет в контексте скажи, что ты не знаешь ответа. Для ответа используй не более 5 предложений
Контекст: {context}
"""

retriever = db.as_retriever()

init_model(temperature=0.05,
           max_tokens=1024,
           n_ctx=32768,
           n_gpu_layers=-1,
           n_batch=512,
           verbose=False)

llm = get_model()

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', f"<|im_start|><|system|>{system_prompt}<|im_end|>"),
        ('user', '<|im_start|><|user|>{input}<|im_end|><|endoftext|>')
    ]
)


def format_docs(docs):
    logger.info("Sources:\n%s\n", '\n'.join(doc.metadata['source'] for doc in docs))
    return "\n\n".join(doc.page_content for doc in docs)


def log(inp):
    logger.info(inp)
    return inp


# A processing chain for user input and response generation.
chain = (
        {
            "context": db.as_retriever() | format_docs,
            "input":   RunnablePassthrough(),
        }
        | prompt | log
        | llm
        | StrOutputParser()
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Sends a welcome message to the user.

    Parameters:
    - update: Incoming message and user info.
    - context: Context for the callback.

    Returns:
    - int: next state
    """
    await update.message.reply_text(
        f"""Привет, чем я могу помочь?""") # TODO: Улучшить приветственное сообщение

    return 1


async def conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Responds to user messages with a generated answer.

    Parameters:
    - update: Incoming message and user info.
    - context: Context for the callback.

    Returns:
    - int: next state
    """
    user = update.message.from_user
    message = update.message.text
    logger.info("Message of %s: %s", user.first_name, message)
    res = chain.invoke(message)
    logger.info("Model answer: %s", res)

    await update.message.reply_text(res)
    return 1
