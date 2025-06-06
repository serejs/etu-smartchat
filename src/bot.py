import logging
import numpy as np

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from telegram import Update
from telegram.ext import ContextTypes

from db import collection
from model import model, sdk
from chromadb import QueryResult

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
    system_prompt = """Ты — ассистент, отвечающий на вопросы пользователей **ИСКЛЮЧИТЕЛЬНО** на основе предоставленного контекста
{context}

Твои действия:
1. Внимательно анализируй вставленный контекст из сообщения пользователя
2. Отвечай ТОЛЬКО если информация есть в контексте
3. Если ответа в контексте нет — честно говори «В предоставленной информации нет ответа на этот вопрос»
4. Никогда не используй внешние знания или предположения
5. ТОЛЬКО ЕСЛИ ответ БЫЛ найден в контексте то в конце выпиши источники В ТОМ ЖЕ ФОРМАТЕ, что тебе был передан, иначе не выводи НИЧЕГО

**Формат ответа:**
- Четкий ответ по существу
- Без пояснений о своей работе
- Без упоминания «контекста» в ответе
- В ответе содержится не более трех предложений

**Важно:**
- НЕ дополняй информацию, даже если тема тебе знакома
- НЕЛЬЗЯ выходить за рамки контекста ни при каких условиях"""

query_embedder = sdk.models.text_embeddings("query")


def retrieve(query):
    query_embed = np.array(query_embedder.run(query))
    return collection.query(query_embeddings=query_embed, n_results=4)


llm = model

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', f"<|im_start|><|system|>{system_prompt}<|im_end|>"),
        ('user', '<|im_start|><|user|>{input}<|im_end|><|endoftext|>')
    ]
)


def format_docs(docs: QueryResult):
    logger.info(docs)
    sources = []
    for metadata in docs["metadatas"][0]:
        source = []
        if "course" in metadata:
            source.append(metadata["course"])
        if "module" in metadata:
            source.append(metadata["module"])
        if "source" in metadata:
            source.append(metadata["source"])
        if "url" in metadata:
            source.append(metadata["url"])
        source = '-'.join(source)
        sources.append(source)
    sources = '\n'.join(list(set(sources)))

    logger.info("Sources:\n%s\n", sources)
    return "\n".join(["Источники:", sources, "Контекст:", '\n\n'.join(docs['documents'][0])])


def log(inp):
    logger.info(inp)
    return inp


# A processing chain for user input and response generation.
chain = (
        {
            "context": RunnablePassthrough() | retrieve | format_docs,
            "input": RunnablePassthrough(),
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
    # docs = collection.get(include=["metadatas"])
    # modules = set()
    # for meta in docs["metadatas"]:
    #     if "module" in meta:
    #         modules.add(meta["module"])
    # modules = '\n'.join(list(modules))
    subjects = [
        "Разработка программных ансамблей и сервисов ИИ на базе больших языков моделей"
    ]
    subjects = '\n'.join(subjects)
    await update.message.reply_text(
f"""Привет, я являюсь помощником по программам online обучения института Искуственного Интеллекта им А. С. Попова  (ЛЭТИ). 
Я только учусь, на данный момент я обучен по дисциплиннам: \n{subjects}\n
Задавайте мне вопросы в рамках этих компетенций и я буду рад на них ответить.""")

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
