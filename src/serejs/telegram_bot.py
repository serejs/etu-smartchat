import logging
import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters, ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    CURRENT_STATE = 1

    await update.message.reply_text(
        f"""Hi, {update.message.from_user.first_name}!\nI'm a SmartChat, how can I help you?""")

    return CURRENT_STATE


async def conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    message = update.message.text
    logger.info("Message of %s: %s", user.first_name, message)

    await update.message.reply_text(f"{(message)[::-1]}")
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
