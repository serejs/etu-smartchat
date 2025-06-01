from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatLlamaCpp

import os

model_repo = os.getenv("MODEL_REPO")
model_basename = os.getenv("MODEL_BASENAME")

model_dir = os.getenv('MODELS_DIR', './models')

model_path = os.path.join(model_dir, model_basename)

if not os.path.exists(model_path):
    print(f"Downloading model {model_basename}")
    model_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_basename,
        local_dir=model_dir,
    )
else:
    print(f"Model {model_basename} is already downloaded.")

model = None


def init_model(**kwargs):
    global model
    if model is None:
        model = ChatLlamaCpp(
            model_path=model_path,
            **kwargs
        )


def get_model() -> ChatLlamaCpp:
    global model
    if model is None:
        init_model()
    if not isinstance(model, ChatLlamaCpp):
        raise RuntimeError("Something went wrong")
    return model
