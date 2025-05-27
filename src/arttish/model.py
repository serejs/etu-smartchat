from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatLlamaCpp

import os

model_repo = "gaianet/Qwen2.5-7B-Instruct-GGUF"
model_basename = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

model_path = "C:/Практика/etu-smartchat/src/models/" + model_basename

if not os.path.exists(model_path):
    print(f"Downloading model {model_basename}")
    model_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_basename,
        local_dir="C:/Практика/etu-smartchat/models",
    )
else:
    print(f"Model {model_basename} is already downloaded.")

llm = None


def get_model():
    global llm
    if llm is None:
        llm = ChatLlamaCpp(
            model_path=model_path,
            temperature=0.1,
            max_tokens=1024,
            n_ctx=32768,
            n_gpu_layers=40,
            n_batch=512,
            verbose=False,
        )
    return llm
