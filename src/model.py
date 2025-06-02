from yandex_cloud_ml_sdk import YCloudML

import os

sdk = YCloudML(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),
    auth=os.getenv("YANDEX_AUTH")
)

model = sdk.models.completions("llama").langchain()
