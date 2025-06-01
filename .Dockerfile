FROM ghcr.io/abetlen/llama-cpp-python:v0.3.5

COPY ./src/requirements.txt ./src/requirements.txt
RUN pip install --no-cache-dir -r ./src/requirements.txt

CMD ["python", "./src/app.py"]
