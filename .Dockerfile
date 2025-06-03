FROM python:3.11-slim

COPY ./src/requirements.txt ./src/requirements.txt
RUN pip install --no-cache-dir -r ./src/requirements.txt

CMD ["python", "./src/app.py"]
