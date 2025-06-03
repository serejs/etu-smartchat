FROM python:3.9-slim

WORKDIR /app

RUN apt update && apt install -y ffmpeg

RUN pip install -U openai-whisper

COPY ./process_audio.sh /app/process_audio.sh
COPY ./video_asr.py /app/video_asr.py
RUN chmod +x /app/process_audio.sh && ./app/process_audio.sh

ENV AUDIO_DIR=/videos

CMD ["python", "/app/video_asr"]
RUN rm /videos/*.wav
