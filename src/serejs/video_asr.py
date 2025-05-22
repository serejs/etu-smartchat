import os

import whisper
from audio_extract import extract_audio

from pathlib import Path

model_type = os.getenv('WHISPER_MODEL', 'base')  # "tiny", "base", "small", "medium", "large"
audio_dir = os.getenv('AUDIO_DIR', 'audio_dir')
video_dir = os.getenv('VIDEO_DIR', 'video_dir')
texts_dir = os.getenv('TEXTS_DIR', 'texts_dir')


def transcribe_audio(model, audio_path):
    result = model.transcribe(audio_path)
    return result['text']


def prepare_env(dirs: list[str]):
    for path in dirs:
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)


def from_video_to_audio(source_path, destination_path):
    extract_audio(
        input_path=source_path,
        output_path=destination_path,
        overwrite=True,
    )


def write_text(filenames: list, contents: list):
    assert len(filenames) == len(contents)

    for filename, content in zip(filenames, contents):
        with open(filename, 'w') as textf:
            textf.writelines(content)


def end2end_pipeline():
    print('Preparing environment...')
    prepare_env([video_dir, audio_dir, texts_dir])
    print('Environment is ready', end='\n\n')

    print('Fetching video filenames...')
    videos = [item for item in os.listdir(video_dir) if '.mp4' in item]
    print('Fetched following videos: ', *videos, sep='\n- ', end='\n\n')

    print('Start converting videos to audio files...')
    audios = []
    for video in videos:
        source = os.path.join(video_dir, video)
        dest = video.split('.mp4')[0] + '.mp3'
        from_video_to_audio(source, os.path.join(audio_dir, dest))
        audios.append(dest)
        print('-', dest, 'is converted')
    print('Converting is completed', end='\n\n')

    asr_model = whisper.load_model(model_type)

    print('Start recognition audio files...')
    filenames = []
    texts = []
    for audio in audios:
        text = transcribe_audio(asr_model, os.path.join(audio_dir, audio))
        filenames.append(os.path.join(texts_dir, audio.split('.mp3')[0] + '.txt'))
        texts.append(text)
        print('-', audio, 'is converted')
    print('Recognition is completed', end='\n\n')

    print("Writing result...")
    write_text(filenames, texts)
    print("Finishing script")


if __name__ == '__main__':
    end2end_pipeline()
