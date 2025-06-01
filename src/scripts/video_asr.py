import os

import whisper
from audio_extract import extract_audio

from pathlib import Path

model_type = os.getenv('WHISPER_MODEL', 'tiny')  # "tiny", "base", "small", "medium", "large"
default_audio_dir = os.getenv('AUDIO_DIR', 'audio_dir_temp')
default_video_dir = os.getenv('VIDEO_DIR', 'video_dir')
default_texts_dir = os.getenv('TEXTS_DIR', 'texts_dir')


def transcribe_audio(model, audio_path):
    """Recognizes audio from .mp3 audio to text"""
    result = model.transcribe(audio_path)
    return result['text']


def prepare_env(dirs: list[str]):
    """Create dirs for following functions"""
    for path in dirs:
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)


def convert_video_file(source_path, destination_path):
    """Extract audio file to video file"""
    extract_audio(
        input_path=source_path,
        output_path=destination_path,
        overwrite=True,
    )


def write_text(filenames: list, contents: list) -> None:
    """Write contents to files with corresponding filenames"""
    assert len(filenames) == len(contents)

    for filename, content in zip(filenames, contents):
        with open(filename, 'w') as textf:
            textf.writelines(content)


def video_to_audio(video_dir=default_video_dir, audio_dir=default_audio_dir) -> None:
    """Convert video files to audio files"""
    print('Fetching video filenames...')
    videos = [item for item in os.listdir(video_dir) if '.mp4' in item]
    print('Fetched following videos: ', *videos, sep='\n- ', end='\n\n')

    for video in videos:
        source = os.path.join(video_dir, video)
        dest = video.split('.mp4')[0] + '.mp3'
        convert_video_file(source, os.path.join(audio_dir, dest))
        print('-', dest, 'is converted')


def audio_to_text(audio_dir=default_audio_dir, texts_dir=default_texts_dir) -> None:
    """Convert audio files to text files"""
    print('Fetching audio filenames...')
    audios = [item for item in os.listdir(audio_dir) if '.mp3' in item]
    print('Fetched following audios: ', *audios, sep='\n- ', end='\n\n')

    asr_model = whisper.load_model(model_type)

    for audio in audios:
        text = transcribe_audio(asr_model, os.path.join(audio_dir, audio))
        filename = os.path.join(texts_dir, audio.split('.mp3')[0] + '.txt')
        print('-', audio, 'is converted')

        try:
            with open(filename, 'w') as textf:
                textf.writelines(text)
                print('File is written')
        except Exception as e:
            print('Something went wrong:', e)


def text_from_video(path: str) -> str:
    """Get text from video"""
    audio_path = path.split('.mp4')[0] + '.mp3'
    convert_video_file(path, audio_path)
    asr_model = whisper.load_model(model_type)
    text = transcribe_audio(asr_model, audio_path)
    os.remove(audio_path)
    return text


def save_text(video_path: str, save_dir: str = '.') -> None:
    """Save text file from video"""
    text = text_from_video(video_path)
    new_name = Path(video_path).name.split('.mp4')[0] + '.txt'
    write_text([os.path.join(save_dir, new_name)], [text])


def end2end_pipeline() -> None:
    print('Preparing environment...')
    prepare_env([default_video_dir, default_audio_dir, default_texts_dir])
    print('Environment is ready', end='\n\n')

    print('Start converting videos to audio files...')
    video_to_audio()
    print('Converting is completed', end='\n\n')

    print('Start recognition audio files...')
    audio_to_text()
    print('Recognition is completed', end='\n\n')

    print("Finishing script...")


if __name__ == '__main__':
    audio_to_text()