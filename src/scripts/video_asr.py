import os
import whisper

model_type = os.getenv('WHISPER_MODEL', 'tiny')  # e.g. "tiny", "base", "small", "medium", "large"
audio_dir = os.getenv('AUDIO_DIR', 'videos')

if __name__ == '__main__':
    print('Fetching audio filenames...')
    audios = [item for item in os.listdir(audio_dir) if '.wav' in item]
    print('Fetched following audios: ', *audios, sep='\n- ', end='\n\n')

    asr_model = whisper.load_model(model_type)

    texts = []
    for audio in audios:
        text = asr_model.transcribe(os.path.join(audio_dir, audio))['text']
        text.append(text)
