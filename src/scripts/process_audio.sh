#!/bin/bash

# shellcheck disable=SC2164
cd ./videos || { echo "Directory ./videos not found"; exit 1; }

# List files in the directory for debugging
echo "Files in ./videos:"
ls -la

apt update && apt install -y ffmpeg
# Loop through all MP4 files in the directory
for file in *.mp4; do
    # Check if there are any MP4 files
    if [ ! -e "$file" ]; then
        echo "No MP4 files found in ./videos."
        exit 0
    fi
    # Get the base name of the file (without extension)
    base_name="${file%.mp4}"
    # Convert to WAV
    ffmpeg -i "$file" -q:a 0 -map a "${base_name}.wav"
    echo "${base_name} completed"
done

cd ..
python3 ./video_asr.py -m ../../видео.json -s 1000 -o 200 -a ./video