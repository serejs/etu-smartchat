## Сборка и запуск проекта

```bash
docker compose build
docker compose up
```

[Пример .env файла](src/example.env)


```bash
python video_asr.py -m \path\to\meta.json -s 1000 -o 200 -a path\to\video_dir -e path\to\.env
```