## Сборка и запуск проекта

```bash
docker compose build
docker compose up
```

[Пример .env файла](src/example.env)


```bash
python video_asr.py -m \path\to\meta.json -s 1000 -o 200 -a path\to\video_dir -e path\to\.env
```

## Настройка бота в телеграмме
Зайти к `@BotFather` в telegram.
Нажать `start`, далее `/newbot `
Предлагается выбрать имя, а затем его логин.
После этого предоставляется токен, который стоит добавить в .env, параметр `TG_TOKEN`
