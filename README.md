# J.A.R.V.I.S. — Голосовой ИИ-ассистент для Windows

Голосовой помощник в стиле Джарвиса: PySide6, распознавание речи, нейроголос,
управление компьютером, локальные модели (GGUF/Hugging Face) и агентский режим
с function calling.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB)
![PySide6](https://img.shields.io/badge/PySide6-Qt6-41CD52)

## Возможности

- **Распознавание речи**: Vosk (полностью офлайн), Faster-Whisper (локально), Google Speech
- **Нейроголос**: Edge TTS (бесплатно, Microsoft), Piper (офлайн, ONNX), ElevenLabs, Silero
- **ИИ-модели**: OpenRouter, Gemini, Mistral, DeepSeek, Ollama, LM Studio, локальные GGUF
- **Офлайн-команды**: время, дата, громкость, приложения, клавиатуры, поиск — без интернета
- **Управление ПК**: приложения, сайты, скриншоты, ввод текста, медиа-клавиши, окна
- **Локальные модели**: скачивание .gguf с Hugging Face (многопоточно, с Hugging Face), запуск через llama.cpp
- **3D-визуализатор**: голографическая сфера из малых сфер с эмоциями и предметами
- **Кэш озвучки**: повторы фраз мгновенно, без запроса к Edge
- **Прогрев голоса**: типовые фразы синтезируются в фоне при запуске
- **Авто-прослушивание**: реакция на имя «Джарвис» / «Карл»
- **Агентский режим**: модель сама вызывает инструменты (function calling)

## Установка

```powershell
cd d:\jarvis
python -m pip install -r requirements.txt
```

## Запуск

```powershell
python main.py
```

Отладка без аудио:
```powershell
python main.py --no-audio
```

## Сборка exe

```powershell
python -m PyInstaller --clean --noconfirm --name=JARVIS ^
  --onedir --windowed --uac-admin ^
  --distpath dist_new ^
  --add-data "config.json;." ^
  --icon=logo.ico ^
  --hidden-import=pythoncom ^
  --hidden-import=comtypes ^
  --exclude-module=PyQt5,PyQt6,PySide2,matplotlib,scipy,torch,torchaudio,pandas,IPython,notebook,pytest ^
  main.py

Copy-Item config.json dist_new\JARVIS\config.json
Copy-Item -Recurse -Force *.py dist_new\JARVIS\
Remove-Item -Recurse -Force dist\JARVIS
Move-Item dist_new\JARVIS dist\JARVIS
```

## Настройки

Запуск → Настройки (⚙️):

| Вкладка | Что настраивать |
|---|---|
| **Модель ИИ** | Провайдер, API-ключ, модель, базовый URL |
| **Локальные модели** | Каталог моделей (подбирается по железу), скачивание .gguf, GPU/CPU |
| **Голос** | Edge/PiPer/ElevenLabs, голос, скорость, эффект «Джарвиса», кэш, прогрев |
| **Распознавание** | STT-движок, модель, устройство микрофона |
| **Аватар** | 2D-аватар, прогресс-бар, субтитры, виджет на рабочем столе |

## Локальные модели

1. Вкладка «Локальные модели» → выбрать из каталога (подбирается по VRAM/RAM)
2. «Скачать модель» — многопоточная загрузка с Hugging Face, продолжение после обрыва
3. Сервер запускается автоматически (или вручную кнопкой)
4. На вкладке «Модель ИИ» выбрать провайдера **Ollama** (или `localgguf`)

### Установка Ollama / LM Studio вручную

**Ollama** (рекомендуется):
1. Скачать с [ollama.com](https://ollama.com) → установить
2. `ollama pull qwen2.5:3b` (или другая модель)
3. В Джарвисе: провайдер Ollama, модель `qwen2.5:3b`

**LM Studio**:
1. Скачать с [lmstudio.ai](https://lmstudio.ai) → установить
2. Запустить Local Server (порт 1234)
3. В Джарвисе: провайдер OpenAI-compatible, URL `http://localhost:1234/v1`

## Офлайн-команды

Работают без интернета и ИИ:

- «который час» / «какая дата» / «статус системы»
- «громче» / «тише» / «выключи звук» / «включи звук»
- «скриншот» / «закрой окно» / «блокируй» / «выключи компьютер»
- «открой браузер» / «калькулятор» / «проводник» / «блочнот»
- «новая вкладка» / «обнови» / «скопируй» / «вставь» / «отмени»
- «сверни все окна» / «переключи окно» / «полный экран»
- «найди [что-то]» / «открой [сайт]» / «курс» / «погода»
- «привет» / «как дела» / «спасибо»

## Структура

```
config.json          # настройки (без ключей, для контроля версий)
config.py            # пути, константы, класс Settings
main.py              # точка входа, логирование, обработка ошибок
ui.py                # PySide6 интерфейс, MainWindow
ai_manager.py        # ИИ-запросы, провайдеры, агенты
speech.py            # TTS (Edge/Pyttsx3/Piper/ElevenLabs/Silero), STT
commands.py          # офлайн-команды
os_control.py        # ToolRegistry: 28+ инструментов управления ПК
local_models.py      # каталог .gguf, загрузка с Hugging Face, llama.cpp
avatar.py            # 2D-аватар и прогресс-бар
chat_history.py      # история диалогов (JSON)
jarvis.txt           # исходные идеи и фишки (для справки)
logo.jpg             # логотип
requirements.txt     # зависимости
```

## Логирование

Лог: `logs/jarvis.log`
Дампы: `logs/crash.log`

## Приватный репозиторий

Проект в разработке. `config.json` и каталог `models/` исключены из `.gitignore`.

