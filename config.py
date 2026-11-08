"""Конфигурация приложения J.A.R.V.I.S.

Все настройки хранятся в config.json рядом с приложением.
API-ключи лежат в открытом виде (локальный файл), поэтому не передавайте
config.json третьим лицам и не коммитьте его в git.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Пути приложения
# ---------------------------------------------------------------------------


def _app_dir() -> Path:
    """Каталог с пользовательскими данными.

    В frozen-сборке (exe) — папка рядом с исполняемым файлом, чтобы config.json,
    логи, скриншоты и модели голосов были видны и оставались после запуска.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _bundle_dir() -> Path:
    """Каталог встроенных в exe ресурсов (только чтение)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", str(_app_dir())))
    return Path(__file__).resolve().parent


APP_DIR = _app_dir()
BUNDLE_DIR = _bundle_dir()
CONFIG_PATH = APP_DIR / "config.json"
# config.json из сборки: используем как заготовку при первом запуске exe,
# чтобы ключи и настройки пользователя переехали рядом с программой.
SEED_CONFIG_PATH = BUNDLE_DIR / "config.json"
SCRIPTS_DIR = APP_DIR / "scripts"          # сгенерированные ИИ скрипты
ASSETS_DIR = APP_DIR / "assets"
OFFLINE_AUDIO_DIR = ASSETS_DIR / "offline"  # wav/mp3 для офлайн-фраз
SCREENSHOTS_DIR = APP_DIR / "screenshots"
LOGS_DIR = APP_DIR / "logs"
CHATS_DIR = APP_DIR / "chats"              # истории диалогов
MODELS_DIR = APP_DIR / "models"            # локальные .gguf с Hugging Face
RUNTIME_DIR = APP_DIR / "runtime"          # llama.cpp сервер для локальных моделей

# ---------------------------------------------------------------------------
# Каталог LLM-провайдеров
# ---------------------------------------------------------------------------
# kind:
#   "openai"    — OpenAI-совместимый /chat/completions
#   "anthropic" — нативный Messages API
# base_url — адрес по умолчанию, пользователь может изменить в настройках.

PROVIDERS: dict[str, dict[str, Any]] = {
    "openrouter": {
        "label": "OpenRouter",
        "kind": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "needs_key": True,
        "hint": "Ключ: openrouter.ai/keys. Модель указывается в формате vendor/model.",
    },
    "omniroute": {
        "label": "OmniRoute (локальный шлюз)",
        "kind": "openai",
        "base_url": "http://localhost:20128/v1",
        "default_model": "openai/gpt-4o-mini",
        "needs_key": False,
        "hint": "Бесплатный локальный шлюз к 500+ моделям: npm i -g omniroute → "
                "omniroute → панель на http://localhost:20128. Провайдеры и ключи "
                "задаются внутри OmniRoute, здесь ключ не нужен.",
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "kind": "anthropic",
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-5",
        "needs_key": True,
        "hint": "Ключ: console.anthropic.com. Модель, например claude-sonnet-4-5.",
    },
    "openai": {
        "label": "OpenAI (ChatGPT)",
        "kind": "openai",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "needs_key": True,
        "hint": "Ключ: platform.openai.com/api-keys.",
    },
    "ollama": {
        "label": "Ollama (локально)",
        "kind": "openai",
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5:7b",
        "needs_key": False,
        "hint": "Запустите `ollama serve` и скачайте модель: `ollama pull qwen2.5:7b`.",
    },
    "lmstudio": {
        "label": "LM Studio (локально)",
        "kind": "openai",
        "base_url": "http://localhost:1234/v1",
        "default_model": "local-model",
        "needs_key": False,
        "hint": "В LM Studio включите Local Server (OpenAI-compatible API).",
    },
    # Встроенный рантайм Джарвиса: сам скачивает llama.cpp и поднимает сервер.
    "localgguf": {
        "label": "Локальная модель (встроенный llama.cpp)",
        "kind": "openai",
        "base_url": "http://127.0.0.1:8077/v1",
        "default_model": "local-model",
        "needs_key": False,
        "hint": "Модель GGUF из вкладки «Локальные модели»: Карл сам скачивает "
                "llama.cpp, поднимает сервер и работает офлайн — интернет не нужен.",
    },
    # Провайдеры, которые доступны из России без VPN:
    "gemini": {
        "label": "Google Gemini (без VPN)",
        "kind": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash",
        "needs_key": True,
        # У Gemini путь уже содержит нужную версию: /v1/chat/completions добавлять нельзя.
        "append_v1": False,
        "free_model": "gemini-2.0-flash",
        "hint": "Бесплатный ключ: aistudio.google.com/apikey (нужен аккаунт Google). "
                "Работает без VPN, у бесплатного тарифа щедрые лимиты.",
    },
    "mistral": {
        "label": "Mistral AI (без VPN)",
        "kind": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "open-mistral-nemo",
        "needs_key": True,
        "free_model": "open-mistral-nemo",
        "hint": "Бесплатный ключ: console.mistral.ai. Работает без VPN. "
                "Модель open-mistral-nemo — на бесплатном тарифе (Experiment).",
    },
    "deepseek": {
        "label": "DeepSeek (без VPN)",
        "kind": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "needs_key": True,
        "hint": "Ключ: platform.deepseek.com. Работает без VPN, очень дешёвый.",
    },
    "fireworks": {
        "label": "Fireworks AI (без VPN)",
        "kind": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "needs_key": True,
        "hint": "Ключ: fireworks.ai. Работает без VPN, есть бесплатная квота.",
    },
    "pollinations": {
        "label": "Pollinations AI (бесплатно, без ключа, без VPN)",
        "kind": "pollinations",
        "base_url": "https://text.pollinations.ai",
        "default_model": "openai",
        "needs_key": False,
        "hint": "Абсолютно бесплатно, без ключа и VPN, НО отвечает не чаще одного "
                "сообщения примерно в 45 секунд на IP — Карл сам подождёт окно или "
                "уйдёт на другой провайдер. Для быстрых ответов добавьте бесплатный "
                "ключ Gemini (aistudio.google.com/apikey) или Mistral "
                "(console.mistral.ai) и выберите их здесь.",
    },
}

TTS_ENGINES = {
    "edge": "Edge TTS — нейроголос, онлайн, без ключа",
    "piper": "Piper — нейроголос, офлайн",
    "elevenlabs": "ElevenLabs — облако, клонирование голоса",
    "pyttsx3": "pyttsx3 — системный голос (робот)",
    "silero": "Silero TTS — офлайн, нужен torch",
}

STT_ENGINES = {
    "auto": "Авто: Faster-Whisper → Vosk (локально)",
    "faster-whisper": "Faster-Whisper (локально)",
    "vosk": "Vosk (локально)",
    "google": "Google Speech (облако, только вручную)",
}


def ensure_dirs() -> None:
    """Создаёт рабочие каталоги приложения."""
    for path in (
        SCRIPTS_DIR, OFFLINE_AUDIO_DIR, SCREENSHOTS_DIR, LOGS_DIR,
        CHATS_DIR,
        ASSETS_DIR / "piper", ASSETS_DIR / "whisper",
    ):
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Пользовательские настройки ассистента."""

    # --- Личность ---------------------------------------------------------
    assistant_name: str = "Джарвис"
    wake_word_enabled: bool = True
    auto_start_listening: bool = True
    # Одноразовый режим: после имени — одна команда, затем снова ждём имя.
    # False = режим диалога: слушаем команды без имени до слова «стоп».
    one_shot_mode: bool = True
    autostart: bool = False        # запуск ассистента вместе с системой
    reply_language: str = "auto"   # auto / ru / en

    # --- LLM --------------------------------------------------------------
    provider: str = "openrouter"
    model: str = "openai/gpt-4o-mini"
    # Модель, на которую переключаемся при нехватке кредитов на основной.
    fallback_model: str = ""
    api_keys: dict[str, str] = field(default_factory=dict)
    base_urls: dict[str, str] = field(default_factory=dict)
    temperature: float = 0.4
    max_tool_steps: int = 0        # 0 = без ограничений
    # --- Отдельная модель для кодинга -------------------------------------
    # Пусто = кодинг идёт через основную модель/провайдера.
    # Можно держать разговорную модель в облаке, а кодинг — локально (Ollama).
    coding_provider: str = ""
    coding_model: str = ""
    coding_temperature: float = 0.2
    # Лимит токенов ответа. OpenRouter резервирует кредиты под max_tokens,
    # поэтому большое значение приводит к ошибке 402.
    max_tokens: int = 2048
    # Озвучивать, что модель говорит по ходу работы (её комментарии и статусы).
    speak_reasoning: bool = True

    # --- Локальные модели (GGUF / llama.cpp) -----------------------------
    local_model_path: str = ""       # скачанный файл модели
    local_port: int = 8077           # порт встроенного сервера
    local_ctx: int = 4096            # размер контекста
    local_gpu_layers: int = -1       # -1 = подобрать автоматически
    local_threads: int = 8           # потоков при загрузке с Hugging Face
    local_autostart: bool = True     # поднимать сервер вместе с приложением
    local_show_all: bool = False     # показывать все модели из каталога
    local_uncensored_only: bool = False  # показывать только без цензуры

    # --- TTS --------------------------------------------------------------
    tts_engine: str = "edge"
    tts_voice: str = ""            # подстрока имени голоса pyttsx3, пусто = авто
    tts_rate: int = 180
    # Edge TTS: бесплатный нейроголос Microsoft (нужен интернет).
    edge_voice_ru: str = "ru-RU-DmitryNeural"
    edge_voice_en: str = "en-GB-RyanNeural"   # британский мужской, «джарвисовский»
    edge_rate: str = "-8%"
    edge_pitch: str = "-4Hz"
    # Piper: нейроголос полностью офлайн, модель скачивается один раз.
    piper_voice_ru: str = "ru_RU-dmitri-medium"
    piper_voice_en: str = "en_GB-alan-medium"
    piper_length_scale: float = 1.08
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "nPczCjzI2devNBz1zQrb"  # Brian — глубокий, спокойный
    elevenlabs_model: str = "eleven_multilingual_v2"
    silero_speaker: str = "aidar"
    # Кэш озвучки: повторы фраз звучат мгновенно, без повтора запроса в Edge.
    voice_cache: bool = True
    # «Голос Джарвиса»: узкополосный фильтр + эхо поверх нейроголоса Edge.
    voice_fx: bool = False
    # Заранее озвучить типовые фразы при запуске — первые ответы быстрее.
    voice_prewarm: bool = True

    # --- STT --------------------------------------------------------------
    stt_engine: str = "auto"
    stt_model: str = "small"       # размер модели faster-whisper
    stt_device: str = "cpu"        # cpu / cuda; auto отключён (ломается без CUDA)
    stt_compute_type: str = "int8"
    vosk_model_path: str = ""
    language: str = "ru"
    input_device: str = ""         # индекс или часть имени устройства
    silence_threshold: float = 0.004
    mic_gain: float = 2.0
    wake_beep: bool = True
    # Перебивание: если владелец заговорил во время ответа — ассистент замолкает.
    barge_in: bool = True
    barge_in_threshold: float = 0.055

    # --- Возможности агента -----------------------------------------------
    # Если приложение не установлено — открыть его веб-версию в браузере.
    web_fallback: bool = True
    # Агент может «посмотреть» на экран (снимок передаётся модели как изображение).
    vision_enabled: bool = True
    vision_max_width: int = 1280

    # --- Сеть ---------------------------------------------------------------
    # Прокси не используется: сервисы (Pollinations, Gemini, Mistral, DeepSeek)
    # работают напрямую, без VPN и прокси.

    # --- Безопасность -----------------------------------------------------
    allow_shell: bool = True
    # False = полное доверие агенту: опасные действия выполняются без вопросов.
    confirm_dangerous: bool = False

    # --- Аватар -----------------------------------------------------------
    avatar_enabled: bool = False        # показывать аватара поверх окон
    avatar_scale: float = 1.0           # размер персонажа (0.4–2.5)
    avatar_speed: float = 1.0           # скорость перемещения по экрану
    avatar_opacity: float = 0.95        # прозрачность окна
    avatar_always_on_top: bool = True   # поверх всех окон
    avatar_wander: bool = True          # бегает по экрану
    avatar_subtitles: bool = True       # субтитры ответов
    avatar_objects: bool = True         # предметы-эмоции
    avatar_accent: str = ""             # свой цвет, пусто = по состоянию

    # ------------------------------------------------------------------
    def resolved_base_url(self, provider: str | None = None) -> str:
        """Возвращает base_url провайдера с учётом переопределения."""
        pid = provider or self.provider
        custom = (self.base_urls.get(pid) or "").strip()
        if custom:
            return custom
        return str(PROVIDERS.get(pid, {}).get("base_url", ""))

    def api_key(self, provider: str | None = None) -> str:
        """Возвращает API-ключ выбранного провайдера."""
        pid = provider or self.provider
        return (self.api_keys.get(pid) or "").strip()

    # --- сеть -------------------------------------------------------------
    def session(self) -> Any:
        """HTTP-сессия для запросов JARVIS.

        trust_env=False — requests игнорирует HTTP_PROXY/HTTPS_PROXY из
        окружения и системный прокси Windows.
        """
        import requests

        client = requests.Session()
        client.trust_env = False
        return client

    def provider_label(self) -> str:
        return str(PROVIDERS.get(self.provider, {}).get("label", self.provider))

    # --- кодинг-модель ----------------------------------------------------
    def coding_enabled(self) -> bool:
        """Задана ли отдельная кодинг-модель/провайдер."""
        return bool((self.coding_model or "").strip() or (self.coding_provider or "").strip())

    def coding_provider_id(self) -> str:
        """Провайдер кодинг-модели (по умолчанию — основной)."""
        return (self.coding_provider or "").strip() or self.provider

    def coding_provider_label(self) -> str:
        pid = self.coding_provider_id()
        return str(PROVIDERS.get(pid, {}).get("label", pid))

    def coding_model_resolved(self) -> str:
        """Имя кодинг-модели: своё или дефолт выбранного провайдера."""
        model = (self.coding_model or "").strip()
        if model:
            return model
        return str(PROVIDERS.get(self.coding_provider_id(), {}).get("default_model", ""))

    def token_limit(self) -> int:
        """Безопасный лимит токенов ответа (см. комментарий к max_tokens)."""
        try:
            value = int(self.max_tokens or 2048)
        except (TypeError, ValueError):
            value = 2048
        return max(256, min(value, 8192))

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        # Совместимость: раньше провайдер назывался «omnirouter».
        if values.get("provider") == "omnirouter":
            values["provider"] = "omniroute"
        if values.get("coding_provider") == "omnirouter":
            values["coding_provider"] = "omniroute"
        for key in ("api_keys", "base_urls"):
            mapping = values.get(key)
            if isinstance(mapping, dict) and "omnirouter" in mapping:
                mapping["omniroute"] = mapping.pop("omnirouter")
        return cls(**values)

    def save(self, path: Path | None = None) -> None:
        target = path or CONFIG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(target)
        # На Linux ограничиваем доступ к файлу с ключами.
        if os.name != "nt":
            try:
                target.chmod(0o600)
            except OSError:
                pass

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        target = path or CONFIG_PATH
        if not target.exists() and SEED_CONFIG_PATH.exists():
            # Первый запуск exe-сборки: копируем config.json из сборки
            # (там уже есть ключи и настройки пользователя).
            target.write_text(SEED_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        if not target.exists():
            settings = cls()
            settings.save(target)
            return settings
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("config.json должен содержать объект")
            return cls.from_dict(raw)
        except (OSError, ValueError, json.JSONDecodeError):
            # Повреждённый конфиг не должен блокировать запуск.
            return cls()


# ---------------------------------------------------------------------------
# Автозапуск вместе с системой
# ---------------------------------------------------------------------------

AUTOSTART_NAME = "JARVIS"


def _autostart_command() -> str:
    """Возвращает путь к исполняемому файлу (для schtasks)."""
    import sys

    if getattr(sys, "frozen", False):
        return f'"{Path(sys.executable).resolve()}"'

    executable = Path(sys.executable)
    pythonw = executable.with_name("pythonw.exe")
    if os.name == "nt" and pythonw.exists():
        executable = pythonw
    return f'"{executable}" "{APP_DIR / "main.py"}"'


# ---------------------------------------------------------------------------
# Автозапуск через Task Scheduler (требует прав администратора)
# ---------------------------------------------------------------------------

_AUTOTASK_NAME = r"\Jarvis"


def _create_autostart_task() -> bool:
    """Создаёт задачу в Планировщике заданий с правами HIGHEST.

    Работает ТОЛЬКО когда процесс запущен от имени администратора
    (а J.A.R.V.I.S. требует admin-прав при запуске).
    """
    import subprocess

    cmd = _autostart_command()

    # subprocess.run со списком аргументов (без shell=True).
    # Windows сама построит командную строку через list2cmdline(),
    # корректно экранируя кавычки в пути.
    try:
        result = subprocess.run(
            [
                "schtasks",
                "/create",
                "/tn",
                "JARVIS_AutoStart",
                "/tr",
                cmd,
                "/sc",
                "onlogon",
                "/rl",
                "HIGHEST",
                "/f",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            log.info("Autostart task created via schtasks.")
            return True
        log.warning("schtasks failed: %s", result.stderr.strip())
        return False
    except Exception as exc:
        log.warning("schtasks failed: %s", exc)
        return False


def _delete_autostart_task() -> bool:
    """Удаляет задачу из Планировщика заданий."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "schtasks",
                "/delete",
                "/tn",
                "JARVIS_AutoStart",
                "/f",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            log.info("Autostart task deleted.")
            return True
        log.warning("schtasks delete failed: %s", result.stderr.strip())
        return False
    except Exception as exc:
        log.warning("schtasks delete failed: %s", exc)
        return False


def _task_exists() -> bool:
    """Проверяет наличие задачи в Планировщике заданий."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "schtasks",
                "/query",
                "/tn",
                "JARVIS_AutoStart",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def is_autostart_enabled() -> bool:
    """Проверяет автозапуск через Task Scheduler или HKCU."""
    if _task_exists():
        return True
    # Fallback: HKCU Run (работает без админ-прав).
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
            ) as key:
                value, _ = winreg.QueryValueEx(key, AUTOSTART_NAME)
            return bool(value)
        except OSError:
            return False
    return (Path.home() / ".config" / "autostart" / "jarvis.desktop").exists()


def set_autostart(enabled: bool) -> str:
    """Включает/выключает автозапуск.

    При наличии прав администратора — Task Scheduler с HIGHEST.
    Иначе — HKCU Run (не поднимет права, но запомнит запись).
    """
    if os.name != "nt":
        return set_autostart_desktop(enabled)

    if enabled:
        # Пробуем Task Scheduler (поднимает права).
        if _create_autostart_task():
            return "Автозапуск включён (Планировщик заданий, HIGHEST)."
        # Fallback — HKCU Run.
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE,
            )
            with key:
                winreg.SetValueEx(
                    key, AUTOSTART_NAME, 0, winreg.REG_SZ, _autostart_command()
                )
            return "Автозапуск включён (HKCU Run)."
        except OSError as exc:
            return f"Не удалось включить автозапуск: {exc}"
    else:
        # Сначала пробуем удалить задачу schtasks.
        if _delete_autostart_task():
            return "Автозапуск выключен (Планировщик заданий)."
        # Fallback — HKCU.
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE,
            )
            with key:
                try:
                    winreg.DeleteValue(key, AUTOSTART_NAME)
                except FileNotFoundError:
                    pass
            return "Автозапуск выключен."
        except OSError as exc:
            return f"Не удалось выключить автозапуск: {exc}"


def set_autostart_desktop(enabled: bool) -> str:
    """Linux/macOS: автозапуск через .desktop файл."""
    desktop_dir = Path.home() / ".config" / "autostart"
    target = desktop_dir / "jarvis.desktop"
    if enabled:
        desktop_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=J.A.R.V.I.S.\n"
            f"Exec={_autostart_command()}\n"
            "Terminal=false\n"
            "Comment=Voice assistant\n",
            encoding="utf-8",
        )
        return "Автозапуск включён."
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:
        return f"Не удалось изменить автозапуск: {exc}"
    return "Автозапуск выключен."
