"""Голосовой ввод (STT), wake word и озвучка (TTS).

STT:
  * Faster-Whisper — быстрый локальный движок (рекомендуется, CPU);
  * Vosk — лёгкий локальный движок;
  * Google Speech — облачный вариант (выбирается явно).

TTS:
  * Edge TTS — бесплатный нейроголос Microsoft, очень реалистичный (онлайн);
  * Piper — нейроголос полностью офлайн (модель скачивается один раз);
  * ElevenLabs — облако, клонирование голоса;
  * pyttsx3 — системный офлайн-голос (резервный);
  * Silero TTS — нейросетевой офлайн-голос (нужен torch).

Модуль не зависит от Qt: наружу отдаются обычные callbacks.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import ASSETS_DIR, OFFLINE_AUDIO_DIR, Settings

# Меньше мусора в логах HuggingFace при загрузке модели Whisper.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024          # ~64 мс при 16 кГц
MIN_COMMAND_SECONDS = 0.25

Callback = Callable[..., None]


# ---------------------------------------------------------------------------
# Микрофон
# ---------------------------------------------------------------------------

def resolve_input_device(value: str) -> int | None:
    """Превращает настройку (индекс или часть имени) в индекс sounddevice."""
    value = (value or "").strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    try:
        import sounddevice as sd

        for index, device in enumerate(sd.query_devices()):
            if value.lower() in str(device.get("name", "")).lower() and device.get(
                "max_input_channels", 0
            ) > 0:
                return index
    except Exception:
        log.exception("device lookup failed")
    return None


def list_input_devices() -> list[tuple[int, str]]:
    """Список доступных устройств записи: (индекс, имя)."""
    try:
        import sounddevice as sd

        result: list[tuple[int, str]] = []
        for index, device in enumerate(sd.query_devices()):
            if device.get("max_input_channels", 0) > 0:
                result.append((index, str(device.get("name", f"Устройство {index}"))))
        return result
    except Exception:
        log.exception("cannot list input devices")
        return []


class MicrophoneStream:
    """Потоковый захват моно-аудио с микрофона."""

    def __init__(
        self,
        device: int | None = None,
        samplerate: int = SAMPLE_RATE,
        blocksize: int = BLOCK_SIZE,
    ) -> None:
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: Any = None

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("audio status: %s", status)
        data = indata[:, 0].copy()
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            # Не блокируем аудиопоток: лучше потерять кадр, чем получить задержку.
            pass

    def read(self, timeout: float = 0.2) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


def _rms(audio: np.ndarray) -> float:
    if audio is None or len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _prepare_audio(audio: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Убирает постоянную составляющую и нормализует громкость для STT.

    Whisper плохо распознаёт тихую запись: микрофон ноутбука часто даёт
    пик 0.02–0.05. Поднимаем уровень до ~0.9, но не более чем в 20 раз,
    чтобы не превратить шум в «речь».
    """
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size == 0:
        return audio
    if gain and gain != 1.0:
        audio = np.clip(audio * float(gain), -1.0, 1.0)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0.004:
        factor = min(0.92 / peak, 20.0)
        if factor > 1.0:
            audio = np.clip(audio * factor, -1.0, 1.0)
    return audio


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

class BaseSTT:
    label = "STT"
    wake_prompt: str | None = None
    command_prompt: str | None = None
    confirm_prompt: str | None = None

    def transcribe(self, audio: np.ndarray, initial_prompt: str | None = None) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


# Кэш моделей: при смене настроек не перезагружаем 500-МБ модель заново.
_WHISPER_CACHE: dict[tuple[str, str, str], Any] = {}
_WHISPER_LOCK = threading.Lock()


def _load_whisper_model(size: str, device: str, compute_type: str) -> Any:
    key = (size, device, compute_type)
    with _WHISPER_LOCK:
        cached = _WHISPER_CACHE.get(key)
        if cached is not None:
            return cached

        from faster_whisper import WhisperModel

        try:
            model = WhisperModel(size, device=device, compute_type=compute_type)
        except Exception as exc:
            if device == "cpu":
                raise
            log.warning(
                "Whisper не смог стартовать на %s (%s). Переключаюсь на CPU.", device, exc
            )
            key = (size, "cpu", "int8")
            model = _WHISPER_CACHE.get(key) or WhisperModel(
                size, device="cpu", compute_type="int8"
            )
        _WHISPER_CACHE[key] = model
        return model


class FasterWhisperSTT(BaseSTT):
    """Локальное распознавание через faster-whisper (CTranslate2).

    По умолчанию используется CPU/int8: на машинах без CUDA режим
    device="auto" выбирает CUDA и падает с ошибкой cublas64_12.dll.
    """

    def __init__(self, settings: Settings, on_status: Callback | None = None) -> None:
        size = (settings.stt_model or "small").strip()
        device = (settings.stt_device or "cpu").strip().lower()
        if device in {"auto", "gpu", ""}:
            # auto ненадёжен: CTranslate2 может выбрать CUDA без библиотек.
            device = "cpu"
        compute_type = (settings.stt_compute_type or "").strip()
        if not compute_type:
            compute_type = "float16" if device == "cuda" else "int8"
        self.language = (settings.language or "ru").strip() or None
        if self.language and self.language.lower() in {"auto", "any"}:
            self.language = None
        self.size = size
        self.device = device
        self.compute_type = compute_type
        # Усиление уже применено в VoiceListener; здесь только нормализация.
        self._gain = 1.0
        # Разные подсказки для wake word и для команды: так Whisper не путает
        # короткое имя с командой и наоборот.
        if (self.language or "ru").lower().startswith("en"):
            self.wake_prompt = (
                f"{settings.assistant_name}. Open the browser. What time is it. "
                "Play music. Take a screenshot."
            )
            self.command_prompt = (
                "Open the browser. What time is it. Play music. "
                "Take a screenshot. Shut down the computer. Volume up."
            )
            self.confirm_prompt = "Yes. No. Yes, sir. No, sir. Stop."
        else:
            self.wake_prompt = (
                f"{settings.assistant_name}. Открой браузер. Который час. "
                "Включи музыку. Сделай скриншот."
            )
            self.command_prompt = (
                "Открой браузер. Который час. Включи музыку. Сделай скриншот. "
                "Выключи компьютер. Открой ютуб. Громче. Тише."
            )
            self.confirm_prompt = (
                "Да. Нет. Да, сэр. Нет, сэр. Стоп. Отбой."
            )
        self.initial_prompt = self.command_prompt
        if on_status:
            on_status(
                f"Загружаю Faster-Whisper «{size}» на {device.upper()} "
                "(первый запуск может скачать модель)..."
            )
        self._model = _load_whisper_model(size, device, compute_type)
        self.label = f"Faster-Whisper ({size}, {device})"

    # ------------------------------------------------------------------
    def _decode(self, audio: np.ndarray, vad: bool, prompt: str | None) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=3,
            vad_filter=vad,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            no_speech_threshold=0.6,
        )
        # Генератор ленивый: исключение возникает именно здесь, при обходе.
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe(
        self, audio: np.ndarray, initial_prompt: str | None = None
    ) -> str:
        # Отсекаем тишину: иначе Whisper любит «галлюцинировать» цифры и фразы.
        if _rms(audio) < 0.0006:
            return ""
        audio = _prepare_audio(audio, self._gain)
        if audio.size < SAMPLE_RATE * 0.2:
            return ""
        prompt = initial_prompt if initial_prompt is not None else self.command_prompt
        try:
            text = self._decode(audio, True, prompt)
            if not text:
                # VAD иногда вырезает всю короткую фразу — пробуем без него.
                text = self._decode(audio, False, prompt)
            return text
        except RuntimeError as exc:
            message = str(exc).lower()
            if "cublas" in message or "cudnn" in message or "cuda" in message:
                log.warning("CUDA недоступна (%s), перезапускаю модель на CPU", exc)
                self._model = _load_whisper_model(self.size, "cpu", "int8")
                self.device = "cpu"
                text = self._decode(audio, True, prompt)
                return text or self._decode(audio, False, prompt)
            raise


class VoskSTT(BaseSTT):
    """Локальное распознавание через Vosk."""

    def __init__(self, settings: Settings, on_status: Callback | None = None) -> None:
        import vosk

        path = (settings.vosk_model_path or "").strip()
        if not path:
            candidates = sorted(ASSETS_DIR.glob("vosk-model*"))
            if candidates:
                path = str(candidates[0])
        if not path or not Path(path).exists():
            raise RuntimeError(
                "Модель Vosk не найдена. Скачайте vosk-model-small-ru и укажите путь в настройках."
            )
        if on_status:
            on_status(f"Загружаю Vosk: {Path(path).name}...")
        vosk.SetLogLevel(-1)
        self._vosk = vosk
        self._model = vosk.Model(path)
        self.label = f"Vosk ({Path(path).name})"

    def transcribe(self, audio: np.ndarray, initial_prompt: str | None = None) -> str:
        recognizer = self._vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        recognizer.AcceptWaveform(pcm)
        try:
            return str(json.loads(recognizer.FinalResult()).get("text", "")).strip()
        except json.JSONDecodeError:
            return ""


class GoogleSTT(BaseSTT):
    """Облачный fallback через SpeechRecognition (нужен интернет)."""

    def __init__(self, settings: Settings, on_status: Callback | None = None) -> None:
        import speech_recognition as sr

        self._sr = sr
        lang = (settings.language or "ru").lower()
        self.language = "ru-RU" if lang.startswith("ru") else f"{lang[:2]}-{lang[:2].upper()}"
        self.label = "Google Speech (онлайн)"
        if on_status:
            on_status("Использую Google Speech (облако) — локальный STT не установлен.")

    def transcribe(self, audio: np.ndarray, initial_prompt: str | None = None) -> str:
        recognizer = self._sr.Recognizer()
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        data = self._sr.AudioData(pcm, SAMPLE_RATE, 2)
        try:
            return recognizer.recognize_google(data, language=self.language)
        except Exception:
            return ""


class NullSTT(BaseSTT):
    label = "STT недоступен"

    def transcribe(self, audio: np.ndarray, initial_prompt: str | None = None) -> str:
        return ""


def create_stt(settings: Settings, on_status: Callback | None = None) -> BaseSTT:
    """Создаёт лучший доступный STT-движок."""
    requested = (settings.stt_engine or "auto").lower()
    # В режиме «auto» используем только локальные движки: аудио не уходит в облако.
    # Google Speech выбирается пользователем явно.
    order = (
        ["faster-whisper", "vosk"]
        if requested in {"auto", ""}
        else [requested]
    )
    errors: list[str] = []
    for name in order:
        try:
            if name == "faster-whisper":
                return FasterWhisperSTT(settings, on_status)
            if name == "vosk":
                return VoskSTT(settings, on_status)
            if name == "google":
                return GoogleSTT(settings, on_status)
        except Exception as exc:
            log.warning("STT %s недоступен: %s", name, exc)
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        "Не удалось запустить распознавание речи. Установите faster-whisper или Vosk.\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _latinize(text: str) -> str:
    """Транслитерация в латиницу для сравнения «Jarvis» и «Джарвис»."""
    table = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
        "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k",
        "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
        "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
        "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya",
    }
    return "".join(table.get(char, char) for char in (text or "").lower().replace("дж", "j"))


def _word_similar(word: str, target: str) -> bool:
    if not word or not target:
        return False
    if word == target:
        return True
    # Простые падежи: «джарвис» / «джарвиса» / «джарвису».
    stem = target[:4] if len(target) >= 4 else target
    if len(stem) >= 3 and word.startswith(stem):
        return True
    import difflib

    # Сравниваем и в латинице: Whisper иногда пишет имя как «Jarvis».
    left = _latinize(word)
    right = _latinize(target)
    if left == right:
        return True
    if right[:4] and left.startswith(right[:4]):
        return True
    return difflib.SequenceMatcher(None, left, right).ratio() >= 0.78


def _strip_wake_prefix(text: str, wake: str) -> str:
    """Убирает повторные «Джарвис, джарвис» из начала остатка фразы."""
    words = text.split()
    while words and _word_similar(words[0], wake):
        words.pop(0)
    return " ".join(words).strip(" ,.!?-")


def match_wake_word(text: str, wake_word: str) -> tuple[bool, str]:
    """Ищет имя ассистента в тексте. Возвращает (найдено, остаток фразы)."""
    normalized = normalize_text(text)
    wake = normalize_text(wake_word)
    if not normalized or not wake:
        return False, normalized

    if wake in normalized:
        tail = normalized.split(wake, 1)[1]
        # «Джарвиса, открой...» — отбрасываем падежное окончание имени.
        tail = re.sub(r"^[,\s]*[а-я]{0,2}\b", "", tail, count=1)
        return True, _strip_wake_prefix(tail, wake)

    words = normalized.split()
    for target in wake.split():
        if len(target) < 3:
            continue
        for index, word in enumerate(words):
            if _word_similar(word, target):
                rest = " ".join(words[index + 1 :]).strip()
                return True, _strip_wake_prefix(rest, wake)
    return False, normalized


# Слова-команды, которые завершают сессию («имя, стоп»).
_STOP_WORDS = {
    "стоп", "отбой", "хватит", "замолчи", "тихо", "отмена", "перестань",
    "не надо", "хватит слушать", "стоп слушать", "закончили", "до свидания",
}
# Ответы «да» и «нет» на вопрос «Вы меня вызывали?».
_YES_WORDS = {
    "да", "ага", "угу", "так", "верно", "конечно", "да сэр", "да сэр да",
    "слушаю", "yes", "yeah", "yep", "точно", "именно", "вызывал", "вызывала",
}
_NO_WORDS = {
    "нет", "не", "неа", "no", "nope", "нет сэр", "не вызывал", "не вызывала",
    "ошибся", "ошиблась", "случайно", "забудь", "неважно",
}
# Просто отклик на имя, а не команда.
_FILLERS = {
    "алло", "ау", "эй", "ты здесь", "ты тут", "слышишь меня", "слушаю",
    "слушаю тебя", "да сэр", "привет", "здравствуй", "здравствуйте",
}


def is_stop_command(text: str) -> bool:
    """«Стоп», «отбой», «имя стоп» — команда завершить сессию."""
    normalized = normalize_text(text)
    if not normalized:
        return False
    if normalized in _STOP_WORDS:
        return True
    if normalized.startswith("стоп ") or normalized.startswith("отбой "):
        return True
    # «Карл, стоп»: короткая фраза, заканчивающаяся стоп-словом.
    words = normalized.split()
    return len(words) <= 3 and words[-1] in {"стоп", "отбой"}


def parse_yes_no(text: str) -> bool | None:
    """Разбирает ответ на вопрос «Вы меня вызывали?». None — не расслышали."""
    normalized = normalize_text(text)
    if not normalized:
        return None
    if "не вызывал" in normalized or "не вызывала" in normalized:
        return False
    words = normalized.split()
    if normalized in _YES_WORDS or (words and words[0] in _YES_WORDS):
        return True
    if normalized in _NO_WORDS or (words and words[0] in _NO_WORDS):
        return False
    return None


def is_filler(text: str) -> bool:
    """«Алло», «да», «слушаю» — отклик на имя, а не команда."""
    normalized = normalize_text(text)
    if not normalized:
        return True
    return normalized in _FILLERS or normalized in _YES_WORDS


# ---------------------------------------------------------------------------
# Слушатель: wake word + запись команды
# ---------------------------------------------------------------------------

class VoiceListener(threading.Thread):
    """Фоновый поток: слушает микрофон, ловит имя и записывает команду."""

    def __init__(
        self,
        stt: BaseSTT,
        settings: Settings,
        on_wake: Callback | None = None,
        on_command: Callback | None = None,
        on_status: Callback | None = None,
        on_level: Callback | None = None,
        on_error: Callback | None = None,
        on_heard: Callback | None = None,
        on_barge_in: Callback | None = None,
        on_session_start: Callback | None = None,
        on_session_end: Callback | None = None,
        on_confirm_retry: Callback | None = None,
    ) -> None:
        super().__init__(daemon=True, name="VoiceListener")
        self.stt = stt
        self.settings = settings
        self._on_wake = on_wake or (lambda: None)
        self._on_command = on_command or (lambda _text: None)
        self._on_status = on_status or (lambda _text: None)
        self._on_level = on_level or (lambda _level: None)
        self._on_error = on_error or (lambda _text: None)
        self._on_heard = on_heard or (lambda _text: None)
        self._on_barge_in = on_barge_in or (lambda: None)
        # Аргумент mode: "oneshot" (одна команда), "dialog" (имя+команда)
        # или "listen" (сказали только имя — слушаем до «стоп»).
        self._on_session_start = on_session_start or (lambda _mode="dialog": None)
        self._on_session_end = on_session_end or (lambda _reason: None)
        self._on_confirm_retry = on_confirm_retry or (lambda: None)

        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._force_command = threading.Event()
        # Ассистент говорит: микрофон слушаем только для перебивания.
        self._speaking = threading.Event()
        # Сессия: после имени слушаем всё подряд, без повторного имени.
        self._session = threading.Event()
        self._barge_hits = 0
        self._mic: MicrophoneStream | None = None
        self._gain = max(0.1, float(settings.mic_gain or 1.0))
        self._threshold = max(0.004, settings.silence_threshold)
        # Стартуем с низкого шума: иначе первые фразы «не слышны».
        self._noise = 0.004

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._mic = MicrophoneStream(resolve_input_device(self.settings.input_device))
            self._mic.start()
        except Exception as exc:
            self._on_error(f"Микрофон недоступен: {exc}")
            return

        self._on_status("Слушаю...")
        preroll: deque[np.ndarray] = deque(maxlen=8)
        last_trigger = 0.0

        try:
            while not self._stop_event.is_set():
                chunk = self._mic.read(timeout=0.1)
                if chunk is None:
                    continue
                if self._paused.is_set():
                    self._mic.drain()
                    preroll.clear()
                    continue

                # Усиление микрофона: помогает тихим гарнитурам.
                chunk = np.clip(chunk * self._gain, -1.0, 1.0)
                level = _rms(chunk)
                # Адаптивная оценка шума и порога.
                self._noise = 0.97 * self._noise + 0.03 * level
                self._threshold = min(
                    0.12,
                    max(
                        0.004,
                        float(self.settings.silence_threshold),
                        self._noise * 1.7,
                    ),
                )
                self._on_level(min(1.0, level * 10.0))

                # Ассистент говорит: не слушаем команды, но ждём перебивания.
                if self._speaking.is_set():
                    self._check_barge_in(level)
                    continue

                # Принудительная запись команды (кнопка «Слушать»).
                if self._force_command.is_set():
                    self._force_command.clear()
                    self._on_status("Слушаю команду...")
                    self._beep()
                    self._mic.drain()
                    audio = self._capture(
                        [], max_seconds=20.0, silence=0.75, initial_silence=1.6
                    )
                    self._handle_command(audio)
                    preroll.clear()
                    self._mic.drain()
                    continue

                if level >= self._threshold and (time.time() - last_trigger) > 0.5:
                    last_trigger = time.time()
                    initial = list(preroll) + [chunk]

                    if self._session.is_set():
                        # Режим диалога: любая фраза — команда, имя не нужно.
                        self._on_status("Слушаю команду...")
                        audio = self._capture(
                            initial, max_seconds=20.0, silence=0.75,
                            initial_silence=1.6,
                        )
                        self._handle_session_audio(audio)
                    elif not self.settings.wake_word_enabled:
                        # Wake word отключён — любая речь считается командой.
                        self._on_status("Слушаю команду...")
                        audio = self._capture(
                            initial, max_seconds=20.0, silence=0.75,
                            initial_silence=1.6,
                        )
                        self._handle_command(audio)
                    else:
                        self._on_status("Распознаю...")
                        audio = self._capture(initial, max_seconds=10.0, silence=0.65)
                        text = self._transcribe(
                            audio, prompt=getattr(self.stt, "wake_prompt", None)
                        )
                        if text:
                            self._wake_flow(text)

                    preroll.clear()
                    self._mic.drain()
                else:
                    preroll.append(chunk)
        except Exception as exc:
            log.exception("listener crashed")
            self._on_error(f"Ошибка прослушивания: {exc}")
        finally:
            if self._mic is not None:
                self._mic.stop()

    # ------------------------------------------------------------------
    def _wake_flow(self, text: str) -> None:
        """Обрабатывает фразу с именем ассистента."""
        found, rest = match_wake_word(text, self.settings.assistant_name)
        if not found:
            return

        # «Имя, стоп» вне сессии — просто подтверждаем, что не слушаем.
        if is_stop_command(rest):
            self._on_session_end("stop")
            return

        rest = rest.strip()
        if rest and not is_filler(rest):
            answer = parse_yes_no(rest)
            if answer is False:
                # «Карл, нет» — сразу уходим в ожидание.
                self._on_session_end("declined")
                return
            if self.settings.one_shot_mode:
                # «Карл, найди песню» — одна команда, и снова ждём имя.
                self._session.clear()
                self._on_session_start("oneshot")
            else:
                # «Карл, найди песню» — команда плюс диалог до слова «стоп».
                self._session.set()
                self._on_session_start("dialog")
            self._on_command(rest)
            return

        # Сказано только имя: подтверждаем и слушаем команды без имени,
        # пока не прозвучит «стоп».
        self._session.set()
        self._speaking.set()  # не слушаем собственное подтверждение
        self._on_session_start("listen")
        self._wait_for_speech_end(8.0)

    def _wait_for_speech_end(self, timeout: float = 10.0) -> None:
        """Ждёт, пока ассистент договорит вопрос, чтобы не слушать себя."""
        started = time.time()
        # Сначала даём UI время запустить озвучку.
        while time.time() - started < 1.5 and not self._speaking.is_set():
            time.sleep(0.05)
        while self._speaking.is_set() and time.time() - started < timeout:
            time.sleep(0.05)
        # resume() после речи снимает паузу с небольшой задержкой.
        while self._paused.is_set() and time.time() - started < timeout:
            time.sleep(0.05)
        time.sleep(0.2)

    def _handle_session_audio(self, audio: np.ndarray) -> None:
        """Фраза внутри сессии: команда без имени или «стоп»."""
        text = self._transcribe(
            audio, prompt=getattr(self.stt, "command_prompt", None)
        )
        if not text:
            self._on_status("Слушаю...")
            return

        found, rest = match_wake_word(text, self.settings.assistant_name)
        if found:
            text = rest
        if not text:
            self._on_status("Слушаю...")
            return

        if is_stop_command(text):
            self._session.clear()
            self._on_session_end("stop")
            return

        if is_filler(text):
            # «Алло», «да», «слушаю» — не команда, не отправляем в LLM.
            self._on_status("Слушаю, сэр...")
            return

        self._on_command(text)

    def end_session(self, reason: str = "stop") -> None:
        """Завершает сессию (вызывается из UI по команде «стоп»)."""
        was_active = self._session.is_set()
        self._session.clear()
        self._barge_hits = 0
        if was_active:
            self._on_session_end(reason)

    def cancel_session(self) -> None:
        """Сбрасывает сессию без события: UI сам вернёт режим ожидания имени."""
        self._session.clear()
        self._barge_hits = 0

    # ------------------------------------------------------------------
    def _capture(
        self,
        initial: list[np.ndarray],
        max_seconds: float,
        silence: float,
        initial_silence: float | None = None,
    ) -> np.ndarray:
        """Записывает сегмент до паузы или до max_seconds.

        initial_silence — сколько ждать начала речи (после сигнала «слушаю»),
        silence — сколько тишины считать концом фразы.
        """
        assert self._mic is not None
        frames = list(initial)
        total = sum(len(frame) for frame in frames)
        started = time.time()
        silent_time = 0.0
        heard_speech = False
        wait_limit = initial_silence if initial_silence is not None else silence

        while not self._stop_event.is_set() and not self._paused.is_set():
            if time.time() - started > max_seconds:
                break
            chunk = self._mic.read(timeout=0.2)
            if chunk is None:
                continue
            frames.append(chunk)
            total += len(chunk)
            level = _rms(chunk)
            self._on_level(min(1.0, level * 10.0))
            if level >= self._threshold:
                heard_speech = True
                silent_time = 0.0
            else:
                silent_time += len(chunk) / SAMPLE_RATE
            limit = silence if heard_speech else wait_limit
            if silent_time >= limit and total >= SAMPLE_RATE * 0.3:
                break

        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).astype(np.float32, copy=False)

    def _transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str:
        if len(audio) < SAMPLE_RATE * MIN_COMMAND_SECONDS:
            return ""
        if _rms(audio) < 0.001:
            # Слишком тихо: не отправляем шум в Whisper.
            return ""
        try:
            text = (self.stt.transcribe(audio, initial_prompt=prompt) or "").strip()
        except Exception as exc:
            log.exception("transcribe failed")
            self._on_error(f"Ошибка распознавания: {exc}")
            return ""
        if text:
            # Показываем, что именно услышал микрофон — это очень помогает
            # при настройке wake word и порога тишины.
            self._on_heard(text)
        return text

    def _beep(self) -> None:
        """Короткий сигнал «слушаю» после имени ассистента."""
        if not self.settings.wake_beep:
            return
        # На Windows используем отдельный API: он не конфликтует с потоком
        # воспроизведения TTS (sounddevice).
        if os.name == "nt":
            try:
                import winsound

                winsound.Beep(880, 120)
                return
            except Exception:
                pass
        try:
            import sounddevice as sd

            sample_rate = 44100
            duration = 0.13
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = 0.18 * np.sin(2 * np.pi * 880 * t)
            # Плавные края, чтобы не было щелчка.
            fade = int(sample_rate * 0.015)
            if fade > 0:
                tone[:fade] *= np.linspace(0, 1, fade)
                tone[-fade:] *= np.linspace(1, 0, fade)
            sd.play(tone.astype(np.float32), sample_rate)
            sd.wait()
        except Exception:
            pass

    def _check_barge_in(self, level: float) -> None:
        """Владелец заговорил во время ответа — сообщаем наверх, чтобы замолчать."""
        if not self.settings.barge_in:
            return
        threshold = max(
            float(self.settings.barge_in_threshold or 0.055),
            self._threshold * 2.5,
        )
        if level >= threshold:
            self._barge_hits += 1
            # ~4 блока по 64 мс: короткий кашель или щелчок не считается.
            if self._barge_hits >= 4:
                self._barge_hits = 0
                self._speaking.clear()
                self._on_barge_in()
        else:
            self._barge_hits = max(0, self._barge_hits - 1)

    def set_speaking(self, speaking: bool) -> None:
        """Переключает режим «ассистент говорит» (barge-in вместо паузы)."""
        self._barge_hits = 0
        if speaking:
            self._speaking.set()
            if self._mic is not None:
                self._mic.drain()
        else:
            self._speaking.clear()

    def _handle_command(self, audio: np.ndarray) -> None:
        text = self._transcribe(
            audio, prompt=getattr(self.stt, "command_prompt", None)
        )
        if text:
            # Whisper иногда добавляет имя ассистента в начало команды.
            found, rest = match_wake_word(text, self.settings.assistant_name)
            if found:
                text = rest
            if text:
                self._on_command(text)
            else:
                self._on_status("Слушаю...")
        else:
            self._on_status("Не расслышал, сэр.")

    # ------------------------------------------------------------------
    def pause(self) -> None:
        """Пауза на время речи ассистента, чтобы не слышать себя."""
        self._paused.set()
        if self._mic is not None:
            self._mic.drain()

    def resume(self, delay: float = 0.4) -> None:
        def _resume() -> None:
            time.sleep(delay)
            if self._mic is not None:
                self._mic.drain()
            self._paused.clear()

        threading.Thread(target=_resume, daemon=True).start()

    def request_command(self) -> None:
        """Записать одну команду без wake word."""
        self._force_command.set()

    def stop(self) -> None:
        self._stop_event.set()
        if self._mic is not None:
            self._mic.drain()
        # join() нельзя вызывать для потока, который ещё не стартовал.
        if self.is_alive():
            self.join(timeout=2.0)


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def _pick_voice(engine: Any, language: str, wanted: str) -> Any | None:
    try:
        voices = engine.getProperty("voices") or []
    except Exception:
        return None
    if wanted:
        for voice in voices:
            if wanted.lower() in str(voice.name).lower():
                return voice
    code = (language or "ru")[:2].lower()
    for voice in voices:
        blob = f"{voice.id} {voice.name} {voice.languages}".lower()
        if code in blob:
            return voice
    return voices[0] if voices else None


class BaseTTS:
    label = "TTS"

    def speak(
        self,
        text: str,
        on_start: Callback | None = None,
        on_end: Callback | None = None,
        on_amplitude: Callback | None = None,
        on_error: Callback | None = None,
    ) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        pass

    @property
    def is_speaking(self) -> bool:
        return False


class Pyttsx3TTS(BaseTTS):
    """Системный офлайн-голос. Амплитуда имитируется по словам (SAPI5)."""

    label = "pyttsx3"

    def __init__(self, settings: Settings) -> None:
        import pyttsx3  # noqa: F401  (проверяем доступность)

        self.settings = settings
        self._engine: Any = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._speaking = threading.Event()

    def speak(
        self,
        text: str,
        on_start: Callback | None = None,
        on_end: Callback | None = None,
        on_amplitude: Callback | None = None,
        on_error: Callback | None = None,
    ) -> None:
        with self._lock:
            if self._speaking.is_set():
                self.stop()
            self._stop_event.clear()
            self._speaking.set()
            threading.Thread(
                target=self._run,
                args=(text, on_start, on_end, on_amplitude, on_error),
                daemon=True,
                name="Pyttsx3TTS",
            ).start()

    def _run(
        self,
        text: str,
        on_start: Callback | None,
        on_end: Callback | None,
        on_amplitude: Callback | None,
        on_error: Callback | None = None,
    ) -> None:
        try:
            # pyttsx3 использует COM (SAPI5) на Windows. COM нужно
            # инициализировать в каждом потоке, иначе вылетает
            # Windows fatal exception: code 0x80010108 (RPC_E_DISCONNECTED).
            try:
                import pythoncom

                pythoncom.CoInitialize()
            except Exception:
                pass

            import pyttsx3

            engine = pyttsx3.init()
            self._engine = engine
            engine.setProperty("rate", int(self.settings.tts_rate or 180))
            voice = _pick_voice(engine, self.settings.language, self.settings.tts_voice)
            if voice is not None:
                engine.setProperty("voice", voice.id)

            if on_start:
                on_start()

            def _on_word(_name: str, _location: int, _length: int) -> None:
                if on_amplitude:
                    on_amplitude(0.55 + 0.35 * random.random())

            try:
                engine.connect("started-word", _on_word)
            except Exception:
                pass

            engine.say(text)
            engine.runAndWait()
            if on_end:
                on_end()
        except Exception as exc:
            log.warning("pyttsx3 failed: %s", exc)
            if on_error:
                on_error(exc)
            elif on_end:
                on_end()
        finally:
            self._engine = None
            self._speaking.clear()
            try:
                import pythoncom

                pythoncom.CoUninitialize()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()


class SileroTTS(BaseTTS):
    """Нейросетевой офлайн-голос Silero. Реальная амплитуда из аудио."""

    label = "Silero TTS"

    def __init__(self, settings: Settings, on_status: Callback | None = None) -> None:
        import torch

        self._torch = torch
        self.settings = settings
        self._stop_event = threading.Event()
        self._speaking = threading.Event()

        language = (settings.language or "ru").lower()
        if language.startswith("ru"):
            if on_status:
                on_status("Загружаю Silero TTS (v4_ru)...")
            self._model, _ = torch.hub.load(
                "snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker="v4_ru",
                trust_repo=True,
            )
            self._speaker = settings.silero_speaker or "aidar"
            self._sample_rate = 48000
            self.label = f"Silero TTS (ru, {self._speaker})"
        else:
            if on_status:
                on_status("Загружаю Silero TTS (v3_en)...")
            self._model, _ = torch.hub.load(
                "snakers4/silero-models",
                model="silero_tts",
                language="en",
                speaker="v3_en",
                trust_repo=True,
            )
            self._speaker = settings.silero_speaker or "en_0"
            self._sample_rate = 48000
            self.label = f"Silero TTS (en, {self._speaker})"

    def speak(
        self,
        text: str,
        on_start: Callback | None = None,
        on_end: Callback | None = None,
        on_amplitude: Callback | None = None,
        on_error: Callback | None = None,
    ) -> None:
        if self._speaking.is_set():
            self.stop()
        self._stop_event.clear()
        self._speaking.set()
        threading.Thread(
            target=self._run,
            args=(text, on_start, on_end, on_amplitude, on_error),
            daemon=True,
            name="SileroTTS",
        ).start()

    def _run(
        self,
        text: str,
        on_start: Callback | None,
        on_end: Callback | None,
        on_amplitude: Callback | None,
        on_error: Callback | None = None,
    ) -> None:
        try:
            import sounddevice as sd

            audio = self._model.apply_tts(
                text=text, speaker=self._speaker, sample_rate=self._sample_rate
            )
            array = audio.cpu().numpy().astype(np.float32)
            if on_start:
                on_start()
            sd.play(array, self._sample_rate)
            chunk = max(1, self._sample_rate // 20)  # 50 мс
            for index in range(0, len(array), chunk):
                if self._stop_event.is_set():
                    sd.stop()
                    break
                segment = array[index : index + chunk]
                if on_amplitude:
                    on_amplitude(min(1.0, _rms(segment) * 6.0))
                sd.sleep(int(1000 * len(segment) / self._sample_rate))
            sd.wait()
            if on_end:
                on_end()
        except Exception as exc:
            log.warning("Silero TTS failed: %s", exc)
            if on_error:
                on_error(exc)
            elif on_end:
                on_end()
        finally:
            self._speaking.clear()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()


class ElevenLabsTTS(BaseTTS):
    """Облачный голос ElevenLabs (MP3)."""

    label = "ElevenLabs"

    def __init__(self, settings: Settings) -> None:
        if not (settings.elevenlabs_api_key or "").strip():
            raise RuntimeError("Не задан ключ ElevenLabs")
        if not (settings.elevenlabs_voice_id or "").strip():
            raise RuntimeError("Не задан voice_id ElevenLabs")
        self.settings = settings
        self._stop_event = threading.Event()
        self._speaking = threading.Event()

    def speak(
        self,
        text: str,
        on_start: Callback | None = None,
        on_end: Callback | None = None,
        on_amplitude: Callback | None = None,
        on_error: Callback | None = None,
    ) -> None:
        if self._speaking.is_set():
            self.stop()
        self._stop_event.clear()
        self._speaking.set()
        threading.Thread(
            target=self._run,
            args=(text, on_start, on_end, on_amplitude, on_error),
            daemon=True,
            name="ElevenLabsTTS",
        ).start()

    def _run(
        self,
        text: str,
        on_start: Callback | None,
        on_end: Callback | None,
        on_amplitude: Callback | None,
        on_error: Callback | None = None,
    ) -> None:
        path: Path | None = None
        try:
            import requests

            url = (
                "https://api.elevenlabs.io/v1/text-to-speech/"
                f"{self.settings.elevenlabs_voice_id.strip()}"
            )
            response = requests.post(
                url,
                headers={
                    "xi-api-key": self.settings.elevenlabs_api_key.strip(),
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": self.settings.elevenlabs_model,
                    "voice_settings": {"stability": 0.45, "similarity_boost": 0.8},
                },
                timeout=(15, 120),
            )
            if response.status_code == 401:
                raise RuntimeError(
                    "ElevenLabs: ключу не разрешён text_to_speech. "
                    "Откройте elevenlabs.io → API Keys → Edit и включите "
                    "разрешение Text to Speech."
                )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            # ElevenLabs иногда отдаёт JSON/ошибку вместо mp3 — проверяем,
            # иначе pygame падает с «corrupt mp3 file».
            content = response.content or b""
            if len(content) < 512 or not content.startswith((b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
                snippet = content[:120].decode("utf-8", "ignore")
                raise RuntimeError(f"ElevenLabs вернул не аудио: {snippet}")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
                handle.write(content)
                path = Path(handle.name)
            _play_file(path, self._stop_event, on_start, on_end, on_amplitude)
            return
        except Exception as exc:
            log.warning("ElevenLabs TTS failed: %s", exc)
            if on_error:
                on_error(exc)
            elif on_end:
                on_end()
        finally:
            self._speaking.clear()
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop_event.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass
        try:
            import pygame

            pygame.mixer.music.stop()
        except Exception:
            pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()


def _has_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text or ""))


# ---------------------------------------------------------------------------
# Edge TTS — бесплатный нейроголос Microsoft (онлайн, без ключа)
# ---------------------------------------------------------------------------

class EdgeTTS(BaseTTS):
    """Реалистичный нейроголос через сервис Microsoft Edge.

    Голос выбирается по языку текста: русский текст читает Дмитрий,
    английский — британский Ryan (очень похож на «джарвисовский» тембр).
    Ключ не нужен, но требуется интернет.
    """

    def __init__(self, settings: Settings) -> None:
        import edge_tts  # noqa: F401  (проверяем доступность)

        self.settings = settings
        self._stop_event = threading.Event()
        self._speaking = threading.Event()
        self._ru = (settings.edge_voice_ru or "ru-RU-DmitryNeural").strip()
        self._en = (settings.edge_voice_en or "en-GB-RyanNeural").strip()
        self.label = f"Edge TTS ({self._ru} / {self._en})"

    def _voice_for(self, text: str) -> str:
        return self._ru if _has_cyrillic(text) else self._en

    # Типовые короткие фразы: если синтезировать их заранее, первый ответ
    # звучит мгновенно (кэш озвучки) и не зависит от задержки Edge.
    PREWARM_PHRASES = (
        "Да, сэр?",
        "Слушаю, сэр.",
        "Выполняю, сэр.",
        "Ищу, сэр.",
        "Открываю.",
        "Печатаю.",
        "Делаю снимок.",
        "Готово, сэр.",
        "Отменено, сэр.",
        "Произошла ошибка, сэр.",
        "Не понял команду, сэр.",
        "Системы запущены. Слушаю вас, сэр.",
    )

    def prewarm(self) -> None:
        """Синтезирует типовые фразы в кэш фоном, не мешая разговорам."""
        if not getattr(self.settings, "voice_cache", True):
            return
        if getattr(self.settings, "voice_prewarm", True) is False:
            return

        def worker() -> None:
            import asyncio
            import hashlib

            cache_dir = OFFLINE_AUDIO_DIR / "voice_cache"
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                return
            rate = (self.settings.edge_rate or "").strip()
            pitch = (self.settings.edge_pitch or "").strip()
            for phrase in self.PREWARM_PHRASES:
                # Не лезем в сеть, пока Джарвис что-то говорит.
                if self._speaking.is_set():
                    time.sleep(1.0)
                voice = self._voice_for(phrase)
                key = hashlib.md5(f"{phrase}|{voice}|{rate}|{pitch}".encode()).hexdigest()
                if (cache_dir / f"{key}.mp3").exists():
                    continue
                try:
                    asyncio.run(self._synthesize(phrase, voice, cache_dir / f"{key}.mp3", rate, pitch))
                except Exception:
                    time.sleep(1.0)

        threading.Thread(target=worker, daemon=True, name="EdgePrewarm").start()

    def speak(
        self,
        text: str,
        on_start: Callback | None = None,
        on_end: Callback | None = None,
        on_amplitude: Callback | None = None,
        on_error: Callback | None = None,
    ) -> None:
        if self._speaking.is_set():
            self.stop()
        self._stop_event.clear()
        self._speaking.set()
        threading.Thread(
            target=self._run,
            args=(text, on_start, on_end, on_amplitude, on_error),
            daemon=True,
            name="EdgeTTS",
        ).start()

    async def _synthesize(
        self,
        text: str,
        voice: str,
        path: Path,
        rate: str,
        pitch: str,
    ) -> None:
        import edge_tts
        import hashlib

        # Кэш озвучки: повторяющиеся фразы не дергают Edge, а сразу берут
        # из mp3-файла на диске — ответ мгновенный.
        use_cache = getattr(self.settings, "voice_cache", True)
        if use_cache:
            cache_key = hashlib.md5(
                f"{text}|{voice}|{rate}|{pitch}".encode()
            ).hexdigest()
            cache_dir = OFFLINE_AUDIO_DIR / "voice_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = cache_dir / f"{cache_key}.mp3"
            if cached.exists():
                shutil.copy2(str(cached), str(path))
                return
            try:
                communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
                await communicate.save(str(cached))
            except Exception:
                # Кэш недоступен — просто синтезируем без кэша.
                pass
        else:
            cached = None

        kwargs: dict[str, str] = {}
        if rate:
            kwargs["rate"] = rate
        if pitch:
            kwargs["pitch"] = pitch
        communicate = edge_tts.Communicate(text, voice, **kwargs)
        await communicate.save(str(path))

    def _run(
        self,
        text: str,
        on_start: Callback | None,
        on_end: Callback | None,
        on_amplitude: Callback | None,
        on_error: Callback | None = None,
    ) -> None:
        import asyncio

        path: Path | None = None
        try:
            voice = self._voice_for(text)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
                path = Path(handle.name)

            last_error: Exception | None = None
            # Сервис Edge периодически отвечает пустым аудио — делаем ретраи.
            # Попыток немного и пауза короткая: иначе ответ «оживает» лишь
            # через несколько секунд после того, как текст уже готов.
            for attempt in range(2):
                try:
                    asyncio.run(
                        self._synthesize(
                            text,
                            voice,
                            path,
                            (self.settings.edge_rate or "").strip(),
                            (self.settings.edge_pitch or "").strip(),
                        )
                    )
                    if path.exists() and path.stat().st_size > 1000:
                        last_error = None
                        break
                    raise RuntimeError("Edge TTS вернул пустой аудиофайл")
                except Exception as exc:
                    last_error = exc
                    log.debug("Edge TTS attempt %s failed: %s", attempt + 1, exc)
                    time.sleep(0.25)
            if last_error is not None:
                raise last_error

            # «Голос Джарвиса»: узкополосный фильтр (110–8500 Гц) + лёгкое
            # эхо-повторение, как в фильме. Применяется поверх нейроголоса Edge.
            if getattr(self.settings, "voice_fx", False):
                try:
                    path = self._apply_jarvis_fx(path)
                except Exception:
                    log.debug("jarvis fx failed, using raw audio", exc_info=True)

            _play_file(path, self._stop_event, on_start, on_end, on_amplitude)
        except Exception as exc:
            log.warning("Edge TTS failed: %s", exc)
            if on_error:
                on_error(exc)
            elif on_end:
                on_end()
        finally:
            self._speaking.clear()
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _apply_jarvis_fx(self, path: Path) -> Path:
        """Эффект голоса Джарвиса: полосовой фильтр 110–8500 Гц + два эха.

        Принимает mp3, возвращает путь к wav-файлу с обработанным звуком.
        """
        import numpy as np
        import soundfile as sf

        data, samplerate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        n = len(mono)
        if n == 0:
            return path

        # Полосовой фильтр: оставляем 110–8500 Гц, остальное обнуляем.
        spec = np.fft.rfft(mono)
        hz = np.fft.rfftfreq(n, 1.0 / samplerate)
        spec[(hz < 110) | (hz > 8500)] = 0
        y = np.fft.irfft(spec, n=n)

        # Добавляем тишину в хвост (0.12 с), чтобы эхо не обрезалось.
        tail = int(samplerate * 0.12)
        y = np.concatenate([y, np.zeros(tail, dtype=np.float32)])

        # Два эха: 40 мс × 0.18 и 95 мс × 0.08.
        dry = y.copy()
        for ms, gain in ((40, 0.18), (95, 0.08)):
            delay = int(samplerate * ms / 1000)
            if delay < len(y):
                y[delay:] += gain * dry[:-delay]

        # Нормализуем громкость.
        peak = float(np.abs(y).max()) or 1.0
        y = np.clip(y * (0.9 / peak), -1.0, 1.0).astype(np.float32)

        out = path.with_suffix(".fx.wav")
        sf.write(str(out), y, samplerate)
        return out

    def stop(self) -> None:
        self._stop_event.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass
        try:
            import pygame

            pygame.mixer.music.stop()
        except Exception:
            pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()


# ---------------------------------------------------------------------------
# Piper TTS — нейроголос полностью офлайн
# ---------------------------------------------------------------------------

PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# voice_id -> (путь в репозитории, язык)
PIPER_VOICES: dict[str, tuple[str, str]] = {
    "ru_RU-dmitri-medium": ("ru/ru_RU/dmitri/medium", "ru"),
    "ru_RU-irina-medium": ("ru/ru_RU/irina/medium", "ru"),
    "en_GB-alan-medium": ("en/en_GB/alan/medium", "en"),
    "en_GB-northern_english_male-medium": (
        "en/en_GB/northern_english_male/medium", "en"
    ),
    "en_US-lessac-medium": ("en/en_US/lessac/medium", "en"),
}


def _download_file(
    url: str, target: Path, on_status: Callback | None = None,
) -> None:
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    if on_status:
        on_status(f"Скачиваю голос Piper: {target.name} (~60 МБ)...")
    with requests.get(url, stream=True, timeout=(15, 300)) as response:
        response.raise_for_status()
        tmp = target.with_suffix(target.suffix + ".part")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
        tmp.replace(target)


def _piper_paths(
    voice: str,
    on_status: Callback | None = None,
) -> tuple[Path, Path]:
    if voice not in PIPER_VOICES:
        raise RuntimeError(f"Неизвестный голос Piper: {voice}")
    folder, _lang = PIPER_VOICES[voice]
    model = ASSETS_DIR / "piper" / f"{voice}.onnx"
    config = ASSETS_DIR / "piper" / f"{voice}.onnx.json"
    if not model.exists():
        _download_file(f"{PIPER_BASE_URL}/{folder}/{voice}.onnx", model, on_status)
    if not config.exists():
        _download_file(
            f"{PIPER_BASE_URL}/{folder}/{voice}.onnx.json", config, on_status,
        )
    return model, config


class PiperTTS(BaseTTS):
    """Быстрый нейроголос на ONNX Runtime, работает без интернета."""

    def __init__(self, settings: Settings, on_status: Callback | None = None) -> None:
        import piper  # noqa: F401  (проверяем доступность)

        self.settings = settings
        self._on_status = on_status
        self._stop_event = threading.Event()
        self._speaking = threading.Event()
        self._voices: dict[str, Any] = {}
        self._ru = (settings.piper_voice_ru or "ru_RU-dmitri-medium").strip()
        self._en = (settings.piper_voice_en or "en_GB-alan-medium").strip()
        self._length_scale = float(settings.piper_length_scale or 1.08)
        self.label = f"Piper ({self._ru} / {self._en})"

    def _voice_for(self, text: str) -> Any:
        from piper import PiperVoice

        name = self._ru if _has_cyrillic(text) else self._en
        voice = self._voices.get(name)
        if voice is None:
            model, config = _piper_paths(name, self._on_status)
            if self._on_status:
                self._on_status(f"Загружаю голос Piper: {name}...")
            voice = PiperVoice.load(str(model), str(config))
            self._voices[name] = voice
        return voice

    def speak(
        self,
        text: str,
        on_start: Callback | None = None,
        on_end: Callback | None = None,
        on_amplitude: Callback | None = None,
        on_error: Callback | None = None,
    ) -> None:
        if self._speaking.is_set():
            self.stop()
        self._stop_event.clear()
        self._speaking.set()
        threading.Thread(
            target=self._run,
            args=(text, on_start, on_end, on_amplitude, on_error),
            daemon=True,
            name="PiperTTS",
        ).start()

    def _run(
        self,
        text: str,
        on_start: Callback | None,
        on_end: Callback | None,
        on_amplitude: Callback | None,
        on_error: Callback | None = None,
    ) -> None:
        try:
            import sounddevice as sd
            from piper import SynthesisConfig

            voice = self._voice_for(text)
            chunks = list(
                voice.synthesize(
                    text,
                    SynthesisConfig(
                        length_scale=self._length_scale,
                        volume=0.95,
                    ),
                )
            )
            if not chunks:
                raise RuntimeError("Piper не вернул аудио")
            sample_rate = chunks[0].sample_rate
            array = np.concatenate([chunk.audio_float_array for chunk in chunks])
            if on_start:
                on_start()
            sd.play(array, sample_rate)
            step = max(1, sample_rate // 20)
            for index in range(0, len(array), step):
                if self._stop_event.is_set():
                    sd.stop()
                    break
                segment = array[index : index + step]
                if on_amplitude:
                    on_amplitude(min(1.0, _rms(segment) * 5.0))
                sd.sleep(int(1000 * len(segment) / sample_rate))
            sd.wait()
            if on_end:
                on_end()
        except Exception as exc:
            log.warning("Piper TTS failed: %s", exc)
            if on_error:
                on_error(exc)
            elif on_end:
                on_end()
        finally:
            self._speaking.clear()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()


# ---------------------------------------------------------------------------
# ElevenLabs: список голосов для окна настроек
# ---------------------------------------------------------------------------

def fetch_elevenlabs_voices(api_key: str) -> list[dict[str, str]]:
    """Возвращает доступные голоса ElevenLabs: [{'id','name','labels'}]."""
    import requests

    key = (api_key or "").strip()
    if not key:
        raise RuntimeError("Сначала введите API-ключ ElevenLabs.")
    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": key},
            timeout=(10, 40),
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Нет соединения с ElevenLabs: {exc}") from exc
    if response.status_code == 401:
        raise RuntimeError(
            "ElevenLabs отклонил ключ. Проверьте ключ и разрешение Voices."
        )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
    data = response.json()
    result: list[dict[str, str]] = []
    for voice in data.get("voices", []):
        labels = voice.get("labels") or {}
        label_parts = [
            str(labels.get(key, ""))
            for key in ("accent", "gender", "description", "use_case")
            if labels.get(key)
        ]
        result.append(
            {
                "id": str(voice.get("voice_id", "")),
                "name": str(voice.get("name", "Без имени")),
                "labels": ", ".join(label_parts),
            }
        )
    return result


def _play_file(
    path: Path,
    stop_event: threading.Event,
    on_start: Callback | None = None,
    on_end: Callback | None = None,
    on_amplitude: Callback | None = None,
) -> None:
    """Проигрывает wav/mp3-файл и отдаёт реальную амплитуду (если возможно)."""
    try:
        try:
            import soundfile as sf
            import sounddevice as sd

            data, samplerate = sf.read(str(path), dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            if len(mono) == 0:
                raise RuntimeError("пустой аудиофайл")
            # on_start — только когда звук реально готов к воспроизведению.
            if on_start:
                on_start()
            sd.play(mono, samplerate)
            chunk = max(1, samplerate // 20)
            for index in range(0, len(mono), chunk):
                if stop_event.is_set():
                    sd.stop()
                    break
                segment = mono[index : index + chunk]
                if on_amplitude:
                    on_amplitude(min(1.0, _rms(segment) * 6.0))
                sd.sleep(int(1000 * len(segment) / samplerate))
            sd.wait()
        except Exception as exc:
            log.debug("soundfile playback failed (%s), using pygame", exc)
            if on_start:
                on_start()
            _play_file_pygame(path, stop_event, on_amplitude)
    except Exception as exc:
        # Плохой mp3/нет аудиоустройства: не роняем поток, но и не зависаем.
        log.warning("Не удалось воспроизвести %s: %s", path, exc)
    finally:
        if on_end:
            on_end()


def _play_file_pygame(
    path: Path, stop_event: threading.Event, on_amplitude: Callback | None
) -> None:
    import pygame

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if stop_event.is_set():
                pygame.mixer.music.stop()
                break
            if on_amplitude:
                on_amplitude(0.35 + 0.25 * random.random())
            time.sleep(0.05)
    finally:
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass


class TTSManager:
    """Выбирает движок озвучки, поддерживает fallback и офлайн-аудиофайлы.

    Порядок по умолчанию: Edge (онлайн, самый живой) → Piper (офлайн,
    нейросеть) → pyttsx3 (системный). Если движок падает во время работы
    (например, ElevenLabs вернул 401), менеджер на лету переключается
    на следующий и повторяет фразу.
    """

    def __init__(self, settings: Settings, on_status: Callback | None = None) -> None:
        self.settings = settings
        self._on_status = on_status or (lambda _text: None)
        self._engine: BaseTTS | None = None
        self._engine_name = ""
        self._order: list[str] = []
        self._cache: dict[str, BaseTTS] = {}
        self._disabled: set[str] = set()
        self._file_thread: threading.Thread | None = None
        self._file_stop = threading.Event()
        self._build()

    # ------------------------------------------------------------------
    def _create(self, name: str) -> BaseTTS | None:
        if name in self._cache:
            return self._cache[name]
        if name in self._disabled:
            return None
        try:
            if name == "edge":
                engine: BaseTTS = EdgeTTS(self.settings)
            elif name == "piper":
                engine = PiperTTS(self.settings, on_status=self._on_status)
            elif name == "elevenlabs":
                engine = ElevenLabsTTS(self.settings)
            elif name == "silero":
                engine = SileroTTS(self.settings, on_status=self._on_status)
            elif name == "pyttsx3":
                engine = Pyttsx3TTS(self.settings)
            else:
                return None
        except Exception as exc:
            log.warning("TTS %s init failed: %s", name, exc)
            self._disabled.add(name)
            return None
        self._cache[name] = engine
        return engine

    def _build(self) -> None:
        requested = (self.settings.tts_engine or "edge").lower()
        self._order = [requested] + [
            name
            for name in ("edge", "piper", "pyttsx3", "elevenlabs", "silero")
            if name != requested
        ]
        engine, name = self._pick(set())
        if engine is None:
            self._engine = None
            self._engine_name = ""
            self._on_status("Озвучка недоступна. Установите edge-tts или piper-tts.")
            return
        self._engine = engine
        self._engine_name = name
        if name != requested:
            self._on_status(f"Озвучка: используется {engine.label}")

    def _pick(self, tried: set[str]) -> tuple[BaseTTS | None, str]:
        for name in self._order:
            if name in tried:
                continue
            engine = self._create(name)
            if engine is not None:
                return engine, name
        return None, ""

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._engine is not None

    @property
    def label(self) -> str:
        return self._engine.label if self._engine else "нет"

    @property
    def is_speaking(self) -> bool:
        if self._engine is not None and self._engine.is_speaking:
            return True
        return bool(self._file_thread and self._file_thread.is_alive())

    # ------------------------------------------------------------------
    def prewarm(self) -> None:
        """Прогревает кэш озвучки, если текущий движок это умеет."""
        engine = self._engine
        if engine is None:
            return
        prewarm = getattr(engine, "prewarm", None)
        if callable(prewarm):
            try:
                prewarm()
            except Exception:
                log.debug("TTS prewarm failed", exc_info=True)

    # ------------------------------------------------------------------
    def speak(
        self,
        text: str,
        on_start: Callback | None = None,
        on_end: Callback | None = None,
        on_amplitude: Callback | None = None,
    ) -> bool:
        text = (text or "").strip()
        if not text:
            if on_end:
                on_end()
            return False
        self._file_stop.set()

        tried: set[str] = set()

        def attempt() -> None:
            engine, name = self._pick(tried)
            if engine is None:
                if on_end:
                    on_end()
                return
            tried.add(name)
            self._engine = engine
            self._engine_name = name

            def handle_error(exc: Exception) -> None:
                log.warning("TTS %s failed at runtime: %s", name, exc)
                self._on_status(f"Озвучка {name}: {exc}")
                # Движок отключаем до конца сеанса, чтобы не ждать его снова.
                self._disabled.add(name)
                attempt()

            try:
                engine.speak(text, on_start, on_end, on_amplitude, handle_error)
            except Exception as exc:
                log.exception("TTS speak failed")
                handle_error(exc)

        attempt()
        return True

    def speak_offline(
        self,
        command_id: str,
        text: str,
        on_start: Callback | None = None,
        on_end: Callback | None = None,
        on_amplitude: Callback | None = None,
    ) -> bool:
        """Озвучивает офлайн-команду: сначала ищет assets/offline/<id>.(wav|mp3)."""
        for extension in (".wav", ".mp3", ".ogg", ".flac"):
            candidate = OFFLINE_AUDIO_DIR / f"{command_id}{extension}"
            if candidate.exists():
                self.stop()
                self._file_stop.clear()
                self._file_thread = threading.Thread(
                    target=_play_file,
                    args=(candidate, self._file_stop, on_start, on_end, on_amplitude),
                    daemon=True,
                    name="OfflineAudio",
                )
                self._file_thread.start()
                return True
        return self.speak(text, on_start, on_end, on_amplitude)

    def stop(self) -> None:
        self._file_stop.set()
        if self._engine is not None:
            self._engine.stop()
