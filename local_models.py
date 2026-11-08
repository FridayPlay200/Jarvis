"""Локальные модели: определение железа, каталог GGUF с Hugging Face,
многопоточная загрузка и запуск собственного сервера llama.cpp.

Модуль не требует сторонних приложений: скачанный llama-server сам поднимает
OpenAI-совместимый API, и Джарвис работает с моделью целиком офлайн.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co"
Progress = Callable[[str, float, str], None]   # (статус, 0..1, подробности)


# ---------------------------------------------------------------------------
# Железо
# ---------------------------------------------------------------------------

@dataclass
class Hardware:
    gpu_name: str = ""
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    cpu_cores: int = 0
    cuda_available: bool = False

    @property
    def label(self) -> str:
        parts = []
        if self.gpu_name:
            parts.append(f"{self.gpu_name} ({self.vram_gb:.0f} ГБ VRAM)")
        else:
            parts.append("встроенная графика / без CUDA")
        parts.append(f"RAM {self.ram_gb:.0f} ГБ")
        parts.append(f"CPU {self.cpu_cores} ядер")
        return " · ".join(parts)


def _nvidia_smi() -> tuple[str, float]:
    """(имя GPU, ГБ видеопамяти) через nvidia-smi. Пустая строка, если нет."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "", 0.0
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip()
    except Exception as exc:
        log.debug("nvidia-smi failed: %s", exc)
        return "", 0.0
    if not out:
        return "", 0.0
    # Несколько карт: берём самую старшую по памяти.
    best: tuple[str, float] = ("", 0.0)
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name, _, mem_mib = parts[0], parts[1], parts[1]
        try:
            mem_gb = float(mem_mib) / 1024.0
        except ValueError:
            continue
        if mem_gb > best[1]:
            best = (name, mem_gb)
    return best


def detect_hardware() -> Hardware:
    hw = Hardware()
    hw.gpu_name, hw.vram_gb = _nvidia_smi()
    hw.cuda_available = hw.vram_gb >= 4.0
    try:
        import psutil

        hw.ram_gb = psutil.virtual_memory().total / 1024**3
        hw.cpu_cores = os.cpu_count() or 4
    except Exception:
        hw.ram_gb = 8.0
        hw.cpu_cores = os.cpu_count() or 4
    return hw


# ---------------------------------------------------------------------------
# Каталог моделей
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Описание GGUF-модели для загрузки."""

    key: str
    title: str
    repo: str
    # Имя файла в репозитории. Если пусто — подбирается по quant.
    filename: str = ""
    quant: str = "Q4_K_M"
    size_gb: float = 0.0
    uncensored: bool = False
    vision: bool = False
    coding: bool = False
    note: str = ""

    def fit(self, hw: Hardware) -> tuple[bool, str]:
        """Подходит ли модель под это железо и почему."""
        need_ram = self.size_gb * 1.15 + 2.0
        if self.size_gb <= 0:
            return True, ""
        if hw.ram_gb < need_ram:
            return False, f"нужно ~{need_ram:.0f} ГБ RAM, есть {hw.ram_gb:.0f} ГБ"
        if hw.cuda_available and self.size_gb <= hw.vram_gb - 1.5:
            return True, "вся модель в видеопамяти — самый быстрый вариант"
        if not hw.cuda_available and self.size_gb > hw.ram_gb * 0.55:
            return False, "без дискретной видеокарты будет слишком медленно"
        if hw.cuda_available:
            return True, "часть слоёв в RAM — будет работать, но медленнее"
        return True, "только CPU — отвечает медленно"


CATALOG: list[ModelSpec] = [
    ModelSpec(
        key="qwen05", title="Qwen2.5 0.5B · Q4", repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf", size_gb=0.4,
        note="Сверхбыстрая, годится для простых команд и проверки связи.",
    ),
    ModelSpec(
        key="qwen15", title="Qwen2.5 1.5B · Q4_K_M", repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf", size_gb=1.0,
        note="Быстрая, понимает команды, тянет русский. Лучший выбор для CPU.",
    ),
    ModelSpec(
        key="qwen3b", title="Qwen2.5 3B · Q4_K_M", repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf", size_gb=1.9,
        note="Разумный баланс скорости и качества.",
    ),
    ModelSpec(
        key="qwen7b", title="Qwen2.5 7B · Q4_K_M", repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename="qwen2.5-7b-instruct-q4_k_m.gguf", size_gb=4.4,
        note="Уверенно работает с инструментами. Нужна видеокарта от 6 ГБ.",
    ),
    ModelSpec(
        key="qwen3_4b", title="Qwen3 4B (2507) · Q4_K_M", repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        quant="Q4_K_M", size_gb=2.5,
        note="Новое поколение, умеет рассуждать. Файл подбирается автоматически.",
    ),
    ModelSpec(
        key="llama32_3b", title="Llama 3.2 3B · Q4_K_M", repo="unsloth/Llama-3.2-3B-Instruct-GGUF",
        quant="Q4_K_M", size_gb=1.9,
        note="Отличный английский, русский слабее Qwen.",
    ),
    ModelSpec(
        key="llama31_8b", title="Llama 3.1 8B · Q4_K_M", repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", size_gb=4.9,
        note="Классика для агентских задач, нужна видеокарта от 6 ГБ.",
    ),
    ModelSpec(
        key="gemma4b", title="Gemma 3 4B · Q4_K_M", repo="unsloth/gemma-3-4b-it-GGUF",
        quant="Q4_K_M", size_gb=2.1, vision=True,
        note="Мультимодальная (может смотреть на снимок экрана).",
    ),
    ModelSpec(
        key="qwen_coder", title="Qwen2.5 Coder 7B · Q4_K_M",
        repo="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf", size_gb=4.5, coding=True,
        note="Специальная модель для кодинга — её удобно ставить кодинг-моделью.",
    ),
    ModelSpec(
        key="dolphin3b", title="Dolphin 3 Llama 3.2 3B · Q4_K_M (без цензуры)",
        repo="bartowski/Dolphin3-Llama3.2-3B-GGUF", quant="Q4_K_M", size_gb=2.0,
        uncensored=True, note="Не отказывает по темам, выполняет любые команды.",
    ),
    ModelSpec(
        key="qwen_abliterated", title="Qwen3 8B Abliterated · Q4_K_M (без цензуры)",
        repo="mradermacher/Qwen3-8B-Abliterated-GGUF", quant="Q4_K_M", size_gb=4.7,
        uncensored=True, note="Снятый отказ,Good русский, нужна видеокарта от 6 ГБ.",
    ),
    ModelSpec(
        key="llama32_uncensored", title="Llama 3.2 3B Uncensored · Q4_K_M (без цензуры)",
        repo="mradermacher/Llama-3.2-3B-Instruct-Uncensored-GGUF",
        filename="Llama-3.2-3B-Instruct-uncensored.Q4_K_M.gguf", size_gb=2.0,
        uncensored=True, note="Компактная модель без фильтров.",
    ),
]


def catalog_for(hw: Hardware, show_all: bool = False, only_uncensored: bool = False) -> list[tuple[ModelSpec, bool, str]]:
    """Модели + признак «подходит железу» с пояснением."""
    out = []
    for spec in CATALOG:
        if only_uncensored and not spec.uncensored:
            continue
        fits, reason = spec.fit(hw)
        if not show_all and not fits:
            continue
        out.append((spec, fits, reason))
    return out


# ---------------------------------------------------------------------------
# Hugging Face: список файлов и подбор кванта
# ---------------------------------------------------------------------------

def list_gguf_files(repo: str) -> list[tuple[str, float]]:
    """[(имя файла, размер в ГБ)] всех gguf в репозитории."""
    url = f"{HF_BASE}/api/models/{repo.strip().strip('/')}/tree/main"
    try:
        response = requests.get(url, timeout=(10, 30))
        if response.status_code != 200:
            return []
        data = response.json()
    except (requests.RequestException, ValueError):
        return []
    files: list[tuple[str, float]] = []
    for item in data if isinstance(data, list) else []:
        name = str(item.get("path", ""))
        if not name.lower().endswith(".gguf"):
            continue
        size = float(item.get("size") or 0) / 1024**3
        files.append((name, size))
    files.sort(key=lambda pair: pair[1])
    return files


def pick_filename(repo: str, spec: ModelSpec) -> str:
    """Точное имя файла: из каталога, иначе подбор по кванту."""
    if spec.filename:
        return spec.filename
    files = list_gguf_files(repo)
    if not files:
        return ""
    quant = (spec.quant or "Q4_K_M").upper()
    for name, _size in files:
        if quant in name.upper():
            return name
    # Ближайший по размеру к ожидаемому.
    if spec.size_gb:
        best = min(files, key=lambda pair: abs(pair[1] - spec.size_gb))
        return best[0]
    return files[len(files) // 2][0]


# ---------------------------------------------------------------------------
# Быстрая загрузка: несколько потоков + докачка
# ---------------------------------------------------------------------------

_CHUNK = 8 * 1024 * 1024


def _remote_size(url: str) -> int:
    try:
        head = requests.head(url, allow_redirects=True, timeout=(10, 30))
        if head.status_code in (200, 206):
            return int(head.headers.get("Content-Length") or 0)
    except requests.RequestException:
        pass
    return 0


def _fetch_chunk(
    url: str,
    target: Path,
    start: int,
    end: int,
    done: dict[str, int],
    lock: threading.Lock,
    session: requests.Session,
) -> None:
    headers = {"Range": f"bytes={start}-{end}"}
    with session.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(f"Hugging Face вернул HTTP {response.status_code}")
        position = start
        with open(target, "r+b") as handle:
            handle.seek(position)
            for block in response.iter_content(chunk_size=_CHUNK):
                if not block:
                    continue
                handle.write(block)
                position += len(block)
                with lock:
                    done[url] = done.get(url, 0) + len(block)


class DownloadCancelled(Exception):
    pass


def download_file(
    url: str,
    target: Path,
    threads: int = 8,
    on_progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Многопоточная докачка range-кусками. Умеет продолжать прерванную."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    size = _remote_size(url)

    def report(text: str, ratio: float, details: str = "") -> None:
        if on_progress:
            on_progress(text, max(0.0, min(1.0, ratio)), details)

    if target.exists() and size and target.stat().st_size == size:
        report("Уже скачано", 1.0, target.name)
        return target

    if not size:
        # Сервер не отдал длину — качаем одним потоком.
        with requests.get(url, stream=True, timeout=(15, 300)) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            got = 0
            with open(part, "wb") as handle:
                for block in response.iter_content(chunk_size=_CHUNK):
                    if cancel is not None and cancel.is_set():
                        raise DownloadCancelled("отменено")
                    handle.write(block)
                    got += len(block)
                    report("Загрузка", got / total if total else 0.0, f"{got/1e6:.0f} МБ")
        part.replace(target)
        return target

    # Готовые куски из предыдущей попытки.
    marks_path = part.with_suffix(part.suffix + ".marks")
    marks: dict[str, int] = {}
    if marks_path.exists():
        try:
            marks = json.loads(marks_path.read_text(encoding="utf-8"))
        except ValueError:
            marks = {}

    if not part.exists() or part.stat().st_size != size:
        with open(part, "wb") as handle:
            handle.truncate(size)
        marks = {}

    bounds: list[tuple[int, int]] = []
    step = max(_CHUNK, size // max(threads * 4, 1))
    start = 0
    while start < size:
        end = min(start + step - 1, size - 1)
        bounds.append((start, end))
        start = end + 1

    done_bytes = {"n": sum(marks.values())}
    lock = threading.Lock()
    started = time.time()
    errors: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futures = {}
        for chunk_start, chunk_end in bounds:
            key = f"{chunk_start}-{chunk_end}"
            if marks.get(key, 0) >= chunk_end - chunk_start + 1:
                continue
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled("отменено")
            futures[
                pool.submit(
                    _fetch_chunk, url, part, chunk_start, chunk_end, marks, lock,
                    requests.Session(),
                )
            ] = key
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                future.result()
            except DownloadCancelled:
                raise
            except Exception as exc:
                errors.append(f"{key}: {exc}")
                continue
            chunk_start, chunk_end = (int(v) for v in key.split("-"))
            length = chunk_end - chunk_start + 1
            marks[key] = length
            with lock:
                done_bytes["n"] += length
                fraction = done_bytes["n"] / size
                speed = done_bytes["n"] / max(time.time() - started, 0.1) / 1e6
            loaded_mb = done_bytes["n"] / 1e6
            total_mb = size / 1e6
            report(
                "Загрузка", fraction,
                f"{loaded_mb:.0f} / {total_mb:.0f} МБ · {speed:.1f} МБ/с",
            )
            marks_path.write_text(json.dumps(marks), encoding="utf-8")

    if errors:
        raise RuntimeError(
            "Загрузка прервалась: " + "; ".join(errors[:2]) +
            ". Повторите — файл продолжится с того места, где встал."
        )

    marks_path.unlink(missing_ok=True)
    if part.stat().st_size != size:
        raise RuntimeError("файл докачался неполным — повторите загрузку")
    part.replace(target)
    report("Готово", 1.0, target.name)
    return target


def model_url(repo: str, filename: str) -> str:
    return f"{HF_BASE}/{repo.strip().strip('/')}/resolve/main/{filename}"


def download_model(
    spec: ModelSpec,
    directory: Path,
    threads: int = 8,
    on_progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    filename = pick_filename(spec.repo, spec)
    if not filename:
        raise RuntimeError(f"В репозитории {spec.repo} нет gguf-файла")
    target = directory / filename
    return download_file(model_url(spec.repo, filename), target, threads, on_progress, cancel)


def download_repo_file(
    repo: str,
    filename: str,
    directory: Path,
    threads: int = 8,
    on_progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Загрузка любого файла с Hugging Face — в том числе чужих репозиториев."""
    return download_file(model_url(repo, filename), directory / filename, threads, on_progress, cancel)


# ---------------------------------------------------------------------------
# llama.cpp: скачивание сервера и запуск
# ---------------------------------------------------------------------------

def runtime_dir() -> Path:
    """Папка рантайма рядом с приложением (в exe — внутри dist)."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).resolve().parent
    return base / "runtime" / "llama.cpp"


def server_binary(hw: Hardware | None = None) -> Path | None:
    directory = runtime_dir()
    for name in ("llama-server.exe", "llama-server"):
        found = directory / name
        if found.exists():
            return found
    return None


def _github_latest_release() -> tuple[str, dict[str, str]]:
    """(tag, {имя ассета: url}) последнего релиза llama.cpp."""
    response = requests.get(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases",
        params={"per_page": 10},
        headers={"Accept": "application/vnd.github+json"},
        timeout=(10, 30),
    )
    response.raise_for_status()
    for release in response.json():
        assets = {
            str(a.get("name", "")): str(a.get("browser_download_url", ""))
            for a in release.get("assets", [])
        }
        wanted = [name for name in assets if name.startswith("llama-") and "bin-win" in name]
        if wanted:
            return str(release.get("tag_name", "")), assets
    raise RuntimeError("на GitHub не нашлось Windows-сборки llama.cpp")


def ensure_runtime(
    hw: Hardware,
    on_progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Скачивает бинарник llama-server (CUDA, если есть карта, иначе CPU)."""
    existing = server_binary(hw)
    if existing is not None:
        return existing

    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    tag, assets = _github_latest_release()
    if hw.cuda_available:
        candidates = [name for name in assets if "bin-win-cuda" in name and "x64" in name]
    else:
        candidates = [name for name in assets if "bin-win-cpu-x64" in name]
    if not candidates:
        candidates = [name for name in assets if "bin-win-cpu-x64" in name]
    if not candidates:
        raise RuntimeError(f"в релизе {tag} нет подходящего архива для Windows")
    asset = sorted(candidates)[-1]
    url = assets[asset]

    archive = directory / asset
    download_file(url, archive, threads=6, on_progress=on_progress, cancel=cancel)

    if on_progress:
        on_progress("Распаковка", 0.95, asset)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(directory)
    archive.unlink(missing_ok=True)

    exe = server_binary(hw)
    if exe is None:
        raise RuntimeError("после распаковки не найден llama-server.exe")
    return exe


class LocalServer:
    """Запущенный llama-server: OpenAI-совместимый API на localhost."""

    def __init__(self, port: int = 8077) -> None:
        self.port = port
        self.process: subprocess.Popen | None = None
        self.model_path: str = ""
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def running(self) -> bool:
        if self.process is None:
            return False
        if self.process.poll() is not None:
            return False
        try:
            return requests.get(f"{self.base_url}/models", timeout=(2, 4)).status_code == 200
        except requests.RequestException:
            return False

    def start(
        self,
        model_path: str | Path,
        hw: Hardware | None = None,
        ctx: int = 4096,
        gpu_layers: int | None = None,
        on_progress: Progress | None = None,
    ) -> None:
        hw = hw or detect_hardware()
        exe = ensure_runtime(hw, on_progress)
        with self._lock:
            self.stop()
            path = Path(model_path)
            if not path.exists():
                raise RuntimeError(f"модель не найдена: {path}")
            if gpu_layers is None:
                model_gb = path.stat().st_size / 1e9
                fits_vram = hw.cuda_available and model_gb <= hw.vram_gb - 1.2
                gpu_layers = 99 if fits_vram else (20 if hw.cuda_available else 0)
            args = [
                str(exe), "-m", str(path),
                "--host", "127.0.0.1", "--port", str(self.port),
                "--ctx-size", str(ctx),
                "--n-gpu-layers", str(gpu_layers),
                "--threads", str(max(4, hw.cpu_cores // 2)),
                "--no-webui",
            ]
            if on_progress:
                on_progress("Запуск модели", 0.98, path.name)
            self.process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.model_path = str(path)
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "локальный сервер завершился с ошибкой — модель не подошла "
                    "или файл повреждён"
                )
            if self.running():
                if on_progress:
                    on_progress("Модель готова", 1.0, path.name)
                return
            time.sleep(1.0)
        raise RuntimeError("модель не успела загрузиться за 3 минуты")

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()

    def loaded_model(self) -> str:
        try:
            data = requests.get(f"{self.base_url}/models", timeout=(2, 6)).json()
            models = data.get("data") or []
            return str(models[0].get("id", "")) if models else ""
        except requests.RequestException:
            return ""


_server: LocalServer | None = None


def get_server() -> LocalServer:
    global _server
    if _server is None:
        _server = LocalServer()
    return _server


# ---------------------------------------------------------------------------
# Помощь для Ollama и LM Studio (своими руками)
# ---------------------------------------------------------------------------

def import_to_ollama(model_path: Path, name: str) -> str:
    """Регистрирует GGUF в Ollama через Modelfile. Возвращает имя модели."""
    exe = shutil.which("ollama")
    if not exe:
        raise RuntimeError("Ollama не установлена — скачайте с ollama.com/download")
    safe = re.sub(r"[^a-zA-Z0-9._-]", "-", name).strip("-").lower() or "jarvis-model"
    modelfile = model_path.with_suffix(".modelfile")
    modelfile.write_text(f'FROM "{model_path}"\n', encoding="utf-8")
    result = subprocess.run(
        [exe, "create", safe, "-f", str(modelfile)],
        capture_output=True, text=True, timeout=600,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    modelfile.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout)[:400])
    return safe


OLLAMA_HOWTO = """Как скачать модель самому через Ollama

1. Установка
   • Откройте https://ollama.com/download и скачайте Ollama Windows (Setup).
   • Запустите установщик, дождитесь завершения — Ollama работает фоном,
     иконки в трее у неё нет, это нормально.
   • Проверка: откройте «Командная строка» (Win+R → cmd) и введите
         ollama --version
     Если пишет версию — всё установлено.

2. Скачать модель (одна команда, докачка поддерживается)
         ollama pull qwen2.5:3b          — 1.9 ГБ, быстрый вариант, CPU
         ollama pull qwen2.5:7b          — 4.4 ГБ, нужна видеокарта от 6 ГБ
         ollama pull llama3.2:3b         — 2.0 ГБ
         ollama pull dolphin3:8b         — без цензуры
     Список всех доступных моделей: https://ollama.com/search

3. Посмотреть, что уже скачано
         ollama list

4. Проверить, что модель отвечает
         ollama run qwen2.5:3b "Привет, кто ты?"

5. Подключить в Джарвисе
   • Настройки → «ИИ-модель»: Провайдер = «Ollama (локально)».
   • Модель = то, что показывает `ollama list` (например qwen2.5:3b).
   • Base URL оставляем http://localhost:11434/v1 — он уже прописан.
   • Нажмите «Проверить подключение» — Карл ответит, что связь есть.

6. Если не работает
   • Команда не найдена → перезапустите компьютер после установки.
   • Ошибка соединения → запустите сервер вручную:
         ollama serve
     и оставьте окно открытым.
   • Порт занят → поменяйте порт переменной OLLAMA_HOST и Base URL.

7. Удалить модель
         ollama rm qwen2.5:3b"""

LMSTUDIO_HOWTO = """Как скачать модель самому через LM Studio

1. Установка
   • Откройте https://lmstudio.ai и нажмите Download (Windows, ~1 ГБ).
   • Установите и запустите. При первом запуске можно пропустить аккаунт.

2. Скачать модель внутри программы
   • Слева нажмите иконку лупы (Search).
   • Введите имя, например:
         qwen 2.5 3b instruct        — быстрый, CPU
         qwen 2.5 7b instruct        — нужен GPU от 6 ГБ
         dolphin 3                   — без цензуры
   • В правой колонке выберите квант:
         Q4_K_M — разумный баланс (рекомендую),
         Q5_K_M / Q6_K — точнее, но тяжелее,
         IQ2 / Q2 — только если памяти мало.
   • Нажмите Download у нужного файла. Прогресс — внизу.

   Как понять, какой квант потянет ваша машина:
     • модель целиком помещается в видеопамять → ответ мгновенный;
     • если файл больше видеопамяти, но меньше RAM → slower but works;
     • правило: файл ≤ (видеопамять − 1.5 ГБ) для полной скорости.

3. Включить сервер, чтобы Джарвис увидел модель
   • Слева нажмите иконку «Developer / Server» (молния или надписью Server).
   • Вверху выберите скачанную модель.
   • Preset «LLM» (или «VLM» для gemma 3 — она видит картинки).
   • Context length: 4096 и выше. GPU offload: максимум (ползунок вправо).
   • Нажмите «Start Server» — адрес будет http://localhost:1234.

4. Подключить в Джарвисе
   • Настройки → «ИИ-модель»: Провайдер = «LM Studio (локально)».
   • Модель = имя, выбранное в LM Studio (можно «local-model»).
   • Base URL: http://localhost:1234/v1 (уже прописан).
   • «Проверить подключение» → «Подключение к LM Studio успешно».

5. Частые проблемы
   • «Failed to connect» → сервер не запущен (шаг 3) или занят порт 1234.
   • Модель грузится вечно → квант больше памяти, берите Q3/IQ2 или 3B.
   • Медленные ответы → убавьте контекст до 4096 и включите GPU offload.

6. Где лежат файлы моделей
   C:\\Users\\<вы>\\.cache\\lm-studio\\models — можно копировать вручную."""
