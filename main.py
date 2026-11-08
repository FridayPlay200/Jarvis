"""Точка входа J.A.R.V.I.S.

Запуск:
    python main.py
    python main.py --no-audio     # без микрофона/озвучки (отладка UI)
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler

from config import LOGS_DIR, ensure_dirs


def check_admin_rights() -> bool:
    """Проверяет, запущен ли процесс с правами администратора.

    Вне Windows проверка не требуется — считаем права достаточными.
    """
    if os.name != "nt":
        return True
    import ctypes

    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


_CRASH_FILE = None


def setup_logging() -> None:
    ensure_dirs()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        LOGS_DIR / "jarvis.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console],
    )

    # Падения на уровне нативного кода (Qt/аудио/OpenMP) не ловятся
    # исключениями Python. faulthandler пишет стек в logs/crash.log.
    global _CRASH_FILE
    try:
        import datetime

        _CRASH_FILE = open(LOGS_DIR / "crash.log", "a", buffering=1, encoding="utf-8")
        _CRASH_FILE.write(
            f"\n=== запуск {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ===\n"
        )
        faulthandler.enable(file=_CRASH_FILE, all_threads=True)
    except Exception:
        pass

    # Убираем шум загрузчиков моделей и HTTP-клиентов: он забивает консоль.
    for name in (
        "httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub",
        "huggingface_hub.utils._http", "faster_whisper", "comtypes",
        "PIL", "matplotlib",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. — голосовой ИИ-ассистент")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Не запускать микрофон и озвучку (полезно для отладки интерфейса)",
    )
    return parser.parse_args()


def main() -> int:
    # --- Проверка прав администратора ---
    if not check_admin_rights():
        from PySide6.QtWidgets import QMessageBox, QApplication
        from PySide6.QtGui import QFont
        app = QApplication(sys.argv)
        app.setFont(QFont("Segoe UI", 10))
        QMessageBox.critical(
            None,
            "Требуемые права",
            "J.A.R.V.I.S. требует прав администратора для полного управления компьютером.\n\n"
            "Пожалуйста, запустите исполняемый файл от имени администратора.\n\n"
            "Для этого: нажмите правой кнопкой на jarvis.exe → «Запуск от имени администратора».",
            QMessageBox.Ok,
        )
        return 1

    args = parse_args()
    if args.no_audio:
        os.environ["JARVIS_DISABLE_AUDIO"] = "1"

    # Эти переменные должны быть выставлены до импорта faster-whisper/huggingface.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    setup_logging()
    log = logging.getLogger("jarvis")

    try:
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "PySide6 не установлен.\n"
            "Установите зависимости: python -m pip install -r requirements.txt"
        )
        return 1

    from ui import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("J.A.R.V.I.S.")
    app.setApplicationDisplayName("J.A.R.V.I.S.")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        log.critical(
            "Необработанное исключение", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception

    # Защита от вылетов: необработанное исключение в ПОТОКЕ убивает весь
    # процесс. Перехватываем их и логируем, но не даём приложению упасть.
    def thread_exception_hook(args):
        log.critical(
            "Необработанное исключение в потоке %s",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = thread_exception_hook

    window = MainWindow(no_audio=args.no_audio)
    window.show()
    log.info("Интерфейс запущен")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())