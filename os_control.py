"""Управление компьютером: инструменты (function calling) для ИИ-агента.

Модуль не зависит от Qt и может использоваться отдельно.
Каждый инструмент описан JSON-схемой в формате OpenAI и Anthropic,
а выполняется обычным Python-методом.

Безопасность:
  * разрушительные shell-команды блокируются чёрным списком;
  * опасные инструменты запрашивают подтверждение у пользователя
    (если включена настройка confirm_dangerous);
  * kill_process не трогает системные процессы.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus

import psutil

from config import APP_DIR, SCRIPTS_DIR, SCREENSHOTS_DIR, Settings

log = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# Команды, которые нельзя выполнять даже с подтверждением.
_FORBIDDEN_PATTERNS = [
    r"\bformat\s+[a-z]:",
    r"\bmkfs(\.\w+)?\b",
    r"\bdiskpart\b",
    r"\bdd\s+.*of=/dev/",
    r"\brm\s+-rf\s+/(\s|$)",
    r"\bdel\s+/[fsq].*[a-z]:\\",
    r"\brd\s+/s\s+/q\s+[a-z]:\\",
    r"\bbcdedit\b",
    r"\bvssadmin\s+delete",
    r"\bcipher\s+/w",
    r"\breg\s+delete\s+hk(lm|cu)\\\\?system",
]

_PROTECTED_PROCESSES = {
    "system", "system idle process", "smss.exe", "csrss.exe", "wininit.exe",
    "winlogon.exe", "services.exe", "lsass.exe", "svchost.exe", "explorer.exe",
    "dwm.exe", "fontdrvhost.exe", "sihost.exe", "taskhostw.exe", "ctfmon.exe",
    "init", "systemd", "kthreadd", "kernel_task", "launchd",
}

_WIN_APPS = {
    "браузер": "browser",
    "browser": "browser",
    "проводник": "explorer",
    "explorer": "explorer",
    "файлы": "explorer",
    "калькулятор": "calc",
    "calculator": "calc",
    "блокнот": "notepad",
    "notepad": "notepad",
    "терминал": "wt",
    "terminal": "wt",
    "консоль": "cmd",
    "cmd": "cmd",
    "paint": "mspaint",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
}

_LINUX_APPS = {
    "browser": "xdg-open https://www.google.com",
    "explorer": "nautilus",
    "файлы": "nautilus",
    "calculator": "gnome-calculator",
    "калькулятор": "gnome-calculator",
    "notepad": "gedit",
    "блокнот": "gedit",
    "terminal": "gnome-terminal",
    "терминал": "gnome-terminal",
}

# Если приложение не установлено — открываем его веб-версию.
# Ключи — нормализованные имена (нижний регистр, без лишних символов).
_WEB_APPS: dict[str, str] = {
    "telegram": "https://web.telegram.org/a/",
    "телеграм": "https://web.telegram.org/a/",
    "телеграмм": "https://web.telegram.org/a/",
    "тг": "https://web.telegram.org/a/",
    "whatsapp": "https://web.whatsapp.com/",
    "ватсап": "https://web.whatsapp.com/",
    "вотсап": "https://web.whatsapp.com/",
    "discord": "https://discord.com/app",
    "дискорд": "https://discord.com/app",
    "spotify": "https://open.spotify.com/",
    "спотифай": "https://open.spotify.com/",
    "яндекс музыка": "https://music.yandex.ru/",
    "яндексмузыка": "https://music.yandex.ru/",
    "яндекс.музыка": "https://music.yandex.ru/",
    "yandex music": "https://music.yandex.ru/",
    "yandexmusic": "https://music.yandex.ru/",
    "яндекс": "https://ya.ru/",
    "yandex": "https://ya.ru/",
    "ютуб": "https://www.youtube.com/",
    "youtube": "https://www.youtube.com/",
    "vk": "https://vk.com/",
    "вк": "https://vk.com/",
    "вконтакте": "https://vk.com/",
    "instagram": "https://www.instagram.com/",
    "инстаграм": "https://www.instagram.com/",
    "twitter": "https://x.com/",
    "твиттер": "https://x.com/",
    "netflix": "https://www.netflix.com/",
    "нетфликс": "https://www.netflix.com/",
    "twitch": "https://www.twitch.tv/",
    "твич": "https://www.twitch.tv/",
    "github": "https://github.com/",
    "гитхаб": "https://github.com/",
    "chatgpt": "https://chatgpt.com/",
    "чатгпт": "https://chatgpt.com/",
    "claude": "https://claude.ai/",
    "клод": "https://claude.ai/",
    "deepseek": "https://chat.deepseek.com/",
    "gmail": "https://mail.google.com/",
    "почта": "https://mail.google.com/",
    "google": "https://www.google.com/",
    "гугл": "https://www.google.com/",
    "notion": "https://www.notion.so/",
    "figma": "https://www.figma.com/",
    "steam": "https://store.steampowered.com/",
    "стим": "https://store.steampowered.com/",
    "zoom": "https://zoom.us/join",
    "зум": "https://zoom.us/join",
    "teams": "https://teams.microsoft.com/",
    "тимс": "https://teams.microsoft.com/",
    "skype": "https://web.skype.com/",
    "скайп": "https://web.skype.com/",
    "viber": "https://web.viber.com/",
    "вайбер": "https://web.viber.com/",
    "word": "https://www.office.com/launch/word",
    "ворд": "https://www.office.com/launch/word",
    "excel": "https://www.office.com/launch/excel",
    "эксель": "https://www.office.com/launch/excel",
    "powerpoint": "https://www.office.com/launch/powerpoint",
    "калькулятор": "https://www.google.com/search?q=calculator",
    "calculator": "https://www.google.com/search?q=calculator",
}

# Каталоги, где Windows обычно ставит программы (для быстрого поиска exe).
_WIN_APP_DIRS = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps",
    Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
    Path(os.environ.get("ProgramData", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
    Path(os.environ.get("PROGRAMFILES", "")),
    Path(os.environ.get("PROGRAMFILES(X86)", "")),
)


def _normalize_app_name(name: str) -> str:
    lowered = (name or "").strip().lower().replace("ё", "е")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _web_url_for(name: str) -> str | None:
    """Подбирает веб-версию приложения по имени или возвращает None."""
    key = _normalize_app_name(name)
    if not key:
        return None
    if key in _WEB_APPS:
        return _WEB_APPS[key]
    # «открой телеграм мессенджер» и подобные уточнения.
    for alias, url in _WEB_APPS.items():
        if len(alias) >= 3 and alias in key:
            return url
    # Похоже на домен: «открой openai.com».
    if re.fullmatch(r"[a-z0-9][a-z0-9\-.]*\.[a-z]{2,}", key):
        return "https://" + key
    return None


def _windows_app_path(target: str) -> str | None:
    """Ищет установленное приложение: PATH, App Paths, меню «Пуск», Program Files."""
    raw = target.strip().strip('"')
    if not raw:
        return None
    exe = shutil.which(raw)
    if exe:
        return exe
    if not raw.lower().endswith(".exe"):
        exe = shutil.which(raw + ".exe")
        if exe:
            return exe

    # Реестр: App Paths — так Windows находит установленные программы.
    try:
        import winreg

        key_name = Path(raw).name
        if not key_name.lower().endswith(".exe"):
            key_name += ".exe"
        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(
                    root,
                    rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{key_name}",
                ) as handle:
                    value, _ = winreg.QueryValueEx(handle, "")
                if value and Path(value).exists():
                    return value
            except OSError:
                continue
    except Exception:
        pass

    # Меню «Пуск»: ярлыки вида «Telegram.lnk».
    wanted = Path(raw).stem.lower()
    for folder in _WIN_APP_DIRS:
        try:
            if not folder or not folder.exists():
                continue
            for shortcut in folder.rglob("*.lnk"):
                if shortcut.stem.lower() == wanted:
                    return str(shortcut)
        except OSError:
            continue

    # Program Files: первый exe, имя которого совпадает с запросом.
    for folder in _WIN_APP_DIRS[:2]:
        try:
            if not folder or not folder.exists():
                continue
            for candidate in folder.glob(f"*/{wanted}.exe"):
                return str(candidate)
        except OSError:
            continue
    return None


def _resolve_shortcut(path: str) -> str:
    """Достаёт цель ярлыка .lnk через WScript.Shell."""
    try:
        result = _powershell(
            f"(New-Object -ComObject WScript.Shell).CreateShortcut('{path}').TargetPath",
            timeout=15,
        )
    except Exception:
        return ""
    for line in result.splitlines():
        line = line.strip()
        if line and not line.startswith("Код возврата"):
            return line
    return ""


def _launch_windows_path(path: str) -> str:
    """Запускает файл/ярлык. Возвращает текст ошибки или пустую строку."""
    if path.lower().endswith(".lnk"):
        # Ярлык UWP-приложения или «битый» ярлык часто даёт WinError 1223.
        # Надёжнее сразу запускать его цель.
        target = _resolve_shortcut(path)
        if target.lower().startswith("shell:"):
            try:
                subprocess.Popen(["explorer", target], creationflags=subprocess.CREATE_NO_WINDOW)
                return ""
            except OSError as exc:
                return f"Не удалось запустить {path}: {exc}"
        if target and Path(target).exists():
            path = target
        else:
            return f"Ярлык не запускается, цель недоступна: {path}"

    try:
        os.startfile(path)  # type: ignore[attr-defined]
        return ""
    except OSError as first_error:
        log.warning("startfile failed for %s: %s", path, first_error)

    try:
        subprocess.Popen(["explorer", path], creationflags=subprocess.CREATE_NO_WINDOW)
        return ""
    except OSError:
        pass

    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", path],
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return ""
    except OSError as exc:
        return f"Не удалось запустить {path}: {exc}"
        if target and Path(target).exists():
            path = target
        else:
            # Цель ярлыка недоступна (UWP, удалённая программа, права).
            # os.startfile на таком ярлыке даёт WinError 1223, а explorer
            # показывает окно ошибки — считаем запуск неудачным.
            return f"Ярлык не запускается, цель недоступна: {path}"

    try:
        os.startfile(path)  # type: ignore[attr-defined]
        return ""
    except OSError as first_error:
        log.warning("startfile failed for %s: %s", path, first_error)

    # WinError 1223 (UAC/ассоциации) и подобное: пробуем explorer.
    try:
        subprocess.Popen(["explorer", path])
        return ""
    except OSError:
        pass

    try:
        subprocess.Popen(["cmd", "/c", "start", "", path], shell=False)
        return ""
    except OSError as exc:
        return f"Не удалось запустить {path}: {exc}"


def _launch_windows_app(path: str) -> str:
    """Запускает приложение и возвращает человекочитаемый результат."""
    error = _launch_windows_path(path)
    if error:
        return error
    return f"Запускаю {Path(path).name}."



def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (обрезано, всего {len(text)} символов)"


def _run(cmd: Any, timeout: int = 60, shell: bool = True, encoding: str | None = None) -> str:
    """Запускает команду и возвращает код возврата + stdout/stderr."""
    try:
        proc = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            errors="replace",
            encoding=encoding,
            timeout=timeout,
            cwd=str(APP_DIR),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return "Ошибка: превышено время ожидания команды."
    except OSError as exc:
        return f"Ошибка запуска: {exc}"
    out = (proc.stdout or "") + (proc.stderr or "")
    return _truncate(f"Код возврата: {proc.returncode}\n{out.strip()}")


def _powershell(script: str, timeout: int = 60) -> str:
    # Без явной кодировки PowerShell отдаёт текст в cp866 — русские имена
    # превращаются в «кракозябры». Принудительно переводим вывод в UTF-8.
    wrapped = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        "$OutputEncoding = [System.Text.Encoding]::UTF8; " + script
    )
    return _run(
        [
            "powershell", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-Command", wrapped,
        ],
        timeout=timeout,
        shell=False,
        encoding="utf-8",
    )


def _win_volume():
    """Возвращает интерфейс IAudioEndpointVolume (pycaw)."""
    from pycaw.pycaw import AudioUtilities

    device = AudioUtilities.GetSpeakers()
    endpoint = getattr(device, "EndpointVolume", None)
    if endpoint is not None:
        return endpoint

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import IAudioEndpointVolume

    interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return interface.QueryInterface(IAudioEndpointVolume)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]
    dangerous: bool = False


class ToolRegistry:
    """Реестр инструментов агента."""

    def __init__(
        self,
        settings: Settings,
        confirm_callback: Callable[[str, dict[str, Any]], bool] | None = None,
        on_event: Callable[[str], None] | None = None,
        on_artifact: Callable[[dict[str, Any]], None] | None = None,
        cursor_animator: Callable[[int, int], bool] | None = None,
    ) -> None:
        self.settings = settings
        self.confirm_callback = confirm_callback
        self.on_event = on_event or (lambda _msg: None)
        self.on_artifact = on_artifact
        self.cursor_animator = cursor_animator
        self._tools: dict[str, Tool] = {}
        self._register_builtin()

    # ------------------------------------------------------------------
    # Регистрация и схемы
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., str],
        dangerous: bool = False,
    ) -> None:
        self._tools[name] = Tool(name, description, parameters, func, dangerous)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def openai_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def anthropic_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    # ------------------------------------------------------------------
    # Выполнение
    # ------------------------------------------------------------------
    def execute(self, name: str, args: dict[str, Any] | None) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Ошибка: инструмент '{name}' не найден."

        if not isinstance(args, dict):
            args = {}

        # Отбрасываем аргументы, которых нет в схеме — модели иногда лишнее.
        allowed = set(tool.parameters.get("properties", {}))
        clean = {k: v for k, v in args.items() if k in allowed}

        if not self.settings.allow_shell and name in {
            "run_shell", "run_powershell", "write_and_run_script", "run_script"
        }:
            return "Выполнение shell-команд и скриптов отключено в настройках."

        if tool.dangerous and self.settings.confirm_dangerous and self.confirm_callback:
            self.on_event(f"Запрашиваю подтверждение: {name}")
            try:
                approved = bool(self.confirm_callback(name, clean))
            except Exception as exc:  # подтверждение не должно ломать агента
                log.exception("confirm callback failed")
                return f"Не удалось получить подтверждение: {exc}"
            if not approved:
                return "Пользователь отклонил выполнение инструмента."

        self.on_event(f"Выполняю инструмент: {name}")
        try:
            result = tool.func(**clean)
        except TypeError as exc:
            return f"Ошибка аргументов инструмента {name}: {exc}"
        except Exception as exc:
            log.exception("tool %s failed", name)
            return f"Ошибка выполнения {name}: {exc}"

        if isinstance(result, (dict, list)):
            result = json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    # ------------------------------------------------------------------
    # Встроенные инструменты
    # ------------------------------------------------------------------
    def _register_builtin(self) -> None:
        no_args = {"type": "object", "properties": {}, "additionalProperties": False}

        self.register(
            "get_system_info",
            "Возвращает сведения о системе: ОС, процессор, память, диск, батарея, время работы.",
            no_args,
            self.tool_get_system_info,
        )
        self.register(
            "run_shell",
            "Выполняет команду в системной оболочке (cmd/PowerShell на Windows, bash на Linux). "
            "Используй для задач, которые не покрыты другими инструментами.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Команда оболочки"},
                    "timeout": {"type": "integer", "description": "Таймаут в секундах", "default": 60},
                },
                "required": ["command"],
            },
            self.tool_run_shell,
            dangerous=True,
        )
        self.register(
            "run_powershell",
            "Выполняет PowerShell-скрипт (только Windows).",
            {
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "Текст PowerShell-скрипта"},
                    "timeout": {"type": "integer", "default": 60},
                },
                "required": ["script"],
            },
            self.tool_run_powershell,
            dangerous=True,
        )
        self.register(
            "write_and_run_script",
            "Сохраняет новый скрипт автоматизации в папку scripts приложения и сразу выполняет его. "
            "Используй, когда задачу проще решить скриптом (массовые действия, циклы, сложная логика).",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Короткое имя команды"},
                    "language": {
                        "type": "string",
                        "enum": ["python", "powershell", "batch", "bash"],
                        "description": "Язык скрипта",
                    },
                    "code": {"type": "string", "description": "Полный текст скрипта"},
                    "run": {"type": "boolean", "default": True, "description": "Запустить сразу"},
                },
                "required": ["name", "language", "code"],
            },
            self.tool_write_and_run_script,
            dangerous=True,
        )
        self.register(
            "run_script",
            "Запускает ранее сохранённый скрипт из папки scripts по имени.",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            self.tool_run_script,
            dangerous=True,
        )
        self.register(
            "list_scripts",
            "Показывает список сохранённых скриптов автоматизации.",
            no_args,
            self.tool_list_scripts,
        )
        self.register(
            "list_dir",
            "Показывает содержимое папки (файлы и подпапки). Используй перед работой с проектом.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Путь к папке"},
                    "depth": {"type": "integer", "default": 1, "description": "Глубина обхода"},
                },
                "required": ["path"],
            },
            self.tool_list_dir,
        )
        self.register(
            "read_file",
            "Читает текстовый файл с номерами строк. Используй, чтобы изучить код перед правкой.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "default": 1},
                    "max_lines": {"type": "integer", "default": 400},
                },
                "required": ["path"],
            },
            self.tool_read_file,
        )
        self.register(
            "write_file",
            "Создаёт файл с кодом или полностью перезаписывает существующий. "
            "Родительские папки создаются автоматически. Владелец увидит созданный файл в чате.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Полный путь к файлу"},
                    "content": {"type": "string", "description": "Полное содержимое файла"},
                },
                "required": ["path", "content"],
            },
            self.tool_write_file,
        )
        self.register(
            "edit_file",
            "Точечно правит файл: заменяет old_text на new_text (как в редакторах кода). "
            "old_text должен встречаться в файле ровно один раз. Владелец увидит дифф в чате.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string", "description": "Что заменить (точный фрагмент)"},
                    "new_text": {"type": "string", "description": "На что заменить"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["path", "old_text", "new_text"],
            },
            self.tool_edit_file,
        )
        self.register(
            "open_app",
            "Открывает приложение по имени (браузер, проводник, калькулятор, блокнот, терминал, "
            "telegram, discord, spotify и т.п.) или по пути к исполняемому файлу. "
            "Если программа не установлена, автоматически открывается её веб-версия в браузере.",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            self.tool_open_app,
        )
        self.register(
            "open_url",
            "Открывает URL в браузере по умолчанию.",
            {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            self.tool_open_url,
        )
        self.register(
            "search_web",
            "Открывает поиск в браузере (Google) по запросу.",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            self.tool_search_web,
        )
        self.register(
            "play_youtube",
            "Находит первое видео на YouTube по запросу и открывает его в браузере. "
            "Идеально для команд вида «включи трек ...».",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            self.tool_play_youtube,
        )
        self.register(
            "type_text",
            "Печатает текст в активном окне (поддерживает русский язык через буфер обмена).",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            self.tool_type_text,
        )
        self.register(
            "press_keys",
            "Нажимает клавиши или их комбинацию, например ['ctrl','w'] или 'alt+f4'.",
            {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список клавиш для одновременного нажатия",
                    },
                    "repeat": {"type": "integer", "default": 1},
                },
                "required": ["keys"],
            },
            self.tool_press_keys,
        )
        self.register(
            "mouse_action",
            "Действия мышью: click, move, scroll.",
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["click", "move", "scroll"]},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
                    "clicks": {"type": "integer", "default": 1},
                    "amount": {"type": "integer", "default": -3},
                },
                "required": ["action"],
            },
            self.tool_mouse_action,
        )
        self.register(
            "screenshot",
            "Делает снимок экрана и сохраняет его в папку screenshots.",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
            self.tool_screenshot,
        )
        self.register(
            "look_at_screen",
            "Смотрит на экран: делает снимок и передаёт его тебе как изображение. "
            "Используй, когда нужно увидеть, что происходит на экране, найти кнопку или поле, "
            "проверить результат действия, прочитать текст с экрана. "
            "Инструмент возвращает реальный размер экрана в пикселях — координаты для "
            "mouse_action указывай именно в этих пикселях.",
            {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Что именно нужно найти или проверить на экране",
                    }
                },
            },
            self.tool_look_at_screen,
        )
        self.register(
            "set_volume",
            "Устанавливает громкость системы в процентах (0-100).",
            {
                "type": "object",
                "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100}},
                "required": ["level"],
            },
            self.tool_set_volume,
        )
        self.register(
            "change_volume",
            "Изменяет громкость на delta процентов (например, -10 или +15).",
            {
                "type": "object",
                "properties": {"delta": {"type": "integer"}},
                "required": ["delta"],
            },
            self.tool_change_volume,
        )
        self.register(
            "mute_audio",
            "Включает или выключает звук.",
            {
                "type": "object",
                "properties": {"mute": {"type": "boolean"}},
                "required": ["mute"],
            },
            self.tool_mute_audio,
        )
        self.register(
            "set_brightness",
            "Устанавливает яркость экрана в процентах (0-100, если поддерживается).",
            {
                "type": "object",
                "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100}},
                "required": ["level"],
            },
            self.tool_set_brightness,
        )
        self.register(
            "media_control",
            "Управляет медиаплеером: play_pause, next, prev, stop.",
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["play_pause", "next", "prev", "stop"]}
                },
                "required": ["action"],
            },
            self.tool_media_control,
        )
        self.register(
            "window_control",
            "Управляет окнами: minimize_all (свернуть всё), close (закрыть активное), "
            "switch (переключить), show_desktop (показать рабочий стол).",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["minimize_all", "close", "switch", "show_desktop"],
                    }
                },
                "required": ["action"],
            },
            self.tool_window_control,
        )
        self.register(
            "list_processes",
            "Показывает топ процессов по потреблению памяти.",
            {
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 15}},
            },
            self.tool_list_processes,
        )
        self.register(
            "kill_process",
            "Завершает процесс по имени. Системные процессы защищены.",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            self.tool_kill_process,
            dangerous=True,
        )
        self.register(
            "system_power",
            "Управление питанием: shutdown (выключить), restart (перезагрузить), sleep (сон), "
            "lock (блокировка), cancel (отменить выключение). delay_seconds — задержка.",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["shutdown", "restart", "sleep", "lock", "cancel"],
                    },
                    "delay_seconds": {"type": "integer", "default": 0},
                },
                "required": ["action"],
            },
            self.tool_system_power,
        )

    # ------------------------------------------------------------------
    # Реализация инструментов
    # ------------------------------------------------------------------
    def tool_get_system_info(self) -> str:
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(str(Path.home()))
        info = {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_used_gb": round(vm.used / 1024**3, 1),
            "ram_total_gb": round(vm.total / 1024**3, 1),
            "disk_free_gb": round(disk.free / 1024**3, 1),
            "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
        }
        battery = getattr(psutil, "sensors_battery", lambda: None)()
        if battery:
            info["battery_percent"] = battery.percent
            info["battery_plugged"] = battery.power_plugged
        return json.dumps(info, ensure_ascii=False)

    def _is_forbidden(self, command: str) -> str | None:
        lowered = command.lower()
        for pattern in _FORBIDDEN_PATTERNS:
            if re.search(pattern, lowered):
                return pattern
        return None

    def tool_run_shell(self, command: str, timeout: int = 60) -> str:
        if not command.strip():
            return "Ошибка: пустая команда."
        bad = self._is_forbidden(command)
        if bad:
            return f"Команда заблокирована политикой безопасности ({bad})."
        timeout = max(1, min(int(timeout or 60), 300))
        return _run(command, timeout=timeout, shell=True)

    def tool_run_powershell(self, script: str, timeout: int = 60) -> str:
        if not IS_WINDOWS:
            return "PowerShell доступен только на Windows."
        if not script.strip():
            return "Ошибка: пустой скрипт."
        bad = self._is_forbidden(script)
        if bad:
            return f"Скрипт заблокирован политикой безопасности ({bad})."
        return _powershell(script, timeout=max(1, min(int(timeout or 60), 300)))

    def _script_path(self, name: str, language: str) -> Path:
        safe = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE).strip() or "script"
        safe = safe[:60]
        ext = {
            "python": ".py", "py": ".py",
            "powershell": ".ps1", "ps1": ".ps1",
            "batch": ".bat", "bat": ".bat", "cmd": ".bat",
            "bash": ".sh", "sh": ".sh",
        }.get(language.lower(), ".py")
        path = SCRIPTS_DIR / (safe if safe.lower().endswith(ext) else safe + ext)
        if path.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = path.with_name(f"{path.stem}-{stamp}{path.suffix}")
        return path

    def _launch_script(self, path: Path, timeout: int = 300) -> str:
        suffix = path.suffix.lower()
        if suffix == ".py":
            cmd = [sys.executable, str(path)]
        elif suffix == ".ps1":
            cmd = [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(path),
            ]
        elif suffix == ".bat":
            cmd = ["cmd", "/c", str(path)]
        elif suffix == ".sh":
            cmd = ["bash", str(path)]
        else:
            return f"Неизвестный тип скрипта: {suffix}"
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, errors="replace",
                timeout=timeout, cwd=str(path.parent),
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except subprocess.TimeoutExpired:
            return f"Скрипт {path.name} прерван по таймауту ({timeout} с)."
        except OSError as exc:
            return f"Не удалось запустить {path.name}: {exc}"
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return f"Скрипт {path.name} завершён (код {proc.returncode}).\n{_truncate(out)}"

    def tool_write_and_run_script(
        self, name: str, language: str, code: str, run: bool = True
    ) -> str:
        if not code.strip():
            return "Ошибка: пустой код скрипта."
        path = self._script_path(name, language)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        self._emit_artifact("create", path, "", code)
        message = f"Скрипт сохранён: {path}"
        if run:
            message += "\n" + self._launch_script(path)
        return message

    def tool_run_script(self, name: str) -> str:
        candidate = Path(name)
        if not candidate.is_absolute():
            matches = list(SCRIPTS_DIR.glob(Path(name).name))
            if not matches:
                matches = list(SCRIPTS_DIR.glob(f"{Path(name).stem}*"))
            if not matches:
                return f"Скрипт '{name}' не найден в {SCRIPTS_DIR}."
            candidate = sorted(matches)[-1]
        if not candidate.exists():
            return f"Файл {candidate} не существует."
        return self._launch_script(candidate)

    def tool_list_scripts(self) -> str:
        files = sorted(SCRIPTS_DIR.glob("*.*"))
        if not files:
            return "Сохранённых скриптов пока нет."
        return "\n".join(f"- {p.name}" for p in files)

    # --- работа с файлами и кодом --------------------------------------
    def _emit_artifact(
        self, action: str, path: Path, before: str, after: str
    ) -> None:
        """Сообщает UI о созданном/изменённом файле (карточка как в Claude Code)."""
        if not self.on_artifact:
            return
        import difflib

        before_lines = before.splitlines()
        after_lines = after.splitlines()
        diff: list[str] = []
        added = removed = 0
        if before:
            for line in difflib.unified_diff(
                before_lines, after_lines, lineterm="", n=1
            ):
                if line.startswith("+++") or line.startswith("---"):
                    continue
                if line.startswith("@@"):
                    diff.append(line)
                elif line.startswith("+"):
                    added += 1
                    diff.append(line)
                elif line.startswith("-"):
                    removed += 1
                    diff.append(line)
                else:
                    diff.append(" " + line)
        else:
            added = len(after_lines)
            diff = [f"+{line}" for line in after_lines]
        try:
            self.on_artifact(
                {
                    "action": action,
                    "path": str(path),
                    "added": added,
                    "removed": removed,
                    "lines": len(after_lines),
                    "language": path.suffix.lstrip(".").lower(),
                    "diff": diff[:200],
                }
            )
        except Exception:
            log.exception("artifact callback failed")

    def _resolve_write_path(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = APP_DIR / path
        return path

    def _write_blocked(self, path: Path) -> bool:
        """Защита системных папок от случайной перезаписи."""
        try:
            resolved = str(path.resolve()).lower()
        except OSError:
            resolved = str(path).lower()
        return resolved.startswith(
            ("c:\windows", "c:\program files", "c:\program files (x86)")
        )

    def tool_list_dir(self, path: str, depth: int = 1) -> str:
        root = self._resolve_write_path(path)
        if not root.exists():
            return f"Папка не найдена: {root}"
        if not root.is_dir():
            return f"Это файл, а не папка: {root}"
        depth = max(1, min(int(depth or 1), 3))
        skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea"}
        rows: list[str] = []
        base_depth = len(root.parts)
        try:
            for item in sorted(root.rglob("*")):
                if any(part in skip for part in item.parts):
                    continue
                level = len(item.parts) - base_depth
                if level > depth:
                    continue
                prefix = "  " * (level - 1)
                if item.is_dir():
                    rows.append(f"{prefix}{item.name}/")
                else:
                    size = item.stat().st_size
                    rows.append(f"{prefix}{item.name} ({size} Б)")
                if len(rows) >= 300:
                    rows.append("... (список обрезан)")
                    break
        except OSError as exc:
            return f"Ошибка чтения папки: {exc}"
        return "\n".join(rows) or "Папка пуста."

    def tool_read_file(
        self, path: str, start_line: int = 1, max_lines: int = 400
    ) -> str:
        target = self._resolve_write_path(path)
        if not target.exists():
            return f"Файл не найден: {target}"
        if target.is_dir():
            return f"Это папка: {target}"
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"Ошибка чтения: {exc}"
        start = max(1, int(start_line or 1))
        limit = max(1, min(int(max_lines or 400), 2000))
        chunk = lines[start - 1 : start - 1 + limit]
        body = "\n".join(
            f"{number:>5}: {line}"
            for number, line in enumerate(chunk, start=start)
        )
        tail = (
            f"\n... (всего {len(lines)} строк, показано до {start + len(chunk) - 1})"
            if start - 1 + len(chunk) < len(lines)
            else f"\n... (всего {len(lines)} строк)"
        )
        return f"{target}\n{body}{tail}"

    def tool_write_file(self, path: str, content: str) -> str:
        target = self._resolve_write_path(path)
        if self._write_blocked(target):
            return f"Запись в системную папку запрещена: {target}"
        if target.exists() and target.is_dir():
            return f"Это папка, а не файл: {target}"
        before = ""
        if target.exists():
            try:
                before = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                before = ""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"Ошибка записи {target}: {exc}"
        action = "update" if before else "create"
        self._emit_artifact(action, target, before, content)
        verb = "Обновил" if before else "Создал"
        return f"{verb} файл: {target} ({len(content.splitlines())} строк)."

    def tool_edit_file(
        self, path: str, old_text: str, new_text: str, replace_all: bool = False
    ) -> str:
        target = self._resolve_write_path(path)
        if self._write_blocked(target):
            return f"Запись в системную папку запрещена: {target}"
        if not target.exists():
            return f"Файл не найден: {target}"
        try:
            before = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Ошибка чтения: {exc}"
        if not old_text:
            return "Ошибка: пустой old_text."
        count = before.count(old_text)
        if count == 0:
            return (
                "Фрагмент не найден. Прочитай файл через read_file и повтори "
                "с точным текстом (включая отступы)."
            )
        if count > 1 and not replace_all:
            return (
                f"Фрагмент встречается {count} раз. Уточни old_text или "
                "передай replace_all=true."
            )
        after = before.replace(old_text, new_text) if replace_all else before.replace(
            old_text, new_text, 1
        )
        try:
            target.write_text(after, encoding="utf-8")
        except OSError as exc:
            return f"Ошибка записи {target}: {exc}"
        self._emit_artifact("update", target, before, after)
        return f"Изменён файл: {target} (заменено {count if replace_all else 1})."

    def tool_open_app(self, name: str) -> str:
        raw = name.strip()
        if not raw:
            return "Ошибка: не указано приложение."
        key = raw.lower()
        alias = _WIN_APPS.get(key) if IS_WINDOWS else _LINUX_APPS.get(key)

        if IS_WINDOWS:
            if alias == "browser":
                webbrowser.open("https://www.google.com")
                return "Открываю браузер."
            target = alias or raw
            try:
                if Path(target).exists():
                    result = _launch_windows_app(target)
                    if not result.startswith("Не удалось") and not result.startswith(
                        "Ярлык"
                    ):
                        return result
                else:
                    path = _windows_app_path(target)
                    if path:
                        result = _launch_windows_app(path)
                        if not result.startswith("Не удалось") and not result.startswith(
                            "Ярлык"
                        ):
                            return result
                        log.warning("app launch failed: %s", result)
            except OSError as exc:
                log.warning("open_app failed for %s: %s", raw, exc)
            # Ничего не запустилось — веб-версия.
            return self._open_web_version(raw)

        # Linux / macOS
        if alias:
            if alias.startswith("xdg-open "):
                subprocess.Popen(shlex.split(alias))
            else:
                subprocess.Popen([alias], start_new_session=True)
            return f"Запускаю {alias}."
        exe = shutil.which(raw)
        if exe:
            subprocess.Popen([exe], start_new_session=True)
            return f"Запускаю {exe}."
        return self._open_web_version(raw)

    def _open_web_version(self, raw: str) -> str:
        """Приложение не установлено — открываем его веб-версию."""
        url = _web_url_for(raw)
        if self.settings.web_fallback and url:
            webbrowser.open(url)
            return (
                f"«{raw}» не установлен на этом компьютере. "
                f"Открываю веб-версию: {url}"
            )
        if url:
            return (
                f"«{raw}» не установлен. Веб-версия доступна по адресу {url}, "
                "но автоматическое открытие сайтов отключено в настройках."
            )
        return f"Приложение '{raw}' не найдено и веб-версия неизвестна."

    def tool_open_url(self, url: str) -> str:
        url = url.strip()
        if not url:
            return "Ошибка: пустой URL."
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
            url = "https://" + url
        webbrowser.open(url)
        return f"Открываю {url}."

    def tool_search_web(self, query: str) -> str:
        url = "https://www.google.com/search?q=" + quote_plus(query)
        webbrowser.open(url)
        return f"Ищу в интернете: {query}"

    def tool_play_youtube(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "Ошибка: пустой запрос."
        try:
            import yt_dlp

            options = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "extract_flat": True,
                "socket_timeout": 15,
            }
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            entries = info.get("entries") or []
            if entries:
                video_id = entries[0].get("id")
                if video_id:
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    webbrowser.open(url)
                    title = entries[0].get("title") or query
                    return f"Включаю на YouTube: {title}\n{url}"
        except Exception as exc:
            log.warning("yt-dlp search failed: %s", exc)

        # Fallback: страница результатов поиска.
        url = "https://www.youtube.com/results?search_query=" + quote_plus(query)
        webbrowser.open(url)
        return f"Открываю результаты поиска YouTube: {query}"

    def tool_type_text(self, text: str) -> str:
        import pyautogui

        if not text:
            return "Ошибка: пустой текст."
        if text.isascii():
            pyautogui.write(text, interval=0.01)
        else:
            # pyautogui не умеет Unicode — используем буфер обмена.
            import pyperclip

            previous = ""
            try:
                previous = pyperclip.paste()
            except Exception:
                pass
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.2)
            if previous:
                try:
                    pyperclip.copy(previous)
                except Exception:
                    pass
        return "Текст введён."

    def tool_press_keys(self, keys: Any, repeat: int = 1) -> str:
        import pyautogui

        if isinstance(keys, str):
            keys = [k.strip() for k in re.split(r"[+,]", keys) if k.strip()]
        keys = [str(k) for k in (keys or [])]
        if not keys:
            return "Ошибка: не указаны клавиши."
        count = max(1, min(int(repeat or 1), 50))
        for _ in range(count):
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            time.sleep(0.15)
        return f"Нажато: {'+'.join(keys)} × {count}."

    def tool_mouse_action(
        self,
        action: str,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
        amount: int = -3,
    ) -> str:
        import pyautogui

        action = (action or "").lower()

        # Аватар сам «идёт» к точке и нажимает вместо курсора.
        if (
            self.cursor_animator is not None
            and x is not None
            and y is not None
            and action in {"click", "move", "scroll"}
        ):
            try:
                self.cursor_animator(int(x), int(y))
            except Exception:
                log.debug("cursor animator failed", exc_info=True)

        if action == "click":
            pyautogui.click(x=x, y=y, clicks=max(1, int(clicks or 1)), button=button)
            return f"Клик мышью ({x}, {y})."
        if action == "move":
            pyautogui.moveTo(x, y, duration=0.2)
            return f"Курсор перемещён в ({x}, {y})."
        if action == "scroll":
            pyautogui.scroll(int(amount or -3), x=x, y=y)
            return f"Прокрутка на {amount}."
        return f"Неизвестное действие мыши: {action}"

    def tool_screenshot(self, name: str = "") -> str:
        import pyautogui

        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = re.sub(r"[^\w\- ]+", "_", name).strip() or f"screenshot-{stamp}"
        path = SCREENSHOTS_DIR / f"{safe}.png"
        pyautogui.screenshot(str(path))
        return f"Снимок экрана сохранён: {path}"

    def tool_look_at_screen(self, question: str = "") -> str:
        """Снимок экрана для передачи модели как изображения."""
        import pyautogui
        from PIL import Image

        shot = pyautogui.screenshot()
        width, height = shot.size
        limit = max(640, int(self.settings.vision_max_width or 1280))
        if width > limit:
            ratio = limit / float(width)
            shot = shot.resize((limit, max(1, int(height * ratio))), Image.LANCZOS)

        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = SCREENSHOTS_DIR / f"screen-{stamp}.jpg"
        shot.convert("RGB").save(path, "JPEG", quality=85)
        return json.dumps(
            {
                "path": str(path),
                "screen_width": width,
                "screen_height": height,
                "question": question or "",
            },
            ensure_ascii=False,
        )

    # --- звук ----------------------------------------------------------
    def _get_volume(self) -> int:
        if IS_WINDOWS:
            try:
                return int(round(_win_volume().GetMasterVolumeLevelScalar() * 100))
            except Exception:
                return -1
        try:
            out = _run("pactl get-sink-volume @DEFAULT_SINK@", timeout=5)
            match = re.search(r"(\d+)%", out)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return -1

    def tool_set_volume(self, level: int) -> str:
        level = max(0, min(100, int(level)))
        if IS_WINDOWS:
            try:
                _win_volume().SetMasterVolumeLevelScalar(level / 100.0, None)
                return f"Громкость: {level}%."
            except Exception as exc:
                return f"Не удалось изменить громкость: {exc}"
        if IS_LINUX:
            return _run(f"pactl set-sink-volume @DEFAULT_SINK@ {level}%", timeout=10)
        return "Управление громкостью не поддерживается на этой ОС."

    def tool_change_volume(self, delta: int) -> str:
        current = self._get_volume()
        if current < 0:
            current = 50
        return self.tool_set_volume(current + int(delta))

    def tool_mute_audio(self, mute: bool) -> str:
        if IS_WINDOWS:
            try:
                _win_volume().SetMute(1 if mute else 0, None)
                return "Звук выключен." if mute else "Звук включён."
            except Exception as exc:
                return f"Не удалось переключить звук: {exc}"
        if IS_LINUX:
            _run(f"pactl set-sink-mute @DEFAULT_SINK@ {'1' if mute else '0'}", timeout=10)
            return "Звук выключен." if mute else "Звук включён."
        return "Управление звуком не поддерживается."

    def tool_set_brightness(self, level: int) -> str:
        level = max(0, min(100, int(level)))
        if IS_WINDOWS:
            return _powershell(
                "(Get-CimInstance -Namespace root/WMI "
                "-ClassName WmiMonitorBrightnessMethods).WmiSetBrightness(1,"
                f"{level})",
                timeout=20,
            )
        if IS_LINUX:
            return _run(f"brightnessctl set {level}%", timeout=10)
        return "Управление яркостью не поддерживается."

    def tool_media_control(self, action: str) -> str:
        action = (action or "").lower()
        if IS_WINDOWS:
            vk = {"play_pause": 0xB3, "next": 0xB0, "prev": 0xB1, "stop": 0xB2}
            code = vk.get(action)
            if not code:
                return f"Неизвестное медиа-действие: {action}"
            user32 = ctypes.windll.user32
            user32.keybd_event(code, 0, 0, 0)
            user32.keybd_event(code, 0, 2, 0)
            return f"Медиа-команда: {action}."
        if IS_LINUX:
            cmd = {
                "play_pause": "playerctl play-pause",
                "next": "playerctl next",
                "prev": "playerctl previous",
                "stop": "playerctl stop",
            }.get(action)
            if cmd:
                return _run(cmd, timeout=10)
        return "Медиа-управление не поддерживается."

    # --- окна и процессы ----------------------------------------------
    def tool_window_control(self, action: str) -> str:
        import pyautogui

        action = (action or "").lower()
        if action == "minimize_all":
            if IS_WINDOWS:
                _powershell("(New-Object -ComObject Shell.Application).MinimizeAll()", timeout=15)
                return "Все окна свёрнуты."
            return _run("wmctrl -k on", timeout=10)
        if action == "close":
            pyautogui.hotkey("alt", "f4")
            return "Активное окно закрыто."
        if action == "switch":
            pyautogui.hotkey("alt", "tab")
            return "Переключил окно."
        if action == "show_desktop":
            if IS_WINDOWS:
                pyautogui.hotkey("win", "d")
            else:
                _run("wmctrl -k on", timeout=10)
            return "Показан рабочий стол."
        return f"Неизвестное действие с окнами: {action}"

    def tool_list_processes(self, limit: int = 15) -> str:
        limit = max(1, min(int(limit or 15), 50))
        rows = []
        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                mem = proc.info["memory_info"]
                rows.append((proc.info["pid"], proc.info["name"] or "?", mem.rss if mem else 0))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        rows.sort(key=lambda r: r[2], reverse=True)
        lines = [f"{pid:>7}  {name:<30} {rss / 1024**2:>8.1f} MB" for pid, name, rss in rows[:limit]]
        return "PID      Процесс                         Память\n" + "\n".join(lines)

    def tool_kill_process(self, name: str) -> str:
        target = Path(name).stem.lower()
        if target in _PROTECTED_PROCESSES or target in {"explorer", "svchost"}:
            return f"Процесс '{name}' защищён и не будет завершён."
        killed = []
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                pname = (proc.info["name"] or "").lower()
                if Path(pname).stem == target:
                    proc.terminate()
                    killed.append(f"{proc.info['name']} ({proc.info['pid']})")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not killed:
            return f"Процесс '{name}' не найден."
        return "Завершены процессы: " + ", ".join(killed)

    # --- питание -------------------------------------------------------
    def tool_system_power(self, action: str, delay_seconds: int = 0) -> str:
        action = (action or "").lower()
        delay = max(0, min(int(delay_seconds or 0), 24 * 3600))

        if action in {"shutdown", "restart", "sleep"} and self.settings.confirm_dangerous:
            if self.confirm_callback and not self.confirm_callback(
                "system_power", {"action": action, "delay_seconds": delay}
            ):
                return "Пользователь отменил действие."

        if IS_WINDOWS:
            if action == "shutdown":
                return _run(f"shutdown /s /t {delay}", timeout=15)
            if action == "restart":
                return _run(f"shutdown /r /t {delay}", timeout=15)
            if action == "cancel":
                return _run("shutdown /a", timeout=15)
            if action == "sleep":
                return _run("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", timeout=15)
            if action == "lock":
                return _run("rundll32.exe user32.dll,LockWorkStation", timeout=15)
        elif IS_LINUX:
            if action == "shutdown":
                return _run(f"shutdown -h +{max(1, delay // 60)}", timeout=15) if delay else _run("systemctl poweroff", timeout=15)
            if action == "restart":
                return _run(f"shutdown -r +{max(1, delay // 60)}", timeout=15) if delay else _run("systemctl reboot", timeout=15)
            if action == "cancel":
                return _run("shutdown -c", timeout=15)
            if action == "sleep":
                return _run("systemctl suspend", timeout=15)
            if action == "lock":
                return _run("loginctl lock-session", timeout=15)
        return f"Действие '{action}' не поддерживается на этой ОС."
