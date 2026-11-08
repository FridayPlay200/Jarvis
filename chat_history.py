"""Хранение и загрузка историй диалогов.

Каждая сессия — это список сообщений (user/assistant/tool), которые
сохраняются в отдельный JSON-файл. Файлы лежат в папке `chats/` рядом
с исполняемым файлом (или в домашней директории при запуске из исходников).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import APP_DIR

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Путь к папке с чатами
# ---------------------------------------------------------------------------

def _chats_dir() -> Path:
    """Возвращает папку для хранения чатов."""
    return APP_DIR / "chats"


# ---------------------------------------------------------------------------
# Модель сессии
# ---------------------------------------------------------------------------

def _make_id() -> str:
    """Уникальный идентификатор сессии."""
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    """Текущее время в ISO-формате."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Открытие / сохранение
# ---------------------------------------------------------------------------

def open_session(title: str = "Новый диалог") -> dict[str, Any]:
    """Создаёт новую сессию и возвращает её как словарь."""
    _chats_dir().mkdir(parents=True, exist_ok=True)
    session: dict[str, Any] = {
        "id": _make_id(),
        "title": title,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "messages": [],
    }
    return session


def save_session(session: dict[str, Any]) -> None:
    """Сохраняет сессию в JSON-файл."""
    chat_dir = _chats_dir()
    chat_dir.mkdir(parents=True, exist_ok=True)
    path = chat_dir / f"{session['id']}.json"
    session["updated_at"] = _now_iso()
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    log.debug("Saved chat session %s", session["id"])


def list_sessions() -> list[dict[str, Any]]:
    """Возвращает список всех сессий (без полного текста), от новых к старым."""
    chat_dir = _chats_dir()
    if not chat_dir.exists():
        return []

    sessions = []
    for p in chat_dir.glob("*.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "id": raw.get("id", p.stem),
                "title": raw.get("title", "Без названия"),
                "created_at": raw.get("created_at", ""),
                "updated_at": raw.get("updated_at", ""),
                "message_count": len(raw.get("messages", [])),
            })
        except Exception:
            log.exception("Failed to read chat %s", p)

    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions


def load_session(session_id: str) -> dict[str, Any] | None:
    """Загружает полную сессию по ID. Возвращает None если не найдена."""
    chat_dir = _chats_dir()
    path = chat_dir / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw
    except Exception:
        log.exception("Failed to load chat %s", session_id)
        return None


def delete_session(session_id: str) -> bool:
    """Удаляет сессию по ID. Возвращает True если удалена."""
    chat_dir = _chats_dir()
    path = chat_dir / f"{session_id}.json"
    if path.exists():
        path.unlink()
        log.info("Deleted chat session %s", session_id)
        return True
    return False


def compact_messages(messages: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    """Ограничивает количество сообщений в истории для экономии контекста.

    Оставляет последние `limit` сообщений и удаляет более старые.
    """
    if len(messages) <= limit:
        return messages
    return messages[-limit:]
