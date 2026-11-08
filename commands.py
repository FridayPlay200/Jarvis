"""Офлайн-команды: мгновенные локальные действия без обращения к LLM.

Каждая команда имеет:
  * id       — используется для поиска аудиофайла assets/offline/<id>.wav;
  * patterns — регулярные выражения по нормализованному тексту;
  * response — фраза-подтверждение (или функция, возвращающая фразу);
  * action   — локальное действие через ToolRegistry;
  * ui_action — действие интерфейса (настройки, свернуть, выход).
"""

from __future__ import annotations

import difflib
import platform
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import psutil

from config import Settings
from os_control import ToolRegistry
from speech import normalize_text

# --- Адаптеры приложений и сайтов (из jarvis.txt) --------------------

APP_ALIASES = [
    (("браузер", "хром", "chrome", "гугл хром", "google chrome"), "chrome"),
    (("блокнот", "notepad", "заметки"), "notepad"),
    (("калькулятор", "calc"), "calc"),
    (("проводник", "мои файлы", "файлы", "explorer"), "explorer"),
    (("спотифай", "spotify", "музыку", "музыка"), "spotify"),
    (("редактор кода", "вс код", "vs code", "код"), "code"),
    (("терминал", "консоль", "командную строку"), "terminal"),
]

SITE_ALIASES = {
    "ютуб": "https://www.youtube.com",
    "youtube": "https://www.youtube.com",
    "почту": "https://mail.google.com",
    "почта": "https://mail.google.com",
    "гMAIL": "https://mail.google.com",
    "вконтакте": "https://vk.com",
    "vk": "https://vk.com",
    "гитхаб": "https://github.com",
    "github": "https://github.com",
    "карты": "https://maps.google.com",
    "google maps": "https://maps.google.com",
    "переводчик": "https://translate.google.com",
    "translate": "https://translate.google.com",
    "ютубе": "https://www.youtube.com",
    "ютюбе": "https://www.youtube.com",
}


@dataclass
class OfflineMatch:
    command_id: str
    response: str
    action: Callable[[], str] | None = None
    ui_action: str | None = None


_MONTHS = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]
_WEEKDAYS = [
    "понедельник", "вторник", "среда", "четверг",
    "пятница", "суббота", "воскресенье",
]


def _time_response() -> str:
    return f"Сейчас {datetime.now():%H:%M}, сэр."


def _date_response() -> str:
    now = datetime.now()
    return (
        f"Сегодня {_WEEKDAYS[now.weekday()]}, {now.day} "
        f"{_MONTHS[now.month - 1]} {now.year} года, сэр."
    )


def _status_response() -> str:
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    free_gb = memory.available / 1024**3
    return (
        f"Процессор загружен на {cpu:.0f} процентов, память — на {memory.percent:.0f} процентов. "
        f"Свободно {free_gb:.1f} гигабайта, сэр."
    )


class OfflineCommands:
    """Сопоставление текста с локальными командами."""

    def __init__(self, tools: ToolRegistry, settings: Settings) -> None:
        self.tools = tools
        self.settings = settings
        self._commands: list[dict] = []
        self._register()

    # ------------------------------------------------------------------
    def _add(
        self,
        command_id: str,
        patterns: list[str],
        response: str | Callable[[], str],
        action: Callable[[], str] | None = None,
        ui_action: str | None = None,
    ) -> None:
        self._commands.append(
            {
                "id": command_id,
                "patterns": [re.compile(p, re.IGNORECASE) for p in patterns],
                "response": response,
                "action": action,
                "ui_action": ui_action,
            }
        )

    def _tool(self, tool_name: str, **arguments) -> Callable[[], str]:
        def run() -> str:
            return self.tools.execute(tool_name, arguments)

        return run

    # ------------------------------------------------------------------
    def _register(self) -> None:
        # --- информация -------------------------------------------------
        self._add(
            "time",
            [r"\b(который час|сколько времени|скажи время|текущее время)\b"],
            _time_response,
        )
        self._add(
            "date",
            [r"\b(какое сегодня число|какая сегодня дата|какой сегодня день|сегодняшняя дата)\b"],
            _date_response,
        )
        self._add(
            "system_status",
            [r"\b(статус системы|состояние системы|загрузка процессора|сколько оперативной памяти|сколько памяти)\b"],
            _status_response,
        )
        self._add(
            "help",
            [r"\b(что ты умеешь|список команд|помощь|твои возможности)\b"],
            "Я управляю громкостью и звуком, открываю приложения и сайты, делаю скриншоты, "
            "показываю состояние системы, блокирую и выключаю компьютер. "
            "Сложные задачи вроде «найди трек на YouTube» я выполняю через ИИ, сэр.",
        )

        # --- звук -------------------------------------------------------
        self._add(
            "volume_up",
            [r"\b(громче|увеличь громкость|прибавь громкость|прибавь звук)\b"],
            "Громкость увеличена, сэр.",
            self._tool("change_volume", delta=10),
        )
        self._add(
            "volume_down",
            [r"\b(тише|уменьши громкость|убавь громкость|убавь звук)\b"],
            "Громкость уменьшена, сэр.",
            self._tool("change_volume", delta=-10),
        )
        self._add(
            "mute",
            [r"\b(выключи звук|отключи звук|без звука)\b"],
            "Звук выключен, сэр.",
            self._tool("mute_audio", mute=True),
        )
        self._add(
            "unmute",
            [r"\b(включи звук|верни звук)\b"],
            "Звук включён, сэр.",
            self._tool("mute_audio", mute=False),
        )

        # --- система ----------------------------------------------------
        self._add(
            "screenshot",
            [r"\b(скриншот|снимок экрана|сделай скрин)\b"],
            "Снимок экрана сохранён, сэр.",
            self._tool("screenshot", name=""),
        )
        self._add(
            "lock",
            [r"\b(заблокируй компьютер|заблокируй экран|заблокируй систему)\b"],
            "Блокирую систему, сэр.",
            self._tool("system_power", action="lock"),
        )
        self._add(
            "shutdown",
            [r"\b(выключи компьютер|выключи пк|заверши работу)\b"],
            "Выключаю компьютер через десять секунд, сэр. "
            "Чтобы отменить, скажите «отмени выключение».",
            self._tool("system_power", action="shutdown", delay_seconds=10),
        )
        self._add(
            "restart",
            [r"\b(перезагрузи компьютер|перезагрузи пк)\b"],
            "Перезагружаю компьютер через десять секунд, сэр.",
            self._tool("system_power", action="restart", delay_seconds=10),
        )
        self._add(
            "cancel_shutdown",
            [r"\b(отмени выключение|отмени перезагрузку|не выключай)\b"],
            "Выключение отменено, сэр.",
            self._tool("system_power", action="cancel"),
        )

        # --- приложения -------------------------------------------------
        self._add(
            "open_browser",
            [r"\b(открой браузер|запусти браузер)\b"],
            "Открываю браузер, сэр.",
            self._tool("open_app", name="браузер"),
        )
        self._add(
            "open_youtube",
            [r"\b(открой ютуб|открой youtube)\b"],
            "Открываю YouTube, сэр.",
            self._tool("open_url", url="https://www.youtube.com"),
        )
        self._add(
            "open_explorer",
            [r"\b(открой проводник|открой файлы|мои файлы)\b"],
            "Открываю проводник, сэр.",
            self._tool("open_app", name="проводник"),
        )
        self._add(
            "open_calc",
            [r"\b(открой калькулятор|запусти калькулятор)\b"],
            "Открываю калькулятор, сэр.",
            self._tool("open_app", name="калькулятор"),
        )
        self._add(
            "open_notepad",
            [r"\b(открой блокнот|запусти блокнот)\b"],
            "Открываю блокнот, сэр.",
            self._tool("open_app", name="блокнот"),
        )
        self._add(
            "minimize_all",
            [r"\b(сверни все окна|покажи рабочий стол|сверни окна)\b"],
            "Сворачиваю все окна, сэр.",
            self._tool("window_control", action="minimize_all"),
        )
        self._add(
            "close_window",
            [r"\b(закрой активное окно|закрой окно|закрой это окно)\b"],
            "Закрываю активное окно, сэр.",
            self._tool("window_control", action="close"),
        )

        # --- интерфейс --------------------------------------------------
        self._add(
            "settings",
            [r"\b(открой настройки|открыть настройки)\b"],
            "Открываю настройки, сэр.",
            ui_action="settings",
        )
        self._add(
            "hide",
            [r"\b(свернись|спрячься)\b"],
            "Сворачиваюсь, сэр.",
            ui_action="minimize",
        )
        self._add(
            "quit",
            [r"\b(закройся|отключись|заверши приложение)\b"],
            "До связи, сэр.",
            ui_action="quit",
        )

        # --- быстрые клавиши --------------------------------------------
        self._add(
            "new_tab",
            [r"\b(открой новую вкладку|новая вкладка|создай вкладку)\b"],
            "Открываю новую вкладку, сэр.",
            self._tool("press_keys", keys=["ctrl", "t"]),
        )
        self._add(
            "close_tab",
            [r"\b(закрой вкладку)\b"],
            "Закрываю вкладку, сэр.",
            self._tool("press_keys", keys=["ctrl", "w"]),
        )
        self._add(
            "refresh_page",
            [r"\b(обнови страницу|перезагрузи страницу)\b"],
            "Обновляю страницу, сэр.",
            self._tool("press_keys", keys=["f5"]),
        )
        self._add(
            "copy",
            [r"\b(скопируй|копируй)\b"],
            "Копирую, сэр.",
            self._tool("press_keys", keys=["ctrl", "c"]),
        )
        self._add(
            "paste",
            [r"\b(вставь)\b"],
            "Вставляю, сэр.",
            self._tool("press_keys", keys=["ctrl", "v"]),
        )
        self._add(
            "select_all",
            [r"\b(выдели все|выбери всё|выделить всё)\b"],
            "Выделяю всё, сэр.",
            self._tool("press_keys", keys=["ctrl", "a"]),
        )
        self._add(
            "save",
            [r"\b(сохрани)\b"],
            "Сохраняю, сэр.",
            self._tool("press_keys", keys=["ctrl", "s"]),
        )
        self._add(
            "undo",
            [r"\b(отмени|верни назад)\b"],
            "Отменяю, сэр.",
            self._tool("press_keys", keys=["ctrl", "z"]),
        )
        self._add(
            "fullscreen",
            [r"\b(полный экран)\b"],
            "Переключаю в полный экран, сэр.",
            self._tool("press_keys", keys=["f11"]),
        )
        self._add(
            "switch_window",
            [r"\b(переключи окно|следующее окно|смени окно)\b"],
            "Переключаю окно, сэр.",
            self._tool("press_keys", keys=["alt", "tab"]),
        )
        self._add(
            "scroll_down",
            [r"\b(прокрути вниз|листай вниз)\b"],
            "Прокручиваю вниз, сэр.",
            self._tool("press_keys", keys=["pagedown"]),
        )
        self._add(
            "scroll_up",
            [r"\b(прокрути вверх|листай вверх)\b"],
            "Прокручиваю вверх, сэр.",
            self._tool("press_keys", keys=["pageup"]),
        )

        # --- вежливость -------------------------------------------------
        self._add(
            "greeting",
            [r"^(привет|здравствуй|доброе утро|добрый день|добрый вечер)\b"],
            "Приветствую, сэр. Чем могу быть полезен?",
        )
        self._add(
            "how_are_you",
            [r"\b(как (у тебя |твои )?дела|как ты|как жизнь|как самочувствие)\b"],
            "Все системы в норме, сэр.",
        )
        self._add(
            "thanks",
            [r"\b(спасибо|благодарю)\b"],
            "Всегда к вашим услугам, сэр.",
        )

        # --- умный поиск: настраивается динамически ---------------------
        self._register_smart_patterns()

    def _register_smart_patterns(self) -> None:
        """Динамические паттерны: поиск, сайты, приложения, ввод текста."""

        # «найди X на ютубе» → search YouTube
        self._add(
            "youtube_search",
            [r"(?:найди|поищи|включи|запусти)\s+(.+?)\s+(?:в|на)\s+(?:ютубе|ютюбе|youtube)"],
            "Ищу на YouTube, сэр.",
            action=None,  # action устанавливается при матчинге
        )
        self._commands[-1]["matcher"] = "youtube_search"

        # «найди X» → поиск в интернете
        self._add(
            "search_web",
            [r"(?:найди|поищи|загугли|погугли|поиск)\s+(.+)"],
            "Ищу в интернете, сэр.",
            action=None,
        )
        self._commands[-1]["matcher"] = "search_web"

        # «напиши X» / «введи X» → печать
        self._add(
            "type_text",
            [r"(?:напиши|набери|введи|впиши|напечатай)\s+(.+)"],
            "Печатаю, сэр.",
            action=None,
        )
        self._commands[-1]["matcher"] = "type_text"

        # «открой X» → сайт или приложение
        self._add(
            "open_smart",
            [r"(?:открой|запусти|включи|зайди на|зайди в|перейди на)\s+(.+)"],
            "Открываю, сэр.",
            action=None,
        )
        self._commands[-1]["matcher"] = "open_smart"

        # «курс доллара / погода / новости / рецепт» → поиск
        self._add(
            "lookup",
            [r"\b(курс валют|курс доллара|курс евро|погода|новости|рецепт|что такое|кто такой)\b"],
            "Ищу в интернете, сэр.",
            action=None,
        )
        self._commands[-1]["matcher"] = "lookup"


    # ------------------------------------------------------------------
    def match(self, text: str) -> OfflineMatch | None:
        normalized = normalize_text(text)
        if not normalized:
            return None
        for command in self._commands:
            for pattern in command["patterns"]:
                m = pattern.search(normalized)
                if not m:
                    continue
                matcher = command.get("matcher", "")
                action = command["action"]
                if matcher == "youtube_search":
                    query = m.group(1).strip()
                    action = self._tool(
                        "play_youtube", query=query
                    )
                elif matcher == "search_web":
                    query = m.group(1).strip()
                    action = self._tool("search_web", query=query)
                elif matcher == "type_text":
                    text_to_type = m.group(1).strip()
                    action = self._tool("type_text", text=text_to_type)
                elif matcher == "open_smart":
                    target = m.group(1).strip()
                    action = self._build_open_action(target)
                    if action is None:
                        continue
                elif matcher == "lookup":
                    query = normalized
                    action = self._tool("search_web", query=query)
                response = command["response"]
                phrase = response() if callable(response) else str(response)
                return OfflineMatch(
                    command_id=command["id"],
                    response=phrase,
                    action=action,
                    ui_action=command["ui_action"],
                )
        return None

    def _build_open_action(self, target: str) -> Callable[[], str] | None:
        """Определяет, что открывать: сайт по алиасу, приложение или поиск."""
        target_lower = target.lower()
        for alias, url in SITE_ALIASES.items():
            if alias in target_lower:
                return self._tool("open_url", url=url)
        for aliases, app in APP_ALIASES:
            if any(al in target_lower for al in aliases):
                return self._tool("open_app", name=app)
        # Неизвестный объект — ищем в интернете.
        return self._tool("search_web", query=target)


    def describe(self) -> str:
        """Список команд для лога/README."""
        lines = []
        for command in self._commands:
            pattern = command["patterns"][0].pattern
            lines.append(f"{command['id']}: {pattern}")
        return "\n".join(lines)
