"""Интерфейс J.A.R.V.I.S. на PySide6.

Содержит:
  * анимированную сферу-визуализатор (пульсирует от уровня микрофона и озвучки);
  * тёмный sci-fi интерфейс с неоновыми акцентами;
  * окно настроек (имя, провайдер, модель, ключи, TTS/STT);
  * связку UI ↔ LLM ↔ инструменты ↔ речь.
"""

from __future__ import annotations

import copy
import html
import json
import logging
import math
import os
import threading

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient, QTextOption
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QSplitter,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai_manager import AIManager
import local_models
from avatar import AvatarPreview, AvatarWindow
from chat_history import (
    delete_session,
    list_sessions,
    load_session,
    open_session,
    save_session,
)
from commands import OfflineCommands
from config import (
    OFFLINE_AUDIO_DIR,
    PROVIDERS,
    STT_ENGINES,
    TTS_ENGINES,
    Settings,
    ensure_dirs,
    is_autostart_enabled,
    set_autostart,
)
from os_control import ToolRegistry
from speech import (
    PIPER_VOICES,
    TTSManager,
    VoiceListener,
    create_stt,
    fetch_elevenlabs_voices,
    is_filler,
    is_stop_command,
    list_input_devices,
    match_wake_word,
    normalize_text,
    resolve_input_device,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Фразы ассистента (вес — относительная вероятность выбора)
# ---------------------------------------------------------------------------

import random

WAKE_PHRASES: list[tuple[str, int]] = [
    ("Вы меня вызывали, сэр?", 40),
    ("Да, сэр? Вы меня звали?", 22),
    ("Слушаю, сэр. Вы меня вызывали?", 15),
    ("Я здесь, сэр. Что угодно?", 10),
    ("К вашим услугам, сэр. Вы меня вызывали?", 8),
    ("Да, сэр?", 5),
]

SESSION_PHRASES: list[tuple[str, int]] = [
    ("Слушаю, сэр.", 45),
    ("Внимательно слушаю, сэр.", 25),
    ("Да, сэр. Говорите.", 20),
    ("К вашим услугам, сэр.", 10),
]

STOP_PHRASES: list[tuple[str, int]] = [
    ("Отключаюсь, сэр.", 35),
    ("Ухожу в режим ожидания, сэр.", 25),
    ("Как прикажете, сэр.", 20),
    ("До связи, сэр.", 12),
    ("Молчу и жду вашего имени, сэр.", 8),
]

DECLINE_PHRASES: list[tuple[str, int]] = [
    ("Понял, сэр. Буду ждать.", 45),
    ("Хорошо, сэр. Если понадоблюсь — позовите.", 30),
    ("Как скажете, сэр. Ухожу в ожидание.", 25),
]

# Короткие реплики, чтобы не молчать во время долгой работы.
ACTIVITY_PHRASES: list[tuple[str, int]] = [
    ("Работаю, сэр.", 20),
    ("Уже делаю.", 15),
    ("Секунду, сэр.", 12),
    ("Почти готово.", 12),
    ("Завершаю.", 10),
    ("Ещё пара движений.", 10),
    ("Всё под контролем.", 8),
    ("Как в лучших домах.", 6),
    ("Делаю, не скучаю.", 7),
]


def pick_phrase(phrases: list[tuple[str, int]]) -> str:
    """Случайная фраза с учётом весов."""
    words = [phrase for phrase, _weight in phrases]
    weights = [max(1, weight) for _phrase, weight in phrases]
    return random.choices(words, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Стили
# ---------------------------------------------------------------------------

STYLE = """
QWidget {
    color: #cfe9f5;
    font-family: "Segoe UI", "Inter", "Noto Sans", Arial;
    font-size: 13px;
}
QWidget#Root { background-color: #04070d; }
QDialog { background-color: #050b13; }
QFrame#TitleBar {
    background-color: #050b13;
    border-bottom: 1px solid rgba(45, 150, 190, 70);
}
QLabel#TitleLabel {
    color: #e8fbff;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 4px;
}
QLabel#SubtitleLabel {
    color: #4b8ea8;
    font-size: 10px;
    letter-spacing: 1px;
}
QLabel#SectionLabel {
    color: #3fa7c9;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#StateLabel {
    color: #67f0ff;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 4px;
}
QLabel#HintLabel { color: #4f7f95; font-size: 11px; }
QLabel#ValueLabel { color: #9fd8ea; font-size: 12px; }
QFrame#Panel {
    background-color: rgba(8, 18, 30, 225);
    border: 1px solid rgba(45, 150, 190, 80);
    border-radius: 14px;
}
QSplitter::handle {
    background-color: rgba(45, 150, 190, 40);
}
QSplitter::handle:horizontal {
    width: 5px;
}
QSplitter::handle:vertical {
    height: 5px;
}
QSplitter::handle:hover {
    background-color: rgba(60, 200, 240, 140);
}
QPushButton {
    background-color: rgba(10, 32, 48, 190);
    border: 1px solid rgba(60, 180, 220, 120);
    border-radius: 9px;
    padding: 9px 14px;
    color: #a9e9ff;
    font-weight: 600;
}
QPushButton:hover {
    background-color: rgba(16, 60, 84, 230);
    border-color: #45d8ff;
}
QPushButton:pressed { background-color: rgba(20, 90, 120, 230); }
QPushButton:disabled { color: #3c5b68; border-color: #1b3a48; }
QPushButton#AccentButton {
    background-color: rgba(0, 170, 210, 70);
    border: 1px solid #35d6ff;
    color: #e6fbff;
}
QPushButton#AccentButton:hover { background-color: rgba(0, 200, 240, 110); }
QPushButton#WinButton {
    border: none;
    background: transparent;
    padding: 4px 10px;
    color: #7fb8cc;
    font-size: 15px;
}
QPushButton#WinButton:hover { background-color: rgba(60, 180, 220, 45); color: #ffffff; }
QPushButton#CloseButton {
    border: none;
    background: transparent;
    padding: 4px 10px;
    color: #7fb8cc;
    font-size: 15px;
}
QPushButton#CloseButton:hover { background-color: #b83232; color: #ffffff; }
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: rgba(6, 16, 26, 235);
    border: 1px solid rgba(50, 150, 190, 95);
    border-radius: 8px;
    padding: 8px;
    selection-background-color: #0e6b8a;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #45d8ff; }
QTextEdit#LogView {
    background-color: rgba(4, 11, 18, 205);
    border: 1px solid rgba(40, 120, 160, 65);
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #07131f;
    border: 1px solid #2a7f9e;
    selection-background-color: #0e4a63;
}
QTabWidget::pane {
    border: 1px solid rgba(45, 150, 190, 80);
    border-radius: 8px;
    background: rgba(6, 14, 24, 205);
}
QTabBar::tab {
    background: transparent;
    padding: 8px 16px;
    color: #6fa8bd;
}
QTabBar::tab:selected { color: #67f0ff; border-bottom: 2px solid #45d8ff; }
QGroupBox {
    border: 1px solid rgba(45, 150, 190, 70);
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #3fa7c9;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #2a7f9e;
    border-radius: 4px;
    background: #061019;
}
QCheckBox::indicator:checked { background: #20b8d8; border-color: #67f0ff; }
QScrollBar:vertical { background: #061019; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #1b5f78; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QMessageBox { background-color: #07131f; }
"""


# ---------------------------------------------------------------------------
# Анимированная сфера
# ---------------------------------------------------------------------------

class SphereWidget(QWidget):
    """Визуализатор голоса: ядро, орбиты, спектральные лучи и частицы.

    Живое поведение: лёгкое дыхание, случайные «вздохи», предметы-эмоции
    вокруг сферы (гаечный ключ, лупа, ноты, </> и т.д.) в зависимости от состояния.
    """

    sig_state = Signal(str)

    # Палитра под состояние: свечение, лучи, ядро, акцент.
    _STATE_COLORS = {
        "idle": ((0, 170, 220), (0, 210, 245), (120, 245, 255), (0, 255, 220)),
        "sleep": ((0, 60, 110), (0, 100, 160), (80, 150, 210), (0, 140, 190)),
        "listening": ((0, 195, 235), (0, 235, 255), (140, 255, 255), (0, 255, 220)),
        "thinking": ((105, 55, 220), (150, 100, 255), (205, 175, 255), (160, 120, 255)),
        "speaking": ((0, 205, 155), (0, 255, 195), (150, 255, 220), (0, 255, 180)),
        "confirm": ((230, 160, 0), (255, 200, 40), (255, 240, 180), (255, 200, 60)),
        "working": ((235, 110, 0), (255, 150, 30), (255, 220, 170), (255, 160, 40)),
        "happy": ((0, 225, 125), (0, 255, 155), (170, 255, 210), (0, 255, 140)),
        "error": ((220, 40, 60), (255, 70, 90), (255, 190, 190), (255, 80, 100)),
    }

    # Предметы-эмоции: орбитальные символы вокруг сферы.
    _STATE_OBJECTS = {
        "idle": ["◇", "○", "△"],                    # спокойные геометрические
        "sleep": ["z", "Z", "z"],                    # спит
        "listening": ["◉", "◉", "◉"],               # внимание (три «глаза»)
        "thinking": ["?", "?", "?"],                 # вопросы
        "speaking": ["♪", "♫", "♪"],                # ноты
        "confirm": ["!", "?", "!"],                  # подтверждение
        "working": ["⚙", "🔧", "⚙"],              # шестерёнки/гаечный ключ
        "happy": ["✦", "★", "✦"],                  # звёзды
        "error": ["✕", "⚠", "✕"],                 # ошибки
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._level = 0.0
        self._target = 0.0
        self._phase = 0.0
        self._state = "idle"

        # Живое поведение: дыхание, случайные вздохи.
        self._breath_phase = 0.0
        self._sigh_timer = 0
        self._sigh_value = 0.0

        # Голографическая сфера: координаты точек на единичной сфере.
        # Считаем один раз — за каждый кадр остаётся только поворот.
        self._holo_rings = self._build_holo_grid(lat_steps=9, lon_steps=22)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)  # ~60 FPS

    # ------------------------------------------------------------------
    @staticmethod
    def _build_holo_grid(lat_steps: int, lon_steps: int) -> list[list[tuple[float, float, float]]]:
        """Параллели единичной сферы: список колец из точек (x, y, z)."""
        rings: list[list[tuple[float, float, float]]] = []
        for index in range(1, lat_steps):
            lat = -math.pi / 2 + math.pi * index / lat_steps
            y = math.sin(lat)
            ring_r = math.cos(lat)
            # Шахматный сдвиг соседних параллелей: сетка читается объёмнее.
            shift = (index % 2) * (math.pi / lon_steps)
            ring = []
            for step in range(lon_steps):
                lon = 2.0 * math.pi * step / lon_steps + shift
                ring.append((ring_r * math.cos(lon), y, ring_r * math.sin(lon)))
            rings.append(ring)
        return rings

    def _holo_project(
        self, cx: float, cy: float, radius: float, phase: float, scale: float
    ) -> list[list[tuple[float, float, float]]]:
        """Поворачивает сетку и проецирует на экран: (px, py, глубина)."""
        angle = phase * 0.9
        ca, sa = math.cos(angle), math.sin(angle)
        # Наклон оси: сфера видна «слегка сверху», как глобус.
        tilt = 0.42 + 0.06 * math.sin(phase * 0.5)
        cb, sb = math.cos(tilt), math.sin(tilt)

        layout: list[list[tuple[float, float, float]]] = []
        for ring in self._holo_rings:
            projected = []
            for x, y, z in ring:
                x1 = x * ca + z * sa
                z1 = -x * sa + z * ca
                y2 = y * cb - z1 * sb
                z2 = y * sb + z1 * cb
                projected.append((cx + x1 * radius, cy - y2 * radius, z2))
            layout.append(projected)
        return layout

    def _draw_holo_layer(
        self,
        painter: QPainter,
        layout: list[list[tuple[float, float, float]]],
        radius: float,
        core_c: tuple[int, int, int],
        accent_c: tuple[int, int, int],
        level: float,
        near: bool,
        phase: float,
    ) -> None:
        """Рисует дальнюю (near=False) или ближнюю (near=True) половину сетки.

        Дальние линии тусклее и тоньше — сфера выглядит прозрачной
        голограммой, собранной из светящихся точек.
        """
        dot_size = radius * 0.042
        for ring_index, ring in enumerate(layout):
            total = len(ring)
            # --- линии параллелей ---
            for step in range(total):
                x1, y1, z1 = ring[step]
                x2, y2, z2 = ring[(step + 1) % total]
                depth = (z1 + z2) * 0.5
                if near != (depth >= 0.0):
                    continue
                front = (depth + 1.0) / 2.0
                alpha = int(22 + 120 * front * front + 70 * level * front)
                painter.setPen(
                    QPen(QColor(*core_c, max(12, min(255, alpha))), 0.7 + 1.1 * front)
                )
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # --- точки-сферы на пересечениях ---
            painter.setPen(Qt.NoPen)
            for x, y, depth in ring:
                front = (depth + 1.0) / 2.0
                if near != (depth >= 0.0):
                    continue
                alpha = int(35 + 190 * front * front + 40 * level * front)
                size = dot_size * (0.55 + 0.75 * front)
                painter.setBrush(QColor(*core_c, max(18, min(255, alpha))))
                painter.drawEllipse(QPointF(x, y), size, size)

        # Сканирующая полоса: яркое кольцо бежит сверху вниз (эффект голограммы).
        scan = (math.sin(phase * 0.8) + 1.0) / 2.0
        for ring_index, ring in enumerate(layout):
            position = ring_index / max(1, len(layout) - 1)
            closeness = 1.0 - min(1.0, abs(position - scan) * 4.5)
            if closeness <= 0.02:
                continue
            total = len(ring)
            for step in range(total):
                x1, y1, z1 = ring[step]
                x2, y2, z2 = ring[(step + 1) % total]
                depth = (z1 + z2) * 0.5
                front = (depth + 1.0) / 2.0
                alpha = int(200 * closeness * front)
                if alpha < 12:
                    continue
                painter.setPen(QPen(QColor(*accent_c, min(255, alpha)), 1.5))
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))


    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state not in self._STATE_COLORS:
            state = "idle"
        changed = state != self._state
        self._state = state
        if changed:
            try:
                self.sig_state.emit(state)
            except Exception:
                pass
        if state == "idle":
            self._target = 0.0
        elif state == "sleep":
            self._target = 0.0
        elif state == "thinking":
            self._target = 0.22
        elif state == "working":
            self._target = 0.35
        elif state == "listening":
            self._target = max(self._target, 0.08)
        elif state in {"happy", "confirm"}:
            self._target = max(self._target, 0.25)

    def set_level(self, value: float) -> None:
        self._target = max(0.0, min(1.0, float(value)))

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        try:
            speed = 0.055 if self._state == "working" else 0.035
            self._phase += speed
            self._breath_phase += 0.025

            # Случайный «вдох» каждые 4–8 секунд — сфера чуть раздувается.
            self._sigh_timer += 1
            if self._sigh_timer > 280:
                self._sigh_timer = 0
                self._sigh_value = 1.0
            self._sigh_value *= 0.985

            if self._state == "speaking":
                # Между словами/слогами амплитуда затухает, создавая «дыхание».
                self._target *= 0.88
                if self._target < 0.08:
                    self._target = 0.08 + 0.06 * abs(math.sin(self._phase * 3.0))
            elif self._state == "thinking":
                self._target = 0.18 + 0.10 * (0.5 + 0.5 * math.sin(self._phase * 2.2))
            elif self._state == "working":
                self._target = 0.30 + 0.16 * (0.5 + 0.5 * math.sin(self._phase * 4.5))
            elif self._state == "confirm":
                self._target = 0.20 + 0.12 * (0.5 + 0.5 * math.sin(self._phase * 3.2))
            elif self._state == "happy":
                self._target = 0.28 + 0.10 * abs(math.sin(self._phase * 2.6))
            elif self._state == "error":
                self._target = 0.16 + 0.10 * (0.5 + 0.5 * math.sin(self._phase * 5.0))
            elif self._state == "listening":
                if self._target < 0.05:
                    self._target = 0.05 + 0.04 * (0.5 + 0.5 * math.sin(self._phase * 1.6))
            elif self._state == "sleep":
                self._target = 0.02 + 0.02 * (0.5 + 0.5 * math.sin(self._phase * 0.8))
            else:
                self._target *= 0.96

            self._level += (self._target - self._level) * 0.30
            self.update()
        except Exception:
            log.exception("_tick failed")

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            width, height = self.width(), self.height()
            cx, cy = width / 2.0, height / 2.0
            base = min(width, height) * 0.255
            level = max(0.0, min(1.0, self._level))
            phase = self._phase
            glow_c, ray_c, core_c, accent_c = self._STATE_COLORS.get(
                self._state, self._STATE_COLORS["idle"]
            )

            # Внешнее свечение.
            glow_radius = base * (2.2 + 0.4 * level)
            glow = QRadialGradient(QPointF(cx, cy), glow_radius)
            glow.setColorAt(0.0, QColor(*glow_c, int(55 + 80 * level)))
            glow.setColorAt(0.45, QColor(*glow_c, int(24 + 34 * level)))
            glow.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(cx, cy), glow_radius, glow_radius)

            # Орбитальные кольца.
            for index, (scale, alpha, dashed) in enumerate(
                [(1.55, 75, False), (1.85, 45, True), (2.18, 24, False)]
            ):
                radius = base * scale
                pen = QPen(QColor(*ray_c, alpha), 1.1)
                if dashed:
                    pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(cx, cy), radius, radius)

                # Вращающаяся дуга на кольце.
                arc_radius = radius
                painter.setPen(QPen(QColor(*accent_c, 90 + int(60 * level)), 1.6))
                start = int(((phase * (18 + index * 9)) % 360) * 16)
                painter.drawArc(
                    QRectF(cx - arc_radius, cy - arc_radius, arc_radius * 2, arc_radius * 2),
                    start,
                    55 * 16,
                )

            # Спектральные лучи вокруг сферы.
            rays = 72
            inner = base * 1.10
            for index in range(rays):
                angle = 2.0 * math.pi * index / rays + phase * 0.18
                wave = 0.5 + 0.5 * math.sin(phase * 4.0 + index * 0.65)
                length = base * (0.035 + 0.62 * level) * (0.30 + 0.70 * wave)
                if self._state in {"idle", "sleep"}:
                    length *= 0.35
                x1 = cx + math.cos(angle) * inner
                y1 = cy + math.sin(angle) * inner
                x2 = cx + math.cos(angle) * (inner + length)
                y2 = cy + math.sin(angle) * (inner + length)
                color = QColor(*ray_c, int(45 + 150 * level))
                painter.setPen(QPen(color, 1.8))
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # --- 3D голографическая сфера -------------------------------
            # Сфера собрана из малых светящихся сфер-точек, лежащих на
            # поверхности и повёрнутых вокруг наклонной оси. Дальние точки
            # тусклее и меньше, ближние — крупнее и ярче: объём читается
            # без текстур. Ядро и лучи просвечивают сквозь сетку.
            radius = base * (1.0 + 0.14 * level + 0.012 * math.sin(phase * 2.0))
            layout = self._holo_project(cx, cy, radius, phase, 1.0)

            # Полупрозрачное «стекло» за сеткой: мягкий объём внутри.
            glass = QRadialGradient(
                QPointF(cx - radius * 0.30, cy - radius * 0.34), radius * 1.6
            )
            glass.setColorAt(0.0, QColor(*core_c, 46))
            glass.setColorAt(0.55, QColor(*accent_c, 22))
            glass.setColorAt(1.0, QColor(2, 10, 20, 120))
            painter.setPen(QPen(QColor(*core_c, 120), 1.4))
            painter.setBrush(glass)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

            # Дальняя половина голограммы (за ядром).
            self._draw_holo_layer(
                painter, layout, radius, core_c, accent_c, level, False, phase
            )

            # Внутренние вращающиеся дуги.
            painter.setPen(QPen(QColor(*accent_c, 150), 1.7))
            inner_radius = radius * 0.86
            rect = QRectF(
                cx - inner_radius, cy - inner_radius, inner_radius * 2, inner_radius * 2
            )
            painter.drawArc(rect, int(((phase * 45) % 360) * 16), 72 * 16)
            painter.drawArc(rect, int(((phase * 45 + 180) % 360) * 16), 42 * 16)

            # Яркое ядро.
            core_radius = radius * 0.24 * (1.0 + 0.30 * level)
            core = QRadialGradient(QPointF(cx, cy), core_radius * 2.4)
            core.setColorAt(0.0, QColor(255, 255, 255, 235))
            core.setColorAt(0.32, QColor(*core_c, 210))
            core.setColorAt(1.0, QColor(*core_c, 0))
            painter.setBrush(core)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cx, cy), core_radius * 2.4, core_radius * 2.4)

            # Ближняя половина голограммы (перед ядром) — даёт объём.
            self._draw_holo_layer(
                painter, layout, radius, core_c, accent_c, level, True, phase
            )

            # Блик на «стекле» сферы.
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 48))
            painter.drawEllipse(
                QPointF(cx - radius * 0.34, cy - radius * 0.42),
                radius * 0.22, radius * 0.14,
            )

            # Частицы на эллиптических орбитах.
            for index in range(20):
                speed = 0.35 + 0.12 * (index % 4)
                angle = phase * speed + index * (2.0 * math.pi / 20)
                orbit = base * (1.32 + 0.28 * ((index % 5) / 5.0))
                px = cx + math.cos(angle) * orbit
                py = cy + math.sin(angle) * orbit * 0.62
                size = 1.4 + 2.6 * level * (0.5 + 0.5 * math.sin(phase * 3 + index))
                painter.setBrush(QColor(*core_c, int(110 + 120 * level)))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(px, py), size, size)

            # --- Предметы-эмоции: символы на орбитах вокруг сферы ---
            objects = self._STATE_OBJECTS.get(self._state, [])
            if objects:
                from PySide6.QtGui import QFont

                icon_font = QFont("Segoe UI Symbol", 16)
                painter.setFont(icon_font)
                count = len(objects)
                for i, symbol in enumerate(objects):
                    # Равномерно распределяем предметы по кругу.
                    angle = phase * 0.4 + i * (2.0 * math.pi / count)
                    orbit_r = base * 1.95 + 8 * math.sin(phase * 2.0 + i)
                    ox = cx + math.cos(angle) * orbit_r
                    oy = cy + math.sin(angle) * orbit_r * 0.55
                    # Пульсация прозрачности.
                    alpha = int(120 + 100 * (0.5 + 0.5 * math.sin(phase * 3.0 + i * 1.2)))
                    painter.setPen(QColor(*core_c, alpha))
                    painter.drawText(QPointF(ox - 10, oy + 6), symbol)

            # --- «Дыхание»: лёгкое расширение ядра раз в несколько секунд ---
            breath_scale = 1.0 + 0.025 * math.sin(self._breath_phase)
            breath_scale += 0.06 * self._sigh_value
            breath_radius = radius * breath_scale
            breath_pen = QPen(QColor(*core_c, 40), 1.2)
            painter.setPen(breath_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), breath_radius, breath_radius)

            painter.end()
        except Exception:
            log.exception("paintEvent failed")


# ---------------------------------------------------------------------------
# Заголовок окна
# ---------------------------------------------------------------------------

class TitleBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TitleBar")
        self.setFixedHeight(46)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 10, 0)
        layout.setSpacing(10)

        dot = QLabel("●")
        dot.setStyleSheet("color:#28e0ff; font-size:13px;")
        title = QLabel("J.A.R.V.I.S.")
        title.setObjectName("TitleLabel")
        self.subtitle = QLabel("Just A Rather Very Intelligent System")
        self.subtitle.setObjectName("SubtitleLabel")

        layout.addWidget(dot)
        layout.addWidget(title)
        layout.addWidget(self.subtitle)
        layout.addStretch(1)

        minimize = QPushButton("—")
        minimize.setObjectName("WinButton")
        minimize.setFixedSize(38, 28)
        minimize.clicked.connect(lambda: self.window().showMinimized())

        maximize = QPushButton("□")
        maximize.setObjectName("WinButton")
        maximize.setFixedSize(38, 28)
        maximize.clicked.connect(self._toggle_maximize)

        close = QPushButton("✕")
        close.setObjectName("CloseButton")
        close.setFixedSize(38, 28)
        close.clicked.connect(lambda: self.window().close())

        layout.addWidget(minimize)
        layout.addWidget(maximize)
        layout.addWidget(close)

    def set_subtitle(self, text: str) -> None:
        self.subtitle.setText(text)

    def _toggle_maximize(self) -> None:
        window = self.window()
        if window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None:
                handle.startSystemMove()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._toggle_maximize()


# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    check_result = Signal(str)
    check_failed = Signal(str)
    voices_ready = Signal(object)
    voices_failed = Signal(str)
    mic_test_result = Signal(str)
    # Прогресс загрузки/запуска локальной модели: (этап, 0..1, подробности).
    local_progress = Signal(str, float, str)
    # Готово: (успех, сообщение).
    local_done = Signal(bool, str)
    # Ответ фонового потока со списком файлов репозитория.
    local_files_ready = Signal(object)

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = copy.deepcopy(settings)
        self._result = copy.deepcopy(settings)
        self._current_provider = settings.provider
        self._loading = True
        self._test_tts: TTSManager | None = None

        self.setWindowTitle("Настройки J.A.R.V.I.S.")
        # Фиксированный размер: окно не растянется за пределы экрана,
        # содержимое вкладок прокручивается (см. _scrollable).
        self.setFixedSize(880, 720)
        self.setStyleSheet(STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)

        heading = QLabel("КОНФИГУРАЦИЯ СИСТЕМЫ")
        heading.setObjectName("SectionLabel")
        layout.addWidget(heading)

        tabs = QTabWidget()
        tabs.setUsesScrollButtons(True)
        for builder, title in (
            (self._build_assistant_tab, "Ассистент"),
            (self._build_model_tab, "Модель ИИ"),
            (self._build_local_tab, "Локальные модели"),
            (self._build_voice_tab, "Голос"),
            (self._build_stt_tab, "Распознавание"),
            (self._build_avatar_tab, "Аватар"),
        ):
            tabs.addTab(self._scrollable(builder()), title)
        layout.addWidget(tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("Сохранить")
        buttons.button(QDialogButtonBox.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.check_result.connect(self._on_check_ok)
        self.check_failed.connect(self._on_check_failed)
        self.voices_ready.connect(self._on_voices_loaded)
        self.voices_failed.connect(self._on_voices_failed)
        self.mic_test_result.connect(self._on_mic_test_result)

        self._loading = False
        self._update_hint()

    # ------------------------------------------------------------------
    def _build_assistant_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(16, 18, 16, 18)
        form.setSpacing(12)

        self.name_edit = QLineEdit(self.settings.assistant_name)
        self.name_edit.setPlaceholderText("Джарвис, Пятница, Кейси...")
        form.addRow("Имя / wake word", self.name_edit)

        self.wake_check = QCheckBox("Реагировать на имя (wake word)")
        self.wake_check.setChecked(self.settings.wake_word_enabled)
        form.addRow("", self.wake_check)

        self.auto_listen_check = QCheckBox("Начинать слушать микрофон при запуске")
        self.auto_listen_check.setChecked(self.settings.auto_start_listening)
        form.addRow("", self.auto_listen_check)

        self.oneshot_check = QCheckBox(
            "Имя вместе с командой = одно действие, потом снова ждать имя"
        )
        self.oneshot_check.setChecked(self.settings.one_shot_mode)
        self.oneshot_check.setToolTip(
            "Включено: «Джарвис, открой браузер» — выполняет и замолкает.\n"
            "Выключено: после такой команды остаётся режим диалога.\n"
            "В обоих случаях достаточно сказать только имя — ассистент "
            "ответит и будет слушать команды без имени до слова «стоп»."
        )
        form.addRow("", self.oneshot_check)

        self.autostart_check = QCheckBox("Запускать ассистент вместе с системой")
        self.autostart_check.setChecked(is_autostart_enabled())
        form.addRow("", self.autostart_check)

        self.reply_language_combo = QComboBox()
        self.reply_language_combo.addItem("Как у пользователя", "auto")
        self.reply_language_combo.addItem("Всегда по-русски", "ru")
        self.reply_language_combo.addItem("Всегда по-английски (голос Джарвиса)", "en")
        index = self.reply_language_combo.findData(self.settings.reply_language)
        if index >= 0:
            self.reply_language_combo.setCurrentIndex(index)
        form.addRow("Язык ответов", self.reply_language_combo)

        self.confirm_check = QCheckBox("Спрашивать подтверждение перед опасными действиями")
        self.confirm_check.setChecked(False)
        self.confirm_check.setVisible(False)
        self.confirm_check.setToolTip(
            "Подтверждения отключены: агент выполняет действия без вопросов."
        )
        form.addRow("", self.confirm_check)

        self.shell_check = QCheckBox("Разрешить агенту выполнять shell-команды")
        self.shell_check.setChecked(self.settings.allow_shell)
        form.addRow("", self.shell_check)

        self.web_fallback_check = QCheckBox(
            "Если приложение не установлено — открывать его веб-версию"
        )
        self.web_fallback_check.setChecked(self.settings.web_fallback)
        form.addRow("Веб-фолбэк", self.web_fallback_check)

        self.vision_check = QCheckBox("Разрешить агенту смотреть на экран (зрение)")
        self.vision_check.setChecked(self.settings.vision_enabled)
        form.addRow("Зрение", self.vision_check)

        self.barge_in_check = QCheckBox(
            "Перебивание: замолкать, когда я начинаю говорить"
        )
        self.barge_in_check.setChecked(self.settings.barge_in)
        form.addRow("Barge-in", self.barge_in_check)

        self.barge_threshold_spin = QDoubleSpinBox()
        self.barge_threshold_spin.setRange(0.01, 0.5)
        self.barge_threshold_spin.setSingleStep(0.005)
        self.barge_threshold_spin.setDecimals(3)
        self.barge_threshold_spin.setValue(self.settings.barge_in_threshold)
        form.addRow("Порог перебивания", self.barge_threshold_spin)

        hint = QLabel(
            "Скажите имя ассистента, затем команду: «Джарвис, открой браузер». "
            "«Джарвис, отбой» — выключить микрофон. "
            "Если wake word выключен, любая распознанная речь считается командой."
        )
        hint.setWordWrap(True)
        hint.setObjectName("HintLabel")
        form.addRow("", hint)
        return page

    def _build_model_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(16, 18, 16, 18)
        outer.setSpacing(12)

        # --- Основная (разговорная) модель --------------------------------
        main_group = QGroupBox("Основная модель — разговор и команды")
        form = QFormLayout(main_group)
        form.setSpacing(10)

        self.provider_combo = QComboBox()
        for provider_id, config in PROVIDERS.items():
            self.provider_combo.addItem(config["label"], provider_id)
        index = self.provider_combo.findData(self.settings.provider)
        if index >= 0:
            self.provider_combo.setCurrentIndex(index)
        form.addRow("Провайдер", self.provider_combo)

        self.model_edit = QLineEdit(self.settings.model)
        form.addRow("Модель", self.model_edit)

        self.base_url_edit = QLineEdit(self.settings.resolved_base_url())
        form.addRow("Base URL", self.base_url_edit)

        self.api_key_edit = QLineEdit(self.settings.api_key())
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("API-ключ провайдера")
        form.addRow("API-ключ", self.api_key_edit)

        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setObjectName("HintLabel")
        form.addRow("", self.hint_label)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 1.5)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setValue(self.settings.temperature)
        form.addRow("Температура", self.temperature_spin)

        outer.addWidget(main_group)

        # --- Отдельная модель для кодинга ---------------------------------
        coding_group = QGroupBox("Модель для кодинга — можно локальную")
        coding_form = QFormLayout(coding_group)
        coding_form.setSpacing(10)

        self.coding_provider_combo = QComboBox()
        self.coding_provider_combo.addItem("Как основная (без отдельной)", "")
        for provider_id, config in PROVIDERS.items():
            self.coding_provider_combo.addItem(config["label"], provider_id)
        coding_index = self.coding_provider_combo.findData(
            self.settings.coding_provider or ""
        )
        if coding_index >= 0:
            self.coding_provider_combo.setCurrentIndex(coding_index)
        self._current_coding_provider = self.coding_provider_combo.currentData() or ""
        coding_form.addRow("Провайдер кодинга", self.coding_provider_combo)

        self.coding_model_edit = QLineEdit(self.settings.coding_model)
        self.coding_model_edit.setPlaceholderText(
            "Пусто = модель по умолчанию выбранного провайдера"
        )
        self.coding_model_edit.setToolTip(
            "Отдельная модель для кодинга (например, qwen2.5-coder:7b для Ollama "
            "или anthropic/claude-sonnet-4). Если пусто — берётся модель по "
            "умолчанию выбранного кодинг-провайдера."
        )
        coding_form.addRow("Модель", self.coding_model_edit)

        self.coding_base_url_edit = QLineEdit()
        coding_form.addRow("Base URL", self.coding_base_url_edit)

        self.coding_api_key_edit = QLineEdit()
        self.coding_api_key_edit.setEchoMode(QLineEdit.Password)
        self.coding_api_key_edit.setPlaceholderText("API-ключ провайдера кодинга")
        coding_form.addRow("API-ключ", self.coding_api_key_edit)

        self.coding_temperature_spin = QDoubleSpinBox()
        self.coding_temperature_spin.setRange(0.0, 1.5)
        self.coding_temperature_spin.setSingleStep(0.1)
        self.coding_temperature_spin.setDecimals(2)
        self.coding_temperature_spin.setValue(self.settings.coding_temperature)
        self.coding_temperature_spin.setToolTip(
            "Для кода обычно лучше низкая температура (0.1–0.3)."
        )
        coding_form.addRow("Температура", self.coding_temperature_spin)

        self.coding_hint_label = QLabel()
        self.coding_hint_label.setWordWrap(True)
        self.coding_hint_label.setObjectName("HintLabel")
        coding_form.addRow("", self.coding_hint_label)

        outer.addWidget(coding_group)

        # --- Общие параметры агента ---------------------------------------
        common_group = QGroupBox("Агент")
        common_form = QFormLayout(common_group)
        common_form.setSpacing(10)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(0, 500)
        self.steps_spin.setValue(self.settings.max_tool_steps)
        self.steps_spin.setSpecialValueText("∞ (без лимита)")
        self.steps_spin.setToolTip(
            "0 = без ограничений. "
            "При исчерпании лимита модель сама подведёт итог."
        )
        common_form.addRow("Макс. шагов агента", self.steps_spin)

        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(256, 8192)
        self.tokens_spin.setSingleStep(256)
        self.tokens_spin.setValue(self.settings.token_limit())
        self.tokens_spin.setToolTip(
            "Сколько токенов модель может потратить на ответ. OpenRouter "
            "резервирует кредиты под это число: слишком большое значение "
            "вызывает ошибку 402 «мало кредитов»."
        )
        common_form.addRow("Лимит токенов ответа", self.tokens_spin)

        self.check_button = QPushButton("Проверить подключение")
        self.check_button.clicked.connect(self._check_connection)
        common_form.addRow("", self.check_button)

        outer.addWidget(common_group)
        outer.addStretch(1)

        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.coding_provider_combo.currentIndexChanged.connect(
            self._on_coding_provider_changed
        )
        self._update_coding_fields()
        return page

    # ------------------------------------------------------------------
    # Локальные модели (Hugging Face + llama.cpp)
    # ------------------------------------------------------------------
    def _build_local_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        # --- Железо ----------------------------------------------------
        hw_group = QGroupBox("Ваш компьютер")
        hw_layout = QVBoxLayout(hw_group)
        self.local_hw_label = QLabel("Определяю оборудование...")
        self.local_hw_label.setWordWrap(True)
        hw_layout.addWidget(self.local_hw_label)
        refresh_hw = QPushButton("Обновить сведения о железе")
        refresh_hw.clicked.connect(self._refresh_local_hw)
        hw_layout.addWidget(refresh_hw)
        outer.addWidget(hw_group)

        # --- Каталог моделей -------------------------------------------
        cat_group = QGroupBox("Модели, которые потянет этот компьютер")
        cat_layout = QVBoxLayout(cat_group)

        filters = QHBoxLayout()
        self.local_show_all_check = QCheckBox("Показывать все (не только подходящие)")
        self.local_show_all_check.setChecked(bool(self.settings.local_show_all))
        self.local_show_all_check.stateChanged.connect(self._refresh_local_catalog)
        self.local_uncensored_check = QCheckBox("Только без цензуры")
        self.local_uncensored_check.setChecked(
            bool(self.settings.local_uncensored_only)
        )
        self.local_uncensored_check.stateChanged.connect(self._refresh_local_catalog)
        filters.addWidget(self.local_show_all_check)
        filters.addStretch(1)
        cat_layout.addLayout(filters)

        self.local_model_list = QListWidget()
        self.local_model_list.setMinimumHeight(220)
        self.local_model_list.currentItemChanged.connect(self._on_local_model_selected)
        cat_layout.addWidget(self.local_model_list)

        self.local_model_info = QLabel("Выберите модель из списка.")
        self.local_model_info.setWordWrap(True)
        cat_layout.addWidget(self.local_model_info)

        dl_row = QHBoxLayout()
        dl_row.addWidget(QLabel("Потоков загрузки"))
        self.local_threads_spin = QSpinBox()
        self.local_threads_spin.setRange(1, 16)
        self.local_threads_spin.setValue(int(self.settings.local_threads or 8))
        dl_row.addWidget(self.local_threads_spin)
        self.local_download_button = QPushButton("Скачать модель")
        self.local_download_button.clicked.connect(self._download_local_model)
        dl_row.addWidget(self.local_download_button)
        self.local_cancel_button = QPushButton("Отменить")
        self.local_cancel_button.setEnabled(False)
        self.local_cancel_button.clicked.connect(self._cancel_local_download)
        dl_row.addWidget(self.local_cancel_button)
        dl_row.addStretch(1)
        cat_layout.addLayout(dl_row)

        self.local_bar = QProgressBar()
        self.local_bar.setRange(0, 1000)
        self.local_bar.setValue(0)
        cat_layout.addWidget(self.local_bar)
        self.local_progress_label = QLabel("")
        self.local_progress_label.setWordWrap(True)
        cat_layout.addWidget(self.local_progress_label)
        outer.addWidget(cat_group)

        # --- Свой репозиторий Hugging Face ------------------------------
        custom_group = QGroupBox("Любая модель с Hugging Face")
        custom_form = QFormLayout(custom_group)
        custom_form.setSpacing(8)
        self.local_repo_edit = QLineEdit()
        self.local_repo_edit.setPlaceholderText(
            "например, bartowski/Qwen2.5-7B-Instruct-GGUF"
        )
        custom_form.addRow("Репозиторий", self.local_repo_edit)
        file_row = QHBoxLayout()
        self.local_files_combo = QComboBox()
        self.local_files_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.local_files_combo.setMinimumWidth(430)
        file_row.addWidget(self.local_files_combo, 1)
        self.local_list_files_button = QPushButton("Найти файлы")
        self.local_list_files_button.clicked.connect(self._list_repo_files)
        file_row.addWidget(self.local_list_files_button)
        custom_form.addRow("Файл GGUF", file_row)
        custom_btn_row = QHBoxLayout()
        self.local_custom_download_button = QPushButton("Скачать этот файл")
        self.local_custom_download_button.clicked.connect(self._download_custom_model)
        custom_btn_row.addWidget(self.local_custom_download_button)
        custom_btn_row.addStretch(1)
        custom_form.addRow("", custom_btn_row)
        note = QLabel(
            "Скачиваются и модели без цензуры — Джарвис не фильтрует репозитории. "
            "Загрузка идёт в несколько потоков кусками и продолжается после обрыва."
        )
        note.setWordWrap(True)
        custom_form.addRow(note)
        outer.addWidget(custom_group)

        # --- Запуск ------------------------------------------------------
        run_group = QGroupBox("Запуск выбранной модели")
        run_form = QFormLayout(run_group)
        run_form.setSpacing(8)
        self.local_path_edit = QLineEdit(self.settings.local_model_path)
        self.local_path_edit.setPlaceholderText("путь к .gguf после загрузки")
        browse = QPushButton("Обзор...")
        browse.clicked.connect(self._browse_local_model)
        path_row = QHBoxLayout()
        path_row.addWidget(self.local_path_edit, 1)
        path_row.addWidget(browse)
        run_form.addRow("Файл модели", path_row)

        self.local_ctx_spin = QSpinBox()
        self.local_ctx_spin.setRange(1024, 32768)
        self.local_ctx_spin.setSingleStep(1024)
        self.local_ctx_spin.setValue(int(self.settings.local_ctx or 4096))
        self.local_ctx_spin.setToolTip("Больше контекста — точнее память диалога, но нужнее память")
        run_form.addRow("Контекст (токенов)", self.local_ctx_spin)

        self.local_port_spin = QSpinBox()
        self.local_port_spin.setRange(1024, 65535)
        self.local_port_spin.setValue(int(self.settings.local_port or 8077))
        run_form.addRow("Порт сервера", self.local_port_spin)

        self.local_autostart_check = QCheckBox(
            "Запускать локальный сервер вместе с Джарвисом"
        )
        self.local_autostart_check.setChecked(bool(self.settings.local_autostart))
        run_form.addRow("", self.local_autostart_check)

        run_btn_row = QHBoxLayout()
        self.local_start_button = QPushButton("Запустить и подключить")
        self.local_start_button.clicked.connect(self._start_local_server)
        run_btn_row.addWidget(self.local_start_button)
        self.local_stop_button = QPushButton("Остановить сервер")
        self.local_stop_button.clicked.connect(self._stop_local_server)
        run_btn_row.addWidget(self.local_stop_button)
        self.local_ollama_button = QPushButton("Импортировать в Ollama")
        self.local_ollama_button.clicked.connect(self._import_to_ollama)
        run_btn_row.addWidget(self.local_ollama_button)
        run_btn_row.addStretch(1)
        run_form.addRow("", run_btn_row)

        self.local_status_label = QLabel("Сервер не запущен.")
        self.local_status_label.setWordWrap(True)
        run_form.addRow("", self.local_status_label)
        outer.addWidget(run_group)

        # --- Инструкции --------------------------------------------------
        help_group = QGroupBox("Скачать самому: Ollama и LM Studio")
        help_layout = QHBoxLayout(help_group)
        ollama_btn = QPushButton("Подробная инструкция: Ollama")
        ollama_btn.clicked.connect(
            lambda: self._show_howto("Ollama — установка и модели", local_models.OLLAMA_HOWTO)
        )
        help_layout.addWidget(ollama_btn)
        lm_btn = QPushButton("Подробная инструкция: LM Studio")
        lm_btn.clicked.connect(
            lambda: self._show_howto("LM Studio — установка и сервер", local_models.LMSTUDIO_HOWTO)
        )
        help_layout.addWidget(lm_btn)
        help_layout.addStretch(1)
        outer.addWidget(help_group)

        self.local_progress.connect(self._on_local_progress)
        self.local_done.connect(self._on_local_done)
        self.local_done.connect(self._on_local_done_server)
        self.local_files_ready.connect(self._repo_files_ready)
        QTimer.singleShot(120, self._refresh_local_hw)
        return page

    # ------------------------------------------------------------------
    def _show_howto(self, title: str, text: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(760, 640)
        layout = QVBoxLayout(dialog)
        view = QTextBrowser()
        view.setPlainText(text)
        view.setLineWrapMode(QTextBrowser.WidgetWidth)
        layout.addWidget(view, 1)
        close = QPushButton("Закрыть")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()

    def _refresh_local_hw(self) -> None:
        try:
            hw = local_models.detect_hardware()
            self._local_hw = hw
            self.local_hw_label.setText(
                f"Видеокарта: {hw.gpu_name or 'не найдена (CUDA недоступна)'} · "
                f"VRAM {hw.vram_gb:.1f} ГБ\n"
                f"Память: {hw.ram_gb:.1f} ГБ · Ядра CPU: {hw.cpu_cores}\n"
                f"Вывод: {'модель целиком в видеопамяти — самая быстрая работа' if hw.cuda_available and hw.vram_gb >= 6 else 'часть модели будет жить в оперативной памяти — отвечет медленнее'}"
            )
        except Exception as exc:
            self._local_hw = local_models.Hardware()
            self.local_hw_label.setText(f"Не удалось определить железо: {exc}")
        self._refresh_local_catalog()

    def _local_hw_info(self) -> local_models.Hardware:
        hw = getattr(self, "_local_hw", None)
        if hw is None:
            hw = local_models.detect_hardware()
            self._local_hw = hw
        return hw

    def _refresh_local_catalog(self, *_args) -> None:
        if not hasattr(self, "local_model_list"):
            return
        hw = self._local_hw_info()
        show_all = self.local_show_all_check.isChecked()
        only_uncensored = self.local_uncensored_check.isChecked()
        self._local_catalog = local_models.catalog_for(hw, show_all, only_uncensored)
        self.local_model_list.clear()
        for spec, fits, reason in self._local_catalog:
            mark = "" if fits else "  ⚠ не рекомендуется"
            tags = []
            if spec.uncensored:
                tags.append("без цензуры")
            if spec.vision:
                tags.append("зрение")
            if spec.coding:
                tags.append("кодинг")
            tag = f"  [{', '.join(tags)}]" if tags else ""
            item = QListWidgetItem(f"{spec.title}  ·  {spec.size_gb:.1f} ГБ{mark}{tag}")
            item.setData(Qt.UserRole, spec.key)
            self.local_model_list.addItem(item)
        if not self._local_catalog:
            self.local_model_list.addItem("Нет моделей под это железо — включите «Показывать все»")

    def _on_local_model_selected(self, current, _previous=None) -> None:
        if not current or not hasattr(self, "_local_catalog"):
            return
        key = current.data(Qt.UserRole)
        entry = next((row for row in self._local_catalog if row[0].key == key), None)
        if entry is None:
            self.local_model_info.setText("Выберите модель из списка.")
            return
        spec, fits, reason = entry
        hw = self._local_hw_info()
        lines = [
            f"<b>{spec.title}</b>",
            f"Реппозиторий: {spec.repo}",
            f"Размер файла: ~{spec.size_gb:.1f} ГБ · нужно памяти: ~{spec.size_gb * 1.15 + 2:.1f} ГБ "
            f"(у вас RAM {hw.ram_gb:.0f} ГБ, VRAM {hw.vram_gb:.0f} ГБ)",
        ]
        if spec.note:
            lines.append(spec.note)
        if reason:
            lines.append(("Подходит: " if fits else "Осторожно: ") + reason)
        self.local_model_info.setText("<br>".join(lines))

    # --- загрузка -------------------------------------------------------
    def _begin_local_task(self, title: str) -> None:
        self.local_bar.setValue(0)
        self.local_progress_label.setText(title)
        self.local_download_button.setEnabled(False)
        self.local_custom_download_button.setEnabled(False)
        self.local_cancel_button.setEnabled(True)
        self._local_cancel_event = threading.Event()

    def _end_local_task(self) -> None:
        self.local_download_button.setEnabled(True)
        self.local_custom_download_button.setEnabled(True)
        self.local_cancel_button.setEnabled(False)

    @staticmethod
    def _models_dir() -> Path:
        from config import MODELS_DIR

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return MODELS_DIR

    def _download_local_model(self) -> None:
        item = self.local_model_list.currentItem()
        if item is None or not hasattr(self, "_local_catalog"):
            QMessageBox.information(self, "Локальные модели", "Сначала выберите модель в списке.")
            return
        key = item.data(Qt.UserRole)
        entry = next((row for row in self._local_catalog if row[0].key == key), None)
        if entry is None:
            return
        spec = entry[0]
        self._begin_local_task(f"Ищу файл {spec.title}...")
        threads = int(self.local_threads_spin.value())
        cancel = self._local_cancel_event

        def worker() -> None:
            try:
                target = local_models.download_model(
                    spec,
                    self._models_dir(),
                    threads=threads,
                    on_progress=lambda stage, ratio, details: self.local_progress.emit(stage, ratio, details),
                    cancel=cancel,
                )
            except local_models.DownloadCancelled:
                self.local_done.emit(True, "Загрузка отменена.")
                return
            except Exception as exc:
                self.local_done.emit(False, f"Загрузка не удалась: {exc}")
                return
            self._local_downloaded_path = str(target)
            self.local_done.emit(True, f"Модель скачана: {target.name}")

        threading.Thread(target=worker, daemon=True, name="HFDownload").start()

    def _download_custom_model(self) -> None:
        repo = self.local_repo_edit.text().strip()
        filename = self.local_files_combo.currentText().strip()
        if not repo or not filename:
            QMessageBox.information(
                self, "Локальные модели", "Укажите репозиторий и выберите файл."
            )
            return
        self._begin_local_task(f"Качаю {repo}/{filename}...")
        threads = int(self.local_threads_spin.value())
        cancel = self._local_cancel_event

        def worker() -> None:
            try:
                target = local_models.download_repo_file(
                    repo,
                    filename,
                    self._models_dir(),
                    threads=threads,
                    on_progress=lambda stage, ratio, details: self.local_progress.emit(stage, ratio, details),
                    cancel=cancel,
                )
            except local_models.DownloadCancelled:
                self.local_done.emit(True, "Загрузка отменена.")
                return
            except Exception as exc:
                self.local_done.emit(False, f"Загрузка не удалась: {exc}")
                return
            self._local_downloaded_path = str(target)
            self.local_done.emit(True, f"Модель скачана: {target.name}")

        threading.Thread(target=worker, daemon=True, name="HFDownload").start()

    def _list_repo_files(self) -> None:
        repo = self.local_repo_edit.text().strip()
        if not repo:
            QMessageBox.information(self, "Локальные модели", "Введите репозиторий Hugging Face.")
            return
        self.local_list_files_button.setEnabled(False)
        self.local_list_files_button.setText("Ищу...")

        def worker() -> None:
            try:
                files = local_models.list_gguf_files(repo)
            except Exception as exc:
                self.local_done.emit(False, f"Не удалось прочитать репозиторий: {exc}")
                return
            self.local_files_ready.emit(files)

        threading.Thread(target=worker, daemon=True, name="HFList").start()

    def _repo_files_ready(self, files: list) -> None:
        self.local_list_files_button.setEnabled(True)
        self.local_list_files_button.setText("Найти файлы")
        self.local_files_combo.clear()
        for name, size in files:
            self.local_files_combo.addItem(f"{name}  ({size:.1f} ГБ)", name)
        if not files:
            QMessageBox.information(
                self, "Локальные модели", "В этом репозитории нет файлов .gguf"
            )

    def _cancel_local_download(self) -> None:
        event = getattr(self, "_local_cancel_event", None)
        if event is not None:
            event.set()
        self.local_progress_label.setText("Отменяю...")

    def _on_local_progress(self, stage: str, ratio: float, details: str) -> None:
        self.local_bar.setValue(int(ratio * 1000))
        self.local_progress_label.setText(f"{stage} · {details}" if details else stage)

    def _on_local_done(self, ok: bool, message: str) -> None:
        self._end_local_task()
        if ok:
            self.local_bar.setValue(1000)
            path = getattr(self, "_local_downloaded_path", "")
            if path:
                self.local_path_edit.setText(path)
                self.settings.local_model_path = path
        self.local_progress_label.setText(message)

    def _browse_local_model(self) -> None:
        start = str(self._models_dir())
        path, _ = QFileDialog.getOpenFileName(
            self, "Файл модели GGUF", start, "Модели GGUF (*.gguf);;Все файлы (*)"
        )
        if path:
            self.local_path_edit.setText(path)

    # --- запуск сервера --------------------------------------------------
    def _start_local_server(self) -> None:
        path = self.local_path_edit.text().strip()
        if not path:
            QMessageBox.information(
                self, "Локальные модели", "Сначала скачайте модель или укажите файл GGUF."
            )
            return
        if not Path(path).exists():
            QMessageBox.information(self, "Локальные модели", f"Файл не найден: {path}")
            return
        self.local_start_button.setEnabled(False)
        self.local_status_label.setText(
            "Запускаю... при первой загрузке скачивается llama.cpp "
            "(~100 МБ, один раз)."
        )
        hw = self._local_hw_info()
        ctx = int(self.local_ctx_spin.value())
        port = int(self.local_port_spin.value())

        def worker() -> None:
            try:
                server = local_models.get_server()
                server.port = port
                server.start(
                    path,
                    hw=hw,
                    ctx=ctx,
                    on_progress=lambda stage, ratio, details: self.local_progress.emit(stage, ratio, details),
                )
            except Exception as exc:
                self.local_done.emit(False, f"Сервер не запустился: {exc}")
                return
            self.local_done.emit(True, f"Локальная модель работает: {server.base_url}")

        threading.Thread(target=worker, daemon=True, name="LocalServer").start()

    def _stop_local_server(self) -> None:
        try:
            local_models.get_server().stop()
        except Exception as exc:
            log.debug("local server stop failed: %s", exc)
        self.local_status_label.setText("Сервер остановлен.")

    def _import_to_ollama(self) -> None:
        path = self.local_path_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.information(
                self, "Локальные модели", "Укажите скачанный файл модели."
            )
            return
        name = Path(path).stem.lower()
        self.local_ollama_button.setEnabled(False)
        self.local_status_label.setText("Импортирую в Ollama...")

        def worker() -> None:
            try:
                created = local_models.import_to_ollama(Path(path), name)
            except Exception as exc:
                self.local_done.emit(False, f"Ollama: {exc}")
                return
            self.local_done.emit(
                True,
                f"Модель добавлена в Ollama как «{created}». Выберите провайдера "
                "Ollama на вкладке «Модель ИИ» и укажите эту модель.",
            )

        threading.Thread(target=worker, daemon=True, name="OllamaImport").start()

    def _on_local_done_server(self, ok: bool, message: str) -> None:
        self.local_start_button.setEnabled(True)
        self.local_ollama_button.setEnabled(True)
        if ok:
            self.local_status_label.setText(message)
            self.local_server_running = True
        else:
            self.local_status_label.setText(message)

    def _collect_local(self, settings: Settings) -> None:
        settings.local_threads = int(self.local_threads_spin.value())
        settings.local_ctx = int(self.local_ctx_spin.value())
        settings.local_port = int(self.local_port_spin.value())
        settings.local_autostart = self.local_autostart_check.isChecked()
        settings.local_model_path = self.local_path_edit.text().strip()
        settings.local_uncensored_only = self.local_uncensored_check.isChecked()
        settings.local_show_all = self.local_show_all_check.isChecked()

    # ------------------------------------------------------------------
    def _scrollable(self, page: QWidget) -> QScrollArea:
        """Прокручиваемая вкладка: окно фиксировано, содержимое скроллится."""
        from PySide6.QtCore import Qt

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Ширина контента фиксирована: иначе поля сжимаются, а высота растёт
        # нелинейно и скролл «прыгает».
        page.setMinimumWidth(820)
        scroll.setWidget(page)
        return scroll

    def _build_voice_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        engine_form = QFormLayout()
        engine_form.setSpacing(8)
        self.tts_combo = QComboBox()
        for engine_id, label in TTS_ENGINES.items():
            self.tts_combo.addItem(label, engine_id)
        index = self.tts_combo.findData(self.settings.tts_engine)
        if index >= 0:
            self.tts_combo.setCurrentIndex(index)
        engine_form.addRow("Движок озвучки", self.tts_combo)
        outer.addLayout(engine_form)

        # --- Ускорение и стиль ------------------------------------------
        speed_group = QGroupBox("Скорость отклика и стиль")
        speed_layout = QVBoxLayout(speed_group)
        self.speak_reasoning_check = QCheckBox(
            "Озвучивать размышления модели во время долгих задач"
        )
        self.speak_reasoning_check.setChecked(bool(self.settings.speak_reasoning))
        self.speak_reasoning_check.setToolTip(
            "Джарвис вслух комментирует, что делает модель, и её короткие ремарки "
            "из ответов — не придётся молча ждать."
        )
        speed_layout.addWidget(self.speak_reasoning_check)
        self.voice_cache_check = QCheckBox("Кэшировать озвучку (повторы — мгновенно)")
        self.voice_cache_check.setChecked(bool(self.settings.voice_cache))
        self.voice_cache_check.setToolTip(
            "Одинаковые фразы синтезируются один раз и больше не зависят от сети."
        )
        speed_layout.addWidget(self.voice_cache_check)
        self.voice_prewarm_check = QCheckBox(
            "Готовить типовые фразы заранее (первый ответ быстрее)"
        )
        self.voice_prewarm_check.setChecked(bool(self.settings.voice_prewarm))
        speed_layout.addWidget(self.voice_prewarm_check)
        self.voice_fx_check = QCheckBox("Эффект «Джарвиса» (фильтр + лёгкое эхо)")
        self.voice_fx_check.setChecked(bool(self.settings.voice_fx))
        self.voice_fx_check.setToolTip(
            "Узкополосный фильтр и эхо поверх голоса. Заметно ближе к киноголосу, "
            "но чуть дольше обработка."
        )
        speed_layout.addWidget(self.voice_fx_check)
        outer.addWidget(speed_group)

        # --- Edge TTS ---------------------------------------------------
        edge_group = QGroupBox("EDGE TTS — нейроголос, бесплатно, нужен интернет")
        edge_form = QFormLayout(edge_group)
        edge_form.setSpacing(8)

        self.edge_voice_ru_edit = QComboBox()
        self.edge_voice_ru_edit.setEditable(True)
        for voice in ("ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural"):
            self.edge_voice_ru_edit.addItem(voice, voice)
        self.edge_voice_ru_edit.setCurrentText(self.settings.edge_voice_ru)
        edge_form.addRow("Русский голос", self.edge_voice_ru_edit)

        self.edge_voice_en_edit = QComboBox()
        self.edge_voice_en_edit.setEditable(True)
        for voice in (
            "en-GB-RyanNeural",
            "en-GB-ThomasNeural",
            "en-GB-SoniaNeural",
            "en-US-GuyNeural",
        ):
            self.edge_voice_en_edit.addItem(voice, voice)
        self.edge_voice_en_edit.setCurrentText(self.settings.edge_voice_en)
        edge_form.addRow("Английский голос", self.edge_voice_en_edit)

        rate_row = QWidget()
        rate_layout = QHBoxLayout(rate_row)
        rate_layout.setContentsMargins(0, 0, 0, 0)
        self.edge_rate_edit = QLineEdit(self.settings.edge_rate)
        self.edge_rate_edit.setPlaceholderText("-8%")
        self.edge_pitch_edit = QLineEdit(self.settings.edge_pitch)
        self.edge_pitch_edit.setPlaceholderText("-4Hz")
        rate_layout.addWidget(QLabel("темп"))
        rate_layout.addWidget(self.edge_rate_edit)
        rate_layout.addWidget(QLabel("тон"))
        rate_layout.addWidget(self.edge_pitch_edit)
        edge_form.addRow("Скорость / высота", rate_row)

        jarvis_row = QWidget()
        jarvis_layout = QHBoxLayout(jarvis_row)
        jarvis_layout.setContentsMargins(0, 0, 0, 0)
        self.jarvis_button = QPushButton("Пресет: голос Джарвиса (EN)")
        self.jarvis_button.setObjectName("AccentButton")
        self.jarvis_button.clicked.connect(self._apply_jarvis_preset)
        jarvis_layout.addWidget(self.jarvis_button)
        edge_form.addRow("", jarvis_row)

        edge_hint = QLabel(
            "Ryan/Thomas — британские мужские голоса, максимально близкие к манере "
            "Джарвиса. Русский текст читает Дмитрий. Для английского ответа просто "
            "задайте вопрос по-английски."
        )
        edge_hint.setWordWrap(True)
        edge_hint.setObjectName("HintLabel")
        edge_form.addRow("", edge_hint)
        outer.addWidget(edge_group)

        # --- Piper ------------------------------------------------------
        piper_group = QGroupBox("PIPER — нейроголос полностью офлайн")
        piper_form = QFormLayout(piper_group)
        piper_form.setSpacing(8)

        self.piper_voice_ru_edit = QComboBox()
        self.piper_voice_ru_edit.setEditable(True)
        self.piper_voice_en_edit = QComboBox()
        self.piper_voice_en_edit.setEditable(True)
        for voice, (_folder, lang) in PIPER_VOICES.items():
            target = self.piper_voice_ru_edit if lang == "ru" else self.piper_voice_en_edit
            target.addItem(voice, voice)
        self.piper_voice_ru_edit.setCurrentText(self.settings.piper_voice_ru)
        self.piper_voice_en_edit.setCurrentText(self.settings.piper_voice_en)
        piper_form.addRow("Русский голос", self.piper_voice_ru_edit)
        piper_form.addRow("Английский голос", self.piper_voice_en_edit)

        self.piper_length_spin = QDoubleSpinBox()
        self.piper_length_spin.setRange(0.8, 1.4)
        self.piper_length_spin.setSingleStep(0.02)
        self.piper_length_spin.setDecimals(2)
        self.piper_length_spin.setValue(self.settings.piper_length_scale)
        piper_form.addRow("Медлительность", self.piper_length_spin)

        piper_hint = QLabel(
            "Модель (~60 МБ) скачается при первом использовании и сохранится в "
            "assets/piper. После этого Piper работает без интернета."
        )
        piper_hint.setWordWrap(True)
        piper_hint.setObjectName("HintLabel")
        piper_form.addRow("", piper_hint)
        outer.addWidget(piper_group)

        # --- ElevenLabs -------------------------------------------------
        eleven_group = QGroupBox("ELEVENLABS — облако, клонирование голоса")
        eleven_form = QFormLayout(eleven_group)
        eleven_form.setSpacing(8)

        self.eleven_key_edit = QLineEdit(self.settings.elevenlabs_api_key)
        self.eleven_key_edit.setEchoMode(QLineEdit.Password)
        eleven_form.addRow("API-ключ", self.eleven_key_edit)

        voice_row = QWidget()
        voice_layout = QHBoxLayout(voice_row)
        voice_layout.setContentsMargins(0, 0, 0, 0)
        self.eleven_voice_combo = QComboBox()
        self.eleven_voice_combo.setEditable(True)
        self.eleven_voice_combo.addItem(self.settings.elevenlabs_voice_id)
        self.eleven_voice_combo.setCurrentText(self.settings.elevenlabs_voice_id)
        self.load_voices_button = QPushButton("Загрузить голоса")
        self.load_voices_button.clicked.connect(self._load_eleven_voices)
        voice_layout.addWidget(self.eleven_voice_combo, 1)
        voice_layout.addWidget(self.load_voices_button)
        eleven_form.addRow("Голос", voice_row)

        self.eleven_model_edit = QLineEdit(self.settings.elevenlabs_model)
        eleven_form.addRow("Модель", self.eleven_model_edit)

        eleven_hint = QLabel(
            "Голоса, похожие на Джарвиса: Brian (id nPczCjzI2devNBz1zQrb — глубокий, спокойный), "
            "George (id JBFqnCBsd6RMkjVDRZzb — британский, уверенный), "
            "Adam (id pNInz6obpgDQGcFmaJgB — низкий, ровный). "
            "Если API вернул «missing permission text_to_speech»: elevenlabs.io → "
            "API Keys → Edit → включите Text to Speech и Voices."
        )
        eleven_hint.setWordWrap(True)
        eleven_hint.setObjectName("HintLabel")
        eleven_form.addRow("", eleven_hint)
        outer.addWidget(eleven_group)

        # --- Системный голос -------------------------------------------
        system_group = QGroupBox("СИСТЕМНЫЙ ГОЛОС (резерв)")
        system_form = QFormLayout(system_group)
        system_form.setSpacing(8)
        self.voice_edit = QLineEdit(self.settings.tts_voice)
        self.voice_edit.setPlaceholderText("Часть имени голоса, пусто = авто")
        system_form.addRow("Голос pyttsx3", self.voice_edit)
        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(80, 300)
        self.rate_spin.setValue(self.settings.tts_rate)
        system_form.addRow("Скорость речи", self.rate_spin)
        self.silero_speaker_edit = QLineEdit(self.settings.silero_speaker)
        system_form.addRow("Silero speaker", self.silero_speaker_edit)
        outer.addWidget(system_group)

        test_row = QWidget()
        test_layout = QHBoxLayout(test_row)
        test_layout.setContentsMargins(0, 4, 0, 0)
        test_ru = QPushButton("Проверить голос (RU)")
        test_ru.clicked.connect(lambda: self._test_voice(english=False))
        test_en = QPushButton("Проверить голос (EN)")
        test_en.clicked.connect(lambda: self._test_voice(english=True))
        self.test_status_label = QLabel("")
        self.test_status_label.setObjectName("HintLabel")
        test_layout.addWidget(test_ru)
        test_layout.addWidget(test_en)
        test_layout.addWidget(self.test_status_label, 1)
        outer.addWidget(test_row)

        hint = QLabel(
            f"Готовые фразы для офлайн-команд: {OFFLINE_AUDIO_DIR} "
            "(volume_up.wav, shutdown.mp3 и т.д.)."
        )
        hint.setWordWrap(True)
        hint.setObjectName("HintLabel")
        outer.addWidget(hint)
        outer.addStretch(1)
        return self._scrollable(page)

    def _build_stt_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(16, 18, 16, 18)
        form.setSpacing(10)

        self.stt_combo = QComboBox()
        for engine_id, label in STT_ENGINES.items():
            self.stt_combo.addItem(label, engine_id)
        index = self.stt_combo.findData(self.settings.stt_engine)
        if index >= 0:
            self.stt_combo.setCurrentIndex(index)
        form.addRow("Движок STT", self.stt_combo)

        self.stt_model_edit = QLineEdit(self.settings.stt_model)
        self.stt_model_edit.setPlaceholderText("tiny / base / small / medium")
        form.addRow("Модель Faster-Whisper", self.stt_model_edit)

        self.stt_device_combo = QComboBox()
        self.stt_device_combo.addItem("CPU (рекомендуется)", "cpu")
        self.stt_device_combo.addItem("CUDA / GPU", "cuda")
        index = self.stt_device_combo.findData(self.settings.stt_device)
        if index >= 0:
            self.stt_device_combo.setCurrentIndex(index)
        form.addRow("Устройство", self.stt_device_combo)

        self.stt_compute_combo = QComboBox()
        self.stt_compute_combo.setEditable(True)
        for value in ("int8", "float16", "float32", "int8_float16"):
            self.stt_compute_combo.addItem(value, value)
        self.stt_compute_combo.setCurrentText(self.settings.stt_compute_type)
        form.addRow("Точность (compute_type)", self.stt_compute_combo)

        vosk_row = QWidget()
        vosk_layout = QHBoxLayout(vosk_row)
        vosk_layout.setContentsMargins(0, 0, 0, 0)
        self.vosk_path_edit = QLineEdit(self.settings.vosk_model_path)
        vosk_browse = QPushButton("Обзор...")
        vosk_browse.clicked.connect(self._browse_vosk)
        vosk_layout.addWidget(self.vosk_path_edit, 1)
        vosk_layout.addWidget(vosk_browse)
        form.addRow("Модель Vosk", vosk_row)

        self.language_edit = QLineEdit(self.settings.language)
        self.language_edit.setPlaceholderText("ru / en / auto")
        form.addRow("Язык", self.language_edit)

        self.device_combo = QComboBox()
        self.device_combo.setEditable(True)
        self.device_combo.addItem("Системный по умолчанию", "")
        self._fill_devices(self.settings.input_device)
        device_row = QWidget()
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        refresh_button = QPushButton("Обновить")
        refresh_button.setFixedWidth(90)
        refresh_button.clicked.connect(
            lambda: self._fill_devices(self.device_combo.currentData())
        )
        device_layout.addWidget(self.device_combo, 1)
        device_layout.addWidget(refresh_button)
        form.addRow("Микрофон", device_row)

        self.mic_gain_spin = QDoubleSpinBox()
        self.mic_gain_spin.setRange(0.5, 5.0)
        self.mic_gain_spin.setSingleStep(0.1)
        self.mic_gain_spin.setDecimals(1)
        self.mic_gain_spin.setValue(self.settings.mic_gain)
        form.addRow("Усиление микрофона", self.mic_gain_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.001, 0.2)
        self.threshold_spin.setSingleStep(0.001)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setValue(self.settings.silence_threshold)
        form.addRow("Порог тишины", self.threshold_spin)

        self.wake_beep_check = QCheckBox("Сигнал после имени ассистента")
        self.wake_beep_check.setChecked(self.settings.wake_beep)
        form.addRow("", self.wake_beep_check)

        self.mic_test_button = QPushButton("Проверить микрофон (4 секунды)")
        self.mic_test_button.clicked.connect(self._test_microphone)
        form.addRow("", self.mic_test_button)

        self.mic_test_label = QLabel("")
        self.mic_test_label.setWordWrap(True)
        self.mic_test_label.setObjectName("ValueLabel")
        form.addRow("", self.mic_test_label)

        hint = QLabel(
            "Faster-Whisper работает локально на CPU и не отправляет аудио в облако. "
            "Если речь не распознаётся: поднимите усиление микрофона, снизьте порог "
            "тишины и проверьте кнопкой выше."
        )
        hint.setWordWrap(True)
        hint.setObjectName("HintLabel")
        form.addRow("", hint)
        return page

    # ------------------------------------------------------------------
    def _build_avatar_tab(self) -> QWidget:
        """Вкладка настроек аватара с живым превью справа."""
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(14)

        # --- левая колонка: настройки ---
        left = QWidget()
        form = QFormLayout(left)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)

        self.avatar_enabled_check = QCheckBox("Показывать аватара поверх всех окон")
        self.avatar_enabled_check.setChecked(self.settings.avatar_enabled)
        form.addRow("", self.avatar_enabled_check)

        self.avatar_on_top_check = QCheckBox("Всегда поверх остальных окон")
        self.avatar_on_top_check.setChecked(self.settings.avatar_always_on_top)
        form.addRow("", self.avatar_on_top_check)

        self.avatar_wander_check = QCheckBox("Бегает по экрану")
        self.avatar_wander_check.setChecked(self.settings.avatar_wander)
        form.addRow("", self.avatar_wander_check)

        self.avatar_subtitles_check = QCheckBox("Показывать субтитры ответов")
        self.avatar_subtitles_check.setChecked(self.settings.avatar_subtitles)
        form.addRow("", self.avatar_subtitles_check)

        self.avatar_objects_check = QCheckBox("Предметы и эмоции (ключ, лупа, ноты)")
        self.avatar_objects_check.setChecked(self.settings.avatar_objects)
        form.addRow("", self.avatar_objects_check)

        self.avatar_scale_spin = QDoubleSpinBox()
        self.avatar_scale_spin.setRange(0.4, 2.5)
        self.avatar_scale_spin.setSingleStep(0.1)
        self.avatar_scale_spin.setDecimals(2)
        self.avatar_scale_spin.setValue(self.settings.avatar_scale)
        form.addRow("Размер", self.avatar_scale_spin)

        self.avatar_speed_spin = QDoubleSpinBox()
        self.avatar_speed_spin.setRange(0.2, 3.0)
        self.avatar_speed_spin.setSingleStep(0.1)
        self.avatar_speed_spin.setDecimals(2)
        self.avatar_speed_spin.setValue(self.settings.avatar_speed)
        form.addRow("Скорость бега", self.avatar_speed_spin)

        self.avatar_opacity_spin = QDoubleSpinBox()
        self.avatar_opacity_spin.setRange(0.2, 1.0)
        self.avatar_opacity_spin.setSingleStep(0.05)
        self.avatar_opacity_spin.setDecimals(2)
        self.avatar_opacity_spin.setValue(self.settings.avatar_opacity)
        form.addRow("Прозрачность", self.avatar_opacity_spin)

        self.avatar_accent_combo = QComboBox()
        self.avatar_accent_combo.addItem("По состоянию (авто)", "")
        self.avatar_accent_combo.addItem("Неоновый голубой", "#28e0ff")
        self.avatar_accent_combo.addItem("Бирюзовый", "#00ffc3")
        self.avatar_accent_combo.addItem("Оранжевый", "#ff9a1f")
        self.avatar_accent_combo.addItem("Фиолетовый", "#b48cff")
        self.avatar_accent_combo.addItem("Зелёный", "#4dff88")
        index = self.avatar_accent_combo.findData(self.settings.avatar_accent or "")
        self.avatar_accent_combo.setCurrentIndex(max(0, index))
        form.addRow("Цвет", self.avatar_accent_combo)

        self.avatar_preview_state = QComboBox()
        for state_id, label in (
            ("idle", "Ожидание"),
            ("listening", "Слушает"),
            ("thinking", "Думает"),
            ("working", "Работает"),
            ("coding", "Пишет код"),
            ("speaking", "Говорит"),
            ("happy", "Радуется"),
            ("error", "Ошибка"),
            ("sleep", "Спит"),
        ):
            self.avatar_preview_state.addItem(label, state_id)
        form.addRow("Превью эмоции", self.avatar_preview_state)

        hint = QLabel(
            "Аватар — отдельное окно поверх всех программ. Его можно перетащить "
            "мышью. Предметы появляются автоматически: гаечный ключ — работа, "
            "лупа — поиск, ноты — речь, </> — кодинг."
        )
        hint.setWordWrap(True)
        hint.setMaximumWidth(360)
        hint.setObjectName("HintLabel")
        form.addRow("", hint)

        outer.addWidget(left, 1)

        # --- правая колонка: живое превью ---
        self.avatar_preview = AvatarPreview()
        self.avatar_preview.setFixedWidth(200)
        self.avatar_preview.set_state(
            self.avatar_preview_state.currentData() or "idle"
        )
        self.avatar_preview.set_objects(self.settings.avatar_objects)
        self.avatar_preview.set_accent(self.settings.avatar_accent or "")
        outer.addWidget(self.avatar_preview, 0)

        # Превью реагирует на изменения настроек сразу.
        self.avatar_preview_state.currentIndexChanged.connect(
            lambda _i: self.avatar_preview.set_state(
                self.avatar_preview_state.currentData() or "idle"
            )
        )
        self.avatar_objects_check.toggled.connect(self.avatar_preview.set_objects)
        self.avatar_accent_combo.currentIndexChanged.connect(
            lambda _i: self.avatar_preview.set_accent(
                self.avatar_accent_combo.currentData() or ""
            )
        )
        return page

    # ------------------------------------------------------------------
    def _fill_devices(self, selected: str = "") -> None:
        """Заполняет список микрофонов, сохраняя текущий выбор."""
        current = selected or self.device_combo.currentData() or ""
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("Системный по умолчанию", "")
        devices = list_input_devices()
        for index, name in devices:
            # Храним имя: индексы меняются при подключении новых устройств.
            self.device_combo.addItem(f"{name} (#{index})", name)
        target = str(current)
        position = self.device_combo.findData(target)
        if position < 0 and target:
            # Старое значение могло быть частью имени устройства.
            for i in range(self.device_combo.count()):
                if target.lower() in self.device_combo.itemText(i).lower():
                    position = i
                    break
        self.device_combo.setCurrentIndex(max(0, position))
        self.device_combo.blockSignals(False)

    def _browse_vosk(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выберите папку модели Vosk")
        if path:
            self.vosk_path_edit.setText(path)

    def _apply_jarvis_preset(self) -> None:
        """Британский нейроголос + замедленный темп = манера Джарвиса."""
        index = self.tts_combo.findData("edge")
        if index >= 0:
            self.tts_combo.setCurrentIndex(index)
        self.edge_voice_en_edit.setCurrentText("en-GB-RyanNeural")
        self.edge_rate_edit.setText("-10%")
        self.edge_pitch_edit.setText("-6Hz")
        # Чтобы британский голос звучал и на русские вопросы, включаем ответы на English.
        index = self.reply_language_combo.findData("en")
        if index >= 0:
            self.reply_language_combo.setCurrentIndex(index)
        self.test_status_label.setText("Пресет Джарвиса применён — слушаю английский пример.")
        self._test_voice(english=True)

    def _load_eleven_voices(self) -> None:
        key = self.eleven_key_edit.text().strip()
        self.load_voices_button.setEnabled(False)
        self.load_voices_button.setText("Загружаю...")

        def worker() -> None:
            try:
                voices = fetch_elevenlabs_voices(key)
            except Exception as exc:
                self.voices_failed.emit(str(exc))
            else:
                self.voices_ready.emit(voices)

        threading.Thread(target=worker, daemon=True).start()

    def _on_voices_loaded(self, voices: object) -> None:
        self.load_voices_button.setEnabled(True)
        self.load_voices_button.setText("Загрузить голоса")
        current = self.eleven_voice_combo.currentText().strip()
        self.eleven_voice_combo.clear()
        if not isinstance(voices, list) or not voices:
            self.eleven_voice_combo.addItem(current)
            QMessageBox.information(self, "Голоса ElevenLabs", "Список голосов пуст.")
            return
        for voice in voices:
            label = voice.get("name", "Без имени")
            if voice.get("labels"):
                label += f" — {voice['labels']}"
            self.eleven_voice_combo.addItem(label, voice.get("id", ""))
        # Восстанавливаем ранее выбранный voice_id, если он есть в списке.
        for index in range(self.eleven_voice_combo.count()):
            if self.eleven_voice_combo.itemData(index) == current:
                self.eleven_voice_combo.setCurrentIndex(index)
                break
        self.eleven_voice_combo.setEditText(current)
        QMessageBox.information(
            self, "Голоса ElevenLabs", f"Загружено голосов: {len(voices)}."
        )

    def _on_voices_failed(self, message: str) -> None:
        self.load_voices_button.setEnabled(True)
        self.load_voices_button.setText("Загрузить голоса")
        QMessageBox.warning(self, "ElevenLabs", message)

    # ------------------------------------------------------------------
    def _test_voice(self, english: bool) -> None:
        settings = self._collect()
        text = (
            "Good evening, sir. All systems are operational."
            if english
            else "Добрый вечер, сэр. Все системы в норме."
        )
        if self._test_tts is not None:
            self._test_tts.stop()
        self.test_status_label.setText("Синтезирую...")
        self._test_tts = TTSManager(
            settings,
            on_status=lambda message: self.test_status_label.setText(message),
        )
        if not self._test_tts.available:
            self.test_status_label.setText("Озвучка недоступна")
            return
        self._test_tts.speak(
            text,
            on_start=lambda: self.test_status_label.setText(
                f"Говорю: {self._test_tts.label if self._test_tts else ''}"
            ),
            on_end=lambda: self.test_status_label.setText("Готово"),
        )

    def _test_microphone(self) -> None:
        settings = self._collect()
        self.mic_test_button.setEnabled(False)
        self.mic_test_button.setText("Записываю 4 секунды...")
        self.mic_test_label.setText("Говорите: «Джарвис, открой браузер»")

        def worker() -> None:
            try:
                import sounddevice as sd

                from speech import SAMPLE_RATE, _prepare_audio

                stt = create_stt(settings)
                device = resolve_input_device(settings.input_device)
                recording = sd.rec(
                    int(SAMPLE_RATE * 4),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=device,
                )
                sd.wait()
                audio = _prepare_audio(recording[:, 0], settings.mic_gain)
                text = (stt.transcribe(audio) or "").strip()
                self.mic_test_result.emit(text or "(речь не распознана — тишина или шум)")
            except Exception as exc:
                self.mic_test_result.emit(f"Ошибка: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_mic_test_result(self, message: str) -> None:
        self.mic_test_button.setEnabled(True)
        self.mic_test_button.setText("Проверить микрофон (4 секунды)")
        self.mic_test_label.setText(f"Услышано: {message}")

    def done(self, result: int) -> None:  # noqa: N802
        if self._test_tts is not None:
            self._test_tts.stop()
        super().done(result)

    def _on_provider_changed(self, _index: int) -> None:
        if self._loading:
            return
        old = self._current_provider
        if old:
            self.settings.api_keys[old] = self.api_key_edit.text().strip()
        # Сохраняем введённые данные кодинг-провайдера, чтобы не потерять их.
        self._save_coding_fields()

        provider_id = self.provider_combo.currentData()
        self._current_provider = provider_id
        config = PROVIDERS[provider_id]
        self.base_url_edit.setText(
            self.settings.base_urls.get(provider_id) or config["base_url"]
        )
        self.model_edit.setText(config["default_model"])
        self.api_key_edit.setText(self.settings.api_keys.get(provider_id, ""))
        self.api_key_edit.setEnabled(bool(config.get("needs_key")))
        self._update_hint()
        self._update_coding_fields()

    def _update_hint(self) -> None:
        provider_id = self.provider_combo.currentData()
        config = PROVIDERS.get(provider_id, {})
        self.hint_label.setText(str(config.get("hint", "")))

    # --- кодинг-провайдер ---------------------------------------------
    def _on_coding_provider_changed(self, _index: int) -> None:
        if self._loading:
            return
        self._save_coding_fields()
        self._current_coding_provider = self.coding_provider_combo.currentData() or ""
        self._update_coding_fields()

    def _save_coding_fields(self) -> None:
        """Запоминает Base URL и ключ выбранного кодинг-провайдера."""
        cid = self.coding_provider_combo.currentData() or ""
        main_id = self.provider_combo.currentData()
        if not cid or cid == main_id:
            return
        self.settings.base_urls[cid] = self.coding_base_url_edit.text().strip()
        if self.coding_api_key_edit.isEnabled():
            self.settings.api_keys[cid] = self.coding_api_key_edit.text().strip()

    def _update_coding_fields(self) -> None:
        """Показывает поля кодинг-модели в зависимости от выбора провайдера."""
        cid = self.coding_provider_combo.currentData() or ""
        main_id = self.provider_combo.currentData()
        separate = bool(cid) and cid != main_id
        config = PROVIDERS.get(cid or main_id, {})

        self.coding_model_edit.setEnabled(bool(cid))
        self.coding_temperature_spin.setEnabled(bool(cid))
        self.coding_base_url_edit.setEnabled(separate)
        self.coding_api_key_edit.setEnabled(
            separate and bool(config.get("needs_key"))
        )

        if not cid:
            self.coding_model_edit.clear()
            self.coding_base_url_edit.clear()
            self.coding_api_key_edit.clear()
            self.coding_model_edit.setPlaceholderText(
                "не используется — кодинг идёт через основную модель"
            )
            self.coding_hint_label.setText(
                "Отдельная модель не выбрана: задачи по коду выполняет основная "
                "модель."
            )
            return

        if not separate:
            # Тот же провайдер, что и основной: ключ и адрес общие.
            self.coding_base_url_edit.setText(self.base_url_edit.text())
            self.coding_api_key_edit.setText(self.api_key_edit.text())
            self.coding_model_edit.setPlaceholderText(
                f"Пусто = {self.model_edit.text().strip() or config.get('default_model', '')}"
            )
            self.coding_hint_label.setText(
                "Провайдер совпадает с основным — ключ и Base URL общие, "
                "меняется только модель."
            )
            return

        self.coding_base_url_edit.setText(
            self.settings.base_urls.get(cid) or config.get("base_url", "")
        )
        self.coding_api_key_edit.setText(self.settings.api_keys.get(cid, ""))
        self.coding_model_edit.setPlaceholderText(
            f"Пусто = {config.get('default_model', '')}"
        )
        hint = str(config.get("hint", ""))
        if cid in {"ollama", "lmstudio"}:
            hint += (
                "\nКодинг будет считаться локально — интернет и кредиты не нужны."
            )
        self.coding_hint_label.setText(hint)

    # ------------------------------------------------------------------
    def _collect(self) -> Settings:
        settings = copy.deepcopy(self.settings)
        settings.assistant_name = self.name_edit.text().strip() or "Джарвис"
        settings.wake_word_enabled = self.wake_check.isChecked()
        settings.auto_start_listening = self.auto_listen_check.isChecked()
        settings.one_shot_mode = self.oneshot_check.isChecked()
        settings.autostart = self.autostart_check.isChecked()
        settings.reply_language = self.reply_language_combo.currentData() or "auto"
        settings.confirm_dangerous = self.confirm_check.isChecked()
        settings.allow_shell = self.shell_check.isChecked()
        settings.web_fallback = self.web_fallback_check.isChecked()
        settings.vision_enabled = self.vision_check.isChecked()
        settings.barge_in = self.barge_in_check.isChecked()
        settings.barge_in_threshold = float(self.barge_threshold_spin.value())

        provider_id = self.provider_combo.currentData()
        settings.provider = provider_id
        settings.model = self.model_edit.text().strip()
        settings.base_urls[provider_id] = self.base_url_edit.text().strip()
        if self.api_key_edit.isEnabled():
            settings.api_keys[provider_id] = self.api_key_edit.text().strip()
        settings.temperature = float(self.temperature_spin.value())
        settings.max_tool_steps = int(self.steps_spin.value())
        settings.max_tokens = int(self.tokens_spin.value())

        # --- отдельная кодинг-модель -----------------------------------
        coding_id = self.coding_provider_combo.currentData() or ""
        settings.coding_provider = coding_id
        settings.coding_model = self.coding_model_edit.text().strip()
        settings.coding_temperature = float(self.coding_temperature_spin.value())
        if coding_id and coding_id != provider_id:
            settings.base_urls[coding_id] = self.coding_base_url_edit.text().strip()
            if self.coding_api_key_edit.isEnabled():
                settings.api_keys[coding_id] = self.coding_api_key_edit.text().strip()

        settings.tts_engine = self.tts_combo.currentData()
        settings.tts_voice = self.voice_edit.text().strip()
        settings.tts_rate = int(self.rate_spin.value())
        settings.edge_voice_ru = self.edge_voice_ru_edit.currentText().strip()
        settings.edge_voice_en = self.edge_voice_en_edit.currentText().strip()
        settings.edge_rate = self.edge_rate_edit.text().strip() or "-8%"
        settings.edge_pitch = self.edge_pitch_edit.text().strip() or "-4Hz"
        settings.piper_voice_ru = self.piper_voice_ru_edit.currentText().strip()
        settings.piper_voice_en = self.piper_voice_en_edit.currentText().strip()
        settings.piper_length_scale = float(self.piper_length_spin.value())
        settings.silero_speaker = self.silero_speaker_edit.text().strip() or "aidar"
        settings.elevenlabs_api_key = self.eleven_key_edit.text().strip()
        voice_id = self.eleven_voice_combo.currentData()
        if not voice_id:
            voice_id = self.eleven_voice_combo.currentText().strip()
        settings.elevenlabs_voice_id = str(voice_id or "").strip()
        settings.elevenlabs_model = (
            self.eleven_model_edit.text().strip() or "eleven_multilingual_v2"
        )

        settings.stt_engine = self.stt_combo.currentData()
        settings.stt_model = self.stt_model_edit.text().strip() or "small"
        settings.stt_device = self.stt_device_combo.currentData() or "cpu"
        settings.stt_compute_type = self.stt_compute_combo.currentText().strip() or "int8"
        settings.vosk_model_path = self.vosk_path_edit.text().strip()
        settings.language = self.language_edit.text().strip() or "ru"
        settings.input_device = self.device_combo.currentData() or ""
        if not settings.input_device:
            settings.input_device = self.device_combo.currentText().strip()
            if settings.input_device == "Системный по умолчанию":
                settings.input_device = ""
        settings.mic_gain = float(self.mic_gain_spin.value())
        settings.silence_threshold = float(self.threshold_spin.value())
        settings.wake_beep = self.wake_beep_check.isChecked()

        # --- аватар ---
        settings.avatar_enabled = self.avatar_enabled_check.isChecked()
        settings.avatar_always_on_top = self.avatar_on_top_check.isChecked()
        settings.avatar_wander = self.avatar_wander_check.isChecked()
        settings.avatar_subtitles = self.avatar_subtitles_check.isChecked()
        settings.avatar_objects = self.avatar_objects_check.isChecked()
        settings.avatar_scale = float(self.avatar_scale_spin.value())
        settings.avatar_speed = float(self.avatar_speed_spin.value())
        settings.avatar_opacity = float(self.avatar_opacity_spin.value())
        settings.avatar_accent = self.avatar_accent_combo.currentData() or ""

        # --- локальные модели ---
        self._collect_local(settings)

        # --- озвучка размышлений и ускорение голоса ---
        settings.speak_reasoning = self.speak_reasoning_check.isChecked()
        settings.voice_cache = self.voice_cache_check.isChecked()
        settings.voice_fx = self.voice_fx_check.isChecked()
        settings.voice_prewarm = self.voice_prewarm_check.isChecked()
        settings.confirm_dangerous = False
        return settings

    def result_settings(self) -> Settings:
        return self._result

    def accept(self) -> None:  # noqa: N802
        self._result = self._collect()
        super().accept()

    # ------------------------------------------------------------------
    def _check_connection(self) -> None:
        settings = self._collect()
        self.check_button.setEnabled(False)
        self.check_button.setText("Проверяю...")

        def worker() -> None:
            try:
                tools = ToolRegistry(settings)
                result = AIManager(settings, tools).check()
                self.check_result.emit(result)
            except Exception as exc:
                self.check_failed.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_check_ok(self, message: str) -> None:
        self.check_button.setEnabled(True)
        self.check_button.setText("Проверить подключение")
        QMessageBox.information(self, "Проверка подключения", message)

    def _on_check_failed(self, message: str) -> None:
        self.check_button.setEnabled(True)
        self.check_button.setText("Проверить подключение")
        QMessageBox.warning(self, "Ошибка подключения", message)


# ---------------------------------------------------------------------------
# Диалог истории чатов
# ---------------------------------------------------------------------------

class HistoryDialog(QDialog):
    """Список сохранённых диалогов с возможностью загрузки/удаления."""

    sig_load_session = Signal(str)  # session_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("История диалогов")
        self.setMinimumSize(520, 480)

        layout = QVBoxLayout(self)

        # --- Список сессий ---
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        # --- Информационная панель ---
        self.info_label = QLabel("Выберите диалог из списка")
        self.info_label.setWordWrap(True)
        self.info_label.setObjectName("HintLabel")
        layout.addWidget(self.info_label)

        # --- Кнопки ---
        btn_row = QHBoxLayout()
        self.new_button = QPushButton("Новый диалог")
        self.new_button.setObjectName("AccentButton")
        self.load_button = QPushButton("Открыть выбранный")
        self.load_button.setEnabled(False)
        self.delete_button = QPushButton("Удалить")
        self.delete_button.setEnabled(False)

        self.new_button.clicked.connect(self._on_new)
        self.load_button.clicked.connect(self._on_load)
        self.delete_button.clicked.connect(self._on_delete)

        btn_row.addWidget(self.new_button)
        btn_row.addWidget(self.load_button)
        btn_row.addWidget(self.delete_button)
        layout.addLayout(btn_row)

        self._sessions: list[dict] = []
        self.refresh()

    def refresh(self) -> None:
        """Загружает список сессий из файлов."""
        self.list_widget.clear()
        self._sessions = list_sessions()

        for session in self._sessions:
            title = session.get("title", "Без названия")
            updated = session.get("updated_at", "")
            count = session.get("message_count", 0)
            if updated:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    updated_str = dt.strftime("%d.%m.%Y %H:%M")
                except Exception:
                    updated_str = updated[:10]
            else:
                updated_str = ""

            text = f"{title}"
            if updated_str:
                text += f"\n   {updated_str} · {count} сообщений"

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, session["id"])
            self.list_widget.addItem(item)

    def _on_selection_changed(self, current, _previous) -> None:
        if current is None:
            self.info_label.setText("Выберите диалог из списка")
            self.load_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            return

        session_id = current.data(Qt.UserRole)
        session = load_session(session_id)
        if session:
            title = session.get("title", "Без названия")
            created = session.get("created_at", "")
            count = len(session.get("messages", []))
            self.info_label.setText(
                f"{title}\nСоздан: {created[:16] if created else '—'}\n"
                f"Сообщений: {count}"
            )
        self.load_button.setEnabled(True)
        self.delete_button.setEnabled(True)

    def _on_item_double_clicked(self, item) -> None:
        self._on_load()

    def _on_new(self) -> None:
        self.sig_load_session.emit("__new__")
        self.accept()

    def _on_load(self) -> None:
        current = self.list_widget.currentItem()
        if current is None:
            return
        session_id = current.data(Qt.UserRole)
        if session_id:
            self.sig_load_session.emit(session_id)
            self.accept()

    def _on_delete(self) -> None:
        current = self.list_widget.currentItem()
        if current is None:
            return
        session_id = current.data(Qt.UserRole)
        if session_id:
            reply = QMessageBox.question(
                self,
                "Удалить диалог?",
                "Вы уверены, что хотите удалить этот диалог? Это действие нельзя отменить.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                delete_session(session_id)
                self.refresh()
                self.info_label.setText("Выберите диалог из списка")
                self.load_button.setEnabled(False)
                self.delete_button.setEnabled(False)


# ---------------------------------------------------------------------------
# Главное окно
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    sig_ai_result = Signal(str)
    sig_ai_error = Signal(str)
    sig_tool_event = Signal(str)
    sig_tts_level = Signal(float)
    sig_tts_started = Signal()
    sig_tts_finished = Signal()
    sig_stt_level = Signal(float)
    sig_stt_status = Signal(str)
    sig_stt_error = Signal(str)
    sig_heard = Signal(str)
    sig_wake = Signal()
    sig_command = Signal(str)
    sig_confirm = Signal(str, str)
    sig_stt_ready = Signal(object)
    sig_barge_in = Signal()
    sig_session_start = Signal(str)
    sig_session_end = Signal(str)
    sig_artifact = Signal(object)
    sig_confirm_retry = Signal()

    def __init__(self, no_audio: bool = False) -> None:
        super().__init__()
        self.no_audio = no_audio or os.environ.get("JARVIS_DISABLE_AUDIO") == "1"
        self.settings = Settings.load()
        ensure_dirs()

        self._stt = None
        self._listener: VoiceListener | None = None
        self._busy = False
        self._ignore_answer = False
        self._start_after = False
        self._session_active = False
        # Одноразовый режим: имя → одна команда → снова ждём имя.
        self._oneshot_active = False
        # Ручной запуск «Слушать»: тогда статусы показываем даже без имени.
        self._manual_listen = False
        # Текст, который сейчас озвучивается (для синхронного субтитра).
        self._pending_speech = ""
        self._confirm_event = threading.Event()
        self._confirm_result = False

        # --- История диалогов ---
        self._current_session: dict | None = None
        # --- Аватар (создаётся позже, чтобы сигналы сферы не падали) ---
        self.avatar: AvatarWindow | None = None

        # «Не молчать во время работы»: короткая реплика каждые 25-40 секунд.
        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(25000)
        self._activity_timer.timeout.connect(self._safe(self._activity_tick))

        self.setWindowTitle("J.A.R.V.I.S.")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setMinimumSize(820, 560)
        self.resize(1200, 780)
        self.setStyleSheet(STYLE)

        self._build_ui()
        self._connect_signals()
        # Сервисы (LLM, TTS) инициализируются в фоне: окно появляется сразу,
        # загрузка голоса и моделей не блокирует интерфейс.
        QTimer.singleShot(0, self._safe(self._build_services))
        self._update_info()

        # --- Аватар-компаньон (отдельное окно поверх всех) ---
        self._init_avatar()

        self._log(
            "Система",
            f"J.A.R.V.I.S. запущен. Провайдер: {self.settings.provider_label()}, "
            f"модель: {self.settings.model}.",
        )
        self._log(
            "Система",
            f"Скажите «{self.settings.assistant_name}» и команду. "
            "Офлайн-команды выполняются без интернета.",
        )
        self._init_stt_async()

    # ------------------------------------------------------------------
    # Построение интерфейса
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.title_bar = TitleBar()
        outer.addWidget(self.title_bar)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(8)

        # Три панели разделены перетаскиваемыми разделителями: можно менять
        # ширину блока диалога, сферы и системной панели.
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("PanelSplitter")
        splitter.setHandleWidth(6)
        splitter.setChildrenCollapsible(False)
        self.left_panel = self._build_left_panel()
        self.center_panel = self._build_center_panel()
        self.right_panel = self._build_right_panel()
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.center_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([230, 420, 400])
        self.panel_splitter = splitter
        self._dialog_focus = False
        self._saved_sizes: list[int] = []
        content_layout.addWidget(splitter)

        outer.addWidget(content, 1)

        # Ручка изменения размера: окно без рамки, системного края нет.
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 4, 4)
        grip_row.addStretch(1)
        grip = QSizeGrip(root)
        grip.setFixedSize(16, 16)
        grip_row.addWidget(grip)
        outer.addLayout(grip_row)

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setMinimumWidth(200)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(9)

        section = QLabel("СИСТЕМА")
        section.setObjectName("SectionLabel")
        layout.addWidget(section)

        self.status_label = QLabel("Инициализация...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#7fe7ff; font-weight:600;")
        layout.addWidget(self.status_label)

        layout.addSpacing(8)
        layout.addWidget(self._info_row("Провайдер", "provider_value"))
        layout.addWidget(self._info_row("Модель", "model_value"))
        layout.addWidget(self._info_row("Кодинг", "coding_value"))
        layout.addWidget(self._info_row("Озвучка", "tts_value"))
        layout.addSpacing(12)

        self.listen_button = QPushButton("🎙  Слушать")
        self.listen_button.setObjectName("AccentButton")
        self.listen_button.clicked.connect(self._safe(self._toggle_listen))
        layout.addWidget(self.listen_button)

        self.settings_button = QPushButton("Настройки")
        self.settings_button.clicked.connect(self._safe(self._open_settings))
        layout.addWidget(self.settings_button)

        self.stop_button = QPushButton("Стоп")
        self.stop_button.clicked.connect(self._safe(self._stop_speaking))
        layout.addWidget(self.stop_button)

        self.history_button = QPushButton("📜  История")
        self.history_button.clicked.connect(self._safe(self._open_history))
        layout.addWidget(self.history_button)

        layout.addStretch(1)

        hint = QLabel(
            "Скажите имя ассистента, затем команду.\n"
            "Например: «Джарвис, найди трек AC/DC на YouTube»."
        )
        hint.setWordWrap(True)
        hint.setObjectName("HintLabel")
        layout.addWidget(hint)
        return panel

    def _info_row(self, title: str, attr: str) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        label = QLabel(title.upper())
        label.setObjectName("SectionLabel")
        label.setStyleSheet("font-size:9px; color:#3d7f96; letter-spacing:1px;")
        value = QLabel("—")
        value.setObjectName("ValueLabel")
        value.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(value)
        setattr(self, attr, value)
        return container

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.sphere = SphereWidget()
        layout.addWidget(self.sphere, 1)

        self.state_label = QLabel("ОЖИДАНИЕ")
        self.state_label.setObjectName("StateLabel")
        self.state_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.state_label)

        self.transcript_label = QLabel("Система готова к работе")
        self.transcript_label.setAlignment(Qt.AlignCenter)
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setObjectName("HintLabel")
        self.transcript_label.setMinimumHeight(34)
        layout.addWidget(self.transcript_label)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Введите команду, сэр...")
        self.input_edit.returnPressed.connect(self._safe(self._on_input_submitted))
        send_button = QPushButton("Отправить")
        send_button.clicked.connect(self._safe(self._on_input_submitted))
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(send_button)
        layout.addLayout(input_row)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setMinimumWidth(240)
        # Верхней границы ширины нет: блок диалога можно растянуть на всё окно.
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        section = QLabel("ДИАЛОГ")
        section.setObjectName("SectionLabel")
        header.addWidget(section)
        header.addStretch(1)
        self.focus_button = QPushButton("⤢")
        self.focus_button.setObjectName("WinButton")
        self.focus_button.setFixedSize(28, 24)
        self.focus_button.setToolTip(
            "Развернуть диалог на всё окно / вернуть обычный вид"
        )
        self.focus_button.clicked.connect(self._safe(self._toggle_dialog_focus))
        header.addWidget(self.focus_button)
        layout.addLayout(header)

        self.log_view = QTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Segoe UI", 9))
        self.log_view.setMinimumHeight(150)
        # Перенос по границе виджета: ни текст, ни код не выходят за рамки.
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.log_view.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        layout.addWidget(self.log_view, 1)

        hint = QLabel("Локальные команды: громкость, скриншот, приложения, питание.")
        hint.setWordWrap(True)
        hint.setObjectName("HintLabel")
        layout.addWidget(hint)
        return panel

    def _init_avatar(self) -> None:
        """Создаёт окно аватара, если он включён в настройках."""
        try:
            if self.avatar is None:
                self.avatar = AvatarWindow(self.settings)
                self.avatar.set_state("sleep")
            self.avatar.apply_settings(self.settings)
            if self.settings.avatar_enabled:
                self.avatar.show()
                self.avatar.raise_()
            else:
                self.avatar.hide()
        except Exception:
            log.exception("avatar init failed")
            self.avatar = None

    def _on_sphere_state(self, state: str) -> None:
        """Сфера сменила эмоцию — аватар повторяет."""
        if self.avatar is None:
            return
        try:
            self.avatar.set_state(state)
        except Exception:
            log.exception("avatar set_state failed")

    def _cursor_walk(self, x: int, y: int) -> bool:
        """Аватар идёт к точке и «нажимает» вместо курсора (вызов из потока)."""
        avatar = self.avatar
        if avatar is None or not getattr(self.settings, "avatar_enabled", False):
            return False
        try:
            return avatar.request_walk(x, y, timeout=15.0)
        except Exception:
            log.debug("cursor walk failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Сигналы
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.sig_ai_result.connect(self._safe(self._on_ai_result))
        self.sig_ai_error.connect(self._safe(self._on_ai_error))
        self.sig_tool_event.connect(self._safe(self._on_tool_event))
        self.sig_tts_level.connect(self._safe(self._on_tts_level))
        self.sig_tts_started.connect(self._safe(self._on_tts_started))
        self.sig_tts_finished.connect(self._safe(self._on_tts_finished))
        self.sig_stt_level.connect(self._safe(self._on_stt_level))
        self.sig_stt_status.connect(self._safe(self._on_stt_status))
        self.sig_stt_error.connect(self._safe(self._on_stt_error))
        self.sig_heard.connect(self._safe(self._on_heard))
        self.sig_wake.connect(self._safe(self._on_wake))
        self.sig_command.connect(self._safe(self._handle_text))
        self.sig_confirm.connect(self._safe(self._on_confirm_request))
        self.sig_stt_ready.connect(self._safe(self._on_stt_ready))
        self.sig_barge_in.connect(self._safe(self._on_barge_in))
        self.sig_session_start.connect(self._safe(self._on_session_start))
        self.sig_session_end.connect(self._safe(self._on_session_end))
        self.sig_artifact.connect(self._safe(self._on_artifact))
        self.sig_confirm_retry.connect(self._safe(self._on_confirm_retry))
        self.sphere.sig_state.connect(self._safe(self._on_sphere_state))

    @staticmethod
    def _safe(handler):
        """Оборачивает слот: исключение не должно убивать приложение.

        PySide6 аварийно завершает процесс при необработанном исключении
        в слоте, поэтому все обработчики сигналов идут через эту обёртку.
        """

        def wrapper(*args, **kwargs):
            try:
                return handler(*args, **kwargs)
            except Exception:
                log.exception("Слот %s упал", getattr(handler, "__name__", handler))
                return None

        return wrapper

    # ------------------------------------------------------------------
    # Сервисы
    # ------------------------------------------------------------------
    def _build_services(self) -> None:
        self.tools = ToolRegistry(
            self.settings,
            confirm_callback=self._confirm_tool,
            on_event=self.sig_tool_event.emit,
            on_artifact=self.sig_artifact.emit,
            cursor_animator=self._cursor_walk,
        )
        try:
            self.ai = AIManager(
                self.settings, self.tools, on_event=self.sig_tool_event.emit
            )
        except Exception as exc:
            self.ai = None
            log.exception("AI init failed")
            self.sig_tool_event.emit(f"LLM недоступен: {exc}")

        self.offline = OfflineCommands(self.tools, self.settings)
        self.tts = TTSManager(self.settings, on_status=self.sig_stt_status.emit)
        if not self.tts.available:
            self._log("Система", "Озвучка недоступна — проверьте настройки TTS.")
        else:
            self._prewarm_voice()

    def _prewarm_voice(self) -> None:
        """Заранее синтезирует типовые фразы, чтобы первые ответы звучали сразу."""
        if self.no_audio or not getattr(self.settings, "voice_prewarm", True):
            return
        prewarm = getattr(self.tts, "prewarm", None)
        if callable(prewarm):
            try:
                prewarm()
            except Exception:
                log.debug("voice prewarm skipped", exc_info=True)

    def _init_stt_async(self) -> None:
        if self.no_audio:
            self._set_status("Аудио отключено (--no-audio)")
            self._log("Система", "Аудиоввод отключён. Доступен текстовый ввод.")
            return
        self._set_status("Загружаю распознавание речи...")

        def worker() -> None:
            try:
                stt = create_stt(
                    self.settings, on_status=lambda message: self.sig_stt_status.emit(message)
                )
            except Exception as exc:
                log.exception("STT init failed")
                self.sig_stt_error.emit(str(exc))
                return
            self.sig_stt_ready.emit(stt)

        threading.Thread(target=worker, daemon=True).start()

    def _start_listener(self, force: bool = False) -> None:
        if self.no_audio or self._stt is None:
            return
        if self._listener is not None and self._listener.is_alive():
            return
        self._listener = VoiceListener(
            self._stt,
            self.settings,
            on_wake=self.sig_wake.emit,
            on_command=self.sig_command.emit,
            on_status=self.sig_stt_status.emit,
            on_level=self.sig_stt_level.emit,
            on_error=self.sig_stt_error.emit,
            on_heard=self.sig_heard.emit,
            on_barge_in=self.sig_barge_in.emit,
            on_session_start=self.sig_session_start.emit,
            on_session_end=self.sig_session_end.emit,
            on_confirm_retry=self.sig_confirm_retry.emit,
        )
        if self.settings.auto_start_listening or force:
            self._listener.start()
            self._set_status(f"Жду имя: «{self.settings.assistant_name}»")
            self.sphere.set_state("sleep")
            self.state_label.setText("ОЖИДАНИЕ")
        else:
            self._set_status("Микрофон выключен — нажмите «Слушать»")

    # ------------------------------------------------------------------
    # Обработка текста
    # ------------------------------------------------------------------
    def _handle_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._manual_listen = False
        self._log("Вы", text)
        self.transcript_label.setText(text)

        # «Стоп» / «отбой» — прекращаем всё: речь, задачу, прослушивание.
        if is_stop_command(text):
            self._handle_stop()
            return

        # Пока ассистент обрабатывает команду, микрофон не слушаем.
        if self._listener is not None:
            self._listener.pause()

        match = self.offline.match(text)
        if match is not None:
            self._log("Джарвис", match.response)
            self._set_status(f"Офлайн-команда: {match.command_id}")
            self._flash("happy", 900)
            if match.ui_action:
                self._handle_ui_action(match.ui_action)
            if match.action is not None:
                threading.Thread(
                    target=self._run_offline_action, args=(match,), daemon=True
                ).start()
            self._speak(match.response, command_id=match.command_id)
            # Для одноразового режима завершаем сессию после речи
            if self._oneshot_active:
                QTimer.singleShot(1800, self._end_oneshot)
            return

        self._ask_ai(text)

    def _handle_stop(self) -> None:
        """«Имя, стоп»: обрываем речь, отменяем задачу и выходим из сессии."""
        self._stop_activity()
        self._ignore_answer = True
        self._busy = False
        if self.ai is not None:
            self.ai.cancel()
        self.tts.stop()
        if self._listener is not None:
            self._listener.set_speaking(False)
            self._listener.end_session("stop")
        else:
            self._on_session_end("stop")

    def _ask_ai(self, text: str) -> None:
        if self.ai is None:
            self._log("Ошибка", "LLM не настроен. Откройте настройки и выберите провайдера.")
            self._speak("Модель не настроена, сэр. Откройте настройки.")
            return
        if self._busy:
            self._speak("Я ещё выполняю предыдущую задачу, сэр.")
            return

        # Создаём новую сессию при первом сообщении
        if self._current_session is None:
            self._new_session()

        # Обновляем название сессии по первому сообщению
        self._update_session_title(text)

        self._busy = True
        self._ignore_answer = False
        self.sphere.set_state("thinking")
        self.state_label.setText("ДУМАЮ")
        self._set_status("Запрос к модели...")
        self._activity_timer.start()

        def worker() -> None:
            try:
                answer = self.ai.ask(text)
            except Exception as exc:
                log.exception("AI request failed")
                self.sig_ai_error.emit(str(exc))
            else:
                self.sig_ai_result.emit(answer)

        threading.Thread(target=worker, daemon=True, name="AIWorker").start()

    def _activity_tick(self) -> None:
        """Если занят и не говорит — короткая реплика, чтобы не молчать."""
        if not self._busy:
            self._activity_timer.stop()
            return
        if self.tts and self.tts.is_speaking:
            return
        self._speak(pick_phrase(ACTIVITY_PHRASES))
        # Рандомизируем следующий тик: 25–40 секунд.
        self._activity_timer.stop()
        self._activity_timer.setInterval(25000 + random.randint(0, 15000))
        self._activity_timer.start()

    def _stop_activity(self) -> None:
        self._activity_timer.stop()

    def _run_offline_action(self, match) -> None:
        try:
            result = match.action() if match.action else ""
        except Exception as exc:
            log.exception("offline action failed")
            self.sig_tool_event.emit(f"Ошибка команды {match.command_id}: {exc}")
            return
        if result:
            self.sig_tool_event.emit(str(result))

    def _handle_ui_action(self, action: str) -> None:
        if action == "settings":
            QTimer.singleShot(450, self._open_settings)
        elif action == "minimize":
            # Слушатель возобновится после завершения фразы «Сворачиваюсь».
            self.showMinimized()
        elif action == "quit":
            QTimer.singleShot(2200, self.close)

    # ------------------------------------------------------------------
    # Речь
    # ------------------------------------------------------------------
    def _speak(self, text: str, command_id: str | None = None) -> None:
        if not text:
            return
        # Субтитр покажем в момент старта звука (on_start), иначе он
        # появляется раньше голоса и «висит» во время синтеза.
        self._pending_speech = text
        if self.no_audio or not getattr(self.tts, "available", False):
            # Голоса не будет — показываем субтитр сразу.
            self._show_subtitle(text)
        if self.no_audio:
            # В режиме отладки озвучка отключена вместе с микрофоном.
            self.sphere.set_state("idle")
            self.state_label.setText("ОЖИДАНИЕ")
            if self._oneshot_active:
                # Без звука on_end не придёт — завершаем одноразовую сессию сами.
                QTimer.singleShot(1500, self._end_oneshot)
            return
        # Состояние «ГОВОРЮ» включится, когда реально начнётся звук —
        # так эмоция (happy/working/thinking) остаётся видимой во время синтеза.
        if self._listener is not None:
            # Не пауза, а режим перебивания: микрофон слышит владельца.
            self._listener.set_speaking(True)

        callbacks = {
            "on_start": lambda: self.sig_tts_started.emit(),
            "on_end": lambda: self.sig_tts_finished.emit(),
            "on_amplitude": lambda value: self.sig_tts_level.emit(float(value)),
        }
        if command_id:
            self.tts.speak_offline(command_id, text, **callbacks)
        else:
            self.tts.speak(text, **callbacks)

    def _show_subtitle(self, text: str, seconds: float = 10.0) -> None:
        """Субтитр аватара — строго вместе со звуком."""
        if self.avatar is None or not text:
            return
        try:
            self.avatar.show_subtitle(text, seconds=seconds)
        except Exception:
            log.debug("avatar subtitle failed", exc_info=True)

    def _flash(self, state: str, milliseconds: int = 900) -> None:
        """Короткая эмоция сферы с возвратом к базовому состоянию."""
        self.sphere.set_state(state)
        self.state_label.setText(state.upper())
        QTimer.singleShot(milliseconds, self._restore_state)

    def _restore_state(self) -> None:
        tts = getattr(self, "tts", None)
        if tts is not None and tts.is_speaking:
            return
        if self._session_active:
            self.sphere.set_state("listening")
            self.state_label.setText("СЛУШАЮ")
        else:
            self.sphere.set_state("sleep")
            self.state_label.setText("ОЖИДАНИЕ")

    def _stop_speaking(self) -> None:
        if self.tts:
            self.tts.stop()
        if self._listener is not None:
            self._listener.resume()
        self._busy = False
        self.sphere.set_state("idle")
        self.state_label.setText("ОЖИДАНИЕ")

    def _toggle_dialog_focus(self) -> None:
        """Диалог на всю ширину окна или обычный трёхпанельный вид."""
        if not self._dialog_focus:
            self._saved_sizes = self.panel_splitter.sizes()
            self.left_panel.setVisible(False)
            self.center_panel.setVisible(False)
            self.right_panel.setMinimumWidth(320)
            self._dialog_focus = True
            self.focus_button.setText("⤡")
            self.focus_button.setToolTip("Вернуть обычный вид")
        else:
            self.left_panel.setVisible(True)
            self.center_panel.setVisible(True)
            if self._saved_sizes:
                self.panel_splitter.setSizes(self._saved_sizes)
            self._dialog_focus = False
            self.focus_button.setText("⤢")
            self.focus_button.setToolTip("Развернуть диалог на всё окно")

    def _open_history(self) -> None:
        """Открывает диалог с историей чатов."""
        dialog = HistoryDialog(self)
        dialog.sig_load_session.connect(self._safe(self._on_session_loaded))
        dialog.exec()

    def _on_session_loaded(self, session_id: str) -> None:
        """Загружает выбранный диалог или создаёт новый."""
        if session_id == "__new__":
            self._new_session()
            return

        session = load_session(session_id)
        if session is None:
            self._log("Система", "Не удалось загрузить диалог.")
            return

        # Сохраняем текущую сессию перед переключением
        self._save_current_session()

        # Восстанавливаем историю
        self._current_session = session
        messages = session.get("messages", [])

        # Загружаем историю в AIManager
        if self.ai is not None:
            self.ai.history = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if m.get("role") in ("user", "assistant")
            ]

        # Восстанавливаем отображение в log_view
        self.log_view.clear()
        self.log_view.append(
            f'<div style="color:#4f7f95;font-size:11px;margin-bottom:8px;">'
            f'— Восстановлен диалог «{html.escape(session.get("title", "Без названия"))}» —</div>'
        )

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                self._log("Вы", content)
            elif role == "assistant":
                self._log("Джарвис", content)

        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

        self._log("Система", "Диалог восстановлен. Продолжайте общение.")

    def _new_session(self) -> None:
        """Создаёт новую сессию диалога."""
        self._save_current_session()
        self._current_session = open_session()
        if self.ai is not None:
            self.ai.reset()
        self.log_view.clear()
        self._log("Система", "Новый диалог начат.")

    def _save_current_session(self) -> None:
        """Сохраняет текущую сессию с актуальной историей."""
        if self._current_session is None:
            return
        if self.ai is not None:
            self._current_session["messages"] = [
                {"role": m["role"], "content": m["content"]}
                for m in self.ai.history
                if m.get("role") in ("user", "assistant")
            ]
        if self._current_session.get("messages"):
            save_session(self._current_session)

    def _update_session_title(self, first_user_text: str) -> None:
        """Обновляет название сессии по первому сообщению пользователя."""
        if self._current_session is None:
            return
        if self._current_session.get("title") != "Новый диалог":
            return
        title = first_user_text[:60].strip()
        if len(first_user_text) > 60:
            title += "..."
        self._current_session["title"] = title
        self._set_status("Остановлено")
        if self._listener is not None:
            self._listener.set_speaking(False)
            self._listener.resume()

    # ------------------------------------------------------------------
    # Слоты
    # ------------------------------------------------------------------
    def _on_ai_result(self, answer: str) -> None:
        self._busy = False
        self._stop_activity()
        if self.avatar is not None:
            try:
                self.avatar.set_typing(False)
            except Exception:
                pass
        if self._ignore_answer:
            self._ignore_answer = False
            self._log(self.settings.assistant_name, answer)
            self._set_status("Готов")
            if self._listener is not None:
                self._listener.resume()
            return
        self._log(self.settings.assistant_name, answer)
        self.transcript_label.setText(answer[:220])
        self._flash("happy", 1200)
        self._speak(answer)
        self._set_status("Готов")
        # Сохраняем сессию после ответа ассистента
        self._save_current_session()

    def _on_ai_error(self, message: str) -> None:
        self._busy = False
        self._stop_activity()
        self._log("Ошибка", message)
        self._flash("error", 1500)
        self._set_status("Ошибка модели")
        self._speak("Не удалось получить ответ от модели, сэр. Проверьте подключение и ключ.")

    def _on_tool_event(self, message: str) -> None:
        self._log("Агент", message)
        # Аватар печатает на клавиатуре, когда агент пишет код.
        coding_tools = ("write_file", "edit_file", "write_and_run_script")
        if self.avatar is not None:
            try:
                if message.startswith("Инструмент:") and any(
                    tool in message for tool in coding_tools
                ):
                    self.avatar.set_typing(True)
                elif message.startswith("Результат"):
                    self.avatar.set_typing(False)
            except Exception:
                pass

        # Размышления модели вслух: короткая фраза, если TTS свободен.
        if message.startswith("РАССКАЗ: ") and getattr(self.settings, "speak_reasoning", True):
            text = message[len("РАССКАЗ: "):]
            if not self.tts.is_speaking and len(text) > 15:
                self._speak(text[:200])

        # Статус «почти готово» — меняем текст на экране.
        if "формулирую итог" in message.lower():
            self.sphere.set_state("thinking")
            self.state_label.setText("ДУМАЮ НАД ОТВЕТОМ")

        if message.startswith("Выполняю инструмент"):
            self.sphere.set_state("working")
            self.state_label.setText("РАБОТАЮ")
            # Короткий звук-блип — чтобы не молчать.
            try:
                import winsound

                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception:
                pass
        elif message.startswith("Результат") and "Ошибка" in message:
            self._flash("error", 1200)

    def _on_tts_level(self, value: float) -> None:
        self.sphere.set_level(value)
        if self.avatar is not None:
            try:
                self.avatar.set_level(value)
            except Exception:
                pass

    def _on_tts_started(self) -> None:
        self.sphere.set_state("speaking")
        self.state_label.setText("ГОВОРЮ")
        # Субтитр появляется одновременно с первым звуком.
        self._show_subtitle(getattr(self, "_pending_speech", ""))
        if self._listener is not None:
            self._listener.set_speaking(True)

    def _on_tts_finished(self) -> None:
        if self._listener is not None:
            self._listener.set_speaking(False)
            self._listener.resume()
        # Одноразовый режим: ответили — и снова ждём имя.
        if self._oneshot_active and not self._busy:
            self._end_oneshot()
            return
        if self._busy:
            self.sphere.set_state("working")
            self.state_label.setText("РАБОТАЮ")
            self._set_status("Выполняю задачу...")
        elif self._session_active:
            self.sphere.set_state("listening")
            self.state_label.setText("СЛУШАЮ")
            self._set_status("Слушаю, сэр...")
        else:
            self.sphere.set_state("sleep")
            self.state_label.setText("ОЖИДАНИЕ")
            self._set_status(f"Жду имя: «{self.settings.assistant_name}»")

    def _on_barge_in(self) -> None:
        """Владелец перебил ответ — замолкаем и слушаем его команду."""
        self._log("Система", "Перебивание: останавливаю речь.")
        self._ignore_answer = True
        self.tts.stop()
        self.sphere.set_state("listening")
        self.state_label.setText("СЛУШАЮ")
        self._set_status("Слушаю вас...")
        if self._listener is not None:
            self._listener.set_speaking(False)
            self._listener.request_command()

    def _on_stt_level(self, value: float) -> None:
        if self.sphere.state in {"idle", "sleep", "listening"}:
            self.sphere.set_level(value * 0.55)

    def _on_stt_status(self, message: str) -> None:
        # Статусы слушателя не должны перебивать состояние «ГОВОРЮ».
        tts = getattr(self, "tts", None)
        if tts is not None and tts.is_speaking:
            return
        # Пока имя не названо, ассистент работает «в фоне»: микрофон слушает,
        # но ни сфера, ни аватар, ни статус не меняются.
        if self._silent_waiting() and message.startswith(("Распознаю", "Слушаю команду")):
            log.debug("silent stt status: %s", message)
            return
        self._set_status(message)
        if message.startswith("Слушаю команду"):
            self.sphere.set_state("listening")
            self.state_label.setText("СЛУШАЮ")
        elif message == "Слушаю...":
            # Ожидание имени: сфера «спит», пока не позовут.
            if self._session_active:
                self.sphere.set_state("listening")
                self.state_label.setText("СЛУШАЮ")
            else:
                self.sphere.set_state("sleep")
                self.state_label.setText("ОЖИДАНИЕ")
        elif message.startswith("Жду ответа"):
            self.sphere.set_state("confirm")
            self.state_label.setText("ВЫ МЕНЯ ЗВАЛИ?")
        elif message.startswith("Распознаю") or message.startswith("Загружаю"):
            self.sphere.set_state("thinking")
            self.state_label.setText("АНАЛИЗ")

    def _silent_waiting(self) -> bool:
        """Ждём имя: любые фоновые действия не должны менять интерфейс."""
        return (
            bool(self.settings.wake_word_enabled)
            and not self._session_active
            and not self._oneshot_active
            and not self._manual_listen
        )

    def _on_stt_error(self, message: str) -> None:
        self._log("Ошибка", message)
        self._set_status("Микрофон недоступен")
        self._flash("error", 1600)

    def _on_heard(self, text: str) -> None:
        """Диагностика wake word: в чат пишем только то, что относится к делу."""
        if not self.settings.wake_word_enabled:
            self._log("Микрофон", text)
            return
        found, _rest = match_wake_word(text, self.settings.assistant_name)
        if found or self._session_active or self._oneshot_active:
            self._log("Микрофон", text)
            return
        # Имя не названо — распознали в фоне и молчим, интерфейс не трогаем.
        log.debug("heard without name: %s", text)

    def _on_wake(self) -> None:
        """Услышали имя: спрашиваем «Вы меня вызывали?» и ждём да/нет."""
        self._log("Система", "Активация по имени.")
        self.sphere.set_state("confirm")
        self.state_label.setText("ВЫ МЕНЯ ЗВАЛИ?")
        self._set_status("Жду ответа: да или нет")
        self._speak(pick_phrase(WAKE_PHRASES))

    def _on_confirm_retry(self) -> None:
        """Не расслышали «да/нет» — переспрашиваем один раз."""
        self._log("Система", "Не расслышал ответ — переспрашиваю.")
        self._speak("Прошу прощения, сэр. Да или нет?")

    def _on_session_start(self, mode: str = "dialog") -> None:
        """Имя услышано.

        mode="oneshot" — имя сказали вместе с командой: выполняем её одну
        и снова ждём имя.
        mode="dialog"/"listen" — слушаем команды без имени до слова «стоп»;
        на чистое имя отвечаем коротким подтверждением.
        """
        self._session_active = True
        self._manual_listen = False
        self._oneshot_active = mode == "oneshot"
        self.sphere.set_state("listening")
        self.state_label.setText("СЛУШАЮ")
        self._set_status("Слушаю, сэр...")
        if mode == "listen":
            self._log("Система", "Имя услышано — слушаю до слова «стоп».")
            self._speak(pick_phrase(SESSION_PHRASES))
        elif mode == "oneshot":
            self._log("Система", "Имя и команда — выполняю одно действие.")
        else:
            self._log("Система", "Сессия активна — слушаю без имени.")

    def _end_oneshot(self) -> None:
        """Одна команда выполнена — возвращаемся в ожидание имени."""
        if not self._oneshot_active:
            return
        self._oneshot_active = False
        if self._listener is not None:
            self._listener.cancel_session()
        # defer в главный поток — в exe Qt-слоты из TTS-потока могут
        # падать нативно; QTimer.singleShot(0) гарантирует главный поток.
        QTimer.singleShot(0, lambda: self._on_session_end("oneshot"))

    def _on_session_end(self, reason: str) -> None:
        """«Нет», «стоп» или «oneshot»: уходим в сон и ждём имя.

        В exe необработанное исключение в Qt-слоте убивает процесс,
        поэтому весь обработчик под try/except.
        """
        try:
            self._session_active = False
            self._oneshot_active = False
            self._manual_listen = False
            self._stop_activity()
            if reason == "declined":
                self._log("Система", "Ответ «нет» — возвращаюсь в ожидание имени.")
                phrase = pick_phrase(DECLINE_PHRASES)
            elif reason == "oneshot":
                # Одноразовая команда: выполнили и возвращаемся в ожидание имени.
                self._log("Система", "Одноразовая команда выполнена — жду имя.")
                phrase = ""
            else:
                # «Стоп» — прекращаем всё: текущую задачу, инструменты и речь.
                self._ignore_answer = True
                self._busy = False
                if self.ai is not None:
                    self.ai.cancel()
                self.tts.stop()
                if self._listener is not None:
                    self._listener.set_speaking(False)
                self._log("Система", "Сессия завершена по команде «стоп».")
                phrase = pick_phrase(STOP_PHRASES)
            self.sphere.set_state("sleep")
            self.state_label.setText("ОЖИДАНИЕ")
            self._set_status(f"Жду имя: «{self.settings.assistant_name}»")
            self._speak(phrase)
        except Exception:
            log.exception("session end failed (reason=%s)", reason)

    def _on_stt_ready(self, stt) -> None:
        self._stt = stt
        self._log("Система", f"Распознавание речи: {stt.label}")
        self._start_listener()
        if self._start_after and self._listener is not None:
            self._start_after = False
            self._listener.request_command()

    def _on_input_submitted(self) -> None:
        text = self.input_edit.text().strip()
        self.input_edit.clear()
        if text:
            self._handle_text(text)

    def _toggle_listen(self) -> None:
        if self.no_audio:
            self._log("Система", "Аудиоввод отключён. Используйте текстовое поле.")
            return
        if self._stt is None:
            self._start_after = True
            self._init_stt_async()
            return
        if self._listener is None or not self._listener.is_alive():
            self._start_listener(force=True)
        if self._listener is not None:
            self._listener.request_command()
            self._manual_listen = True
            self.sphere.set_state("listening")
            self.state_label.setText("СЛУШАЮ")
            self._set_status("Слушаю команду...")

    # ------------------------------------------------------------------
    # Подтверждения инструментов
    # ------------------------------------------------------------------
    def _confirm_tool(self, name: str, args: dict) -> bool:
        # Подтверждения отключены: агент работает без вопросов.
        return True

    def _show_confirm_dialog(self, name: str, payload: str) -> bool:
        box = QMessageBox(self)
        box.setWindowTitle("Подтверждение действия")
        box.setIcon(QMessageBox.Warning)
        box.setText(f"Агент хочет выполнить инструмент:\n{name}")
        box.setInformativeText("Продолжить?")
        box.setDetailedText(payload)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.button(QMessageBox.Yes).setText("Разрешить")
        box.button(QMessageBox.No).setText("Отклонить")
        return box.exec() == QMessageBox.Yes

    def _on_confirm_request(self, name: str, payload: str) -> None:
        self._confirm_result = self._show_confirm_dialog(name, payload)
        self._confirm_event.set()

    # ------------------------------------------------------------------
    # Настройки и сервисные действия
    # ------------------------------------------------------------------
    def _open_settings(self) -> None:
        # Пока открыт диалог, микрофон не должен слышать тест голоса.
        if self._listener is not None:
            self._listener.pause()
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() != QDialog.Accepted:
            if self._listener is not None:
                self._listener.resume()
            return
        self.settings = dialog.result_settings()
        self.settings.save()
        # Применяем настройки аватара сразу.
        self._init_avatar()
        if self.settings.autostart != is_autostart_enabled():
            self._log("Система", set_autostart(self.settings.autostart))
        self._log("Система", "Настройки сохранены. Перезапускаю модули...")
        self._stop_speaking()
        self._busy = False
        self._build_services()
        self._update_info()

        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._stt = None
        self._init_stt_async()

    def _update_info(self) -> None:
        self.provider_value.setText(self.settings.provider_label())
        self.model_value.setText(self.settings.model)
        if self.settings.coding_enabled():
            self.coding_value.setText(
                f"{self.settings.coding_provider_label()} · "
                f"{self.settings.coding_model_resolved()}"
            )
        else:
            self.coding_value.setText("как основная")
        self.tts_value.setText(self.tts.label if hasattr(self, "tts") else "—")
        self.title_bar.set_subtitle(
            f"Личный ассистент · {self.settings.assistant_name}"
        )
        self.setWindowTitle(f"J.A.R.V.I.S. — {self.settings.assistant_name}")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _log(self, author: str, text: str) -> None:
        colors = {
            "Вы": "#7fe7ff",
            "Джарвис": "#4dffc3",
            "Система": "#ffb86b",
            "Агент": "#b48cff",
            "Ошибка": "#ff6b81",
        }
        color = colors.get(author, "#9fd8ea")
        safe_author = html.escape(author)
        safe_text = html.escape(text)
        # pre-wrap: перенос по границе блока + сохранение переносов/отступов.
        self.log_view.append(
            f'<div style="margin-bottom:6px;white-space:pre-wrap;'
            f'overflow-wrap:break-word;word-break:break-word;">'
            f'<span style="color:{color};font-weight:600;">{safe_author}:</span> '
            f'<span style="color:#cfe9f5;">{safe_text}</span></div>'
        )
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ------------------------------------------------------------------
    def _on_artifact(self, data: object) -> None:
        """Карточка созданного/изменённого файла — как в Claude Code."""
        if not isinstance(data, dict):
            return
        action = str(data.get("action") or "update")
        path = str(data.get("path") or "")
        added = int(data.get("added") or 0)
        removed = int(data.get("removed") or 0)
        lines = int(data.get("lines") or 0)
        language = str(data.get("language") or "")
        diff = data.get("diff") or []

        if action == "create":
            title = "Создан файл"
            badge_color = "#4dffc3"
        else:
            title = "Изменён файл"
            badge_color = "#7fb4ff"

        safe_path = html.escape(path)
        stat = f"+{added}" + (f" −{removed}" if removed else "") + f" · {lines} строк"
        # pre-wrap + break-word: код и диффы never вылезают за границы блока.
        code_style = "white-space:pre-wrap;overflow-wrap:break-word;word-break:break-word;"
        rows = []
        for line in diff[:120]:
            text = html.escape(str(line).replace("	", "    "))
            if line.startswith("+"):
                rows.append(
                    f'<div style="color:#7dffb0;{code_style}">{text}</div>'
                )
            elif line.startswith("-"):
                rows.append(
                    f'<div style="color:#ff8f9c;{code_style}">{text}</div>'
                )
            elif line.startswith("@@"):
                rows.append(
                    f'<div style="color:#8ab4ff;{code_style}">{text}</div>'
                )
            else:
                rows.append(
                    f'<div style="color:#8fa8b8;{code_style}">{text}</div>'
                )
        if len(diff) > 120:
            rows.append(
                f'<div style="color:#8fa8b8;">... (дифф обрезан, всего {len(diff)} строк)</div>'
            )

        self.log_view.append(
            f'<div style="margin:10px 0;padding:8px 10px;'
            f'border:1px solid {badge_color}55;border-left:3px solid {badge_color};'
            f'border-radius:6px;background:#0b1b26;">'
            f'<div style="color:{badge_color};font-weight:600;">'
            f'📄 {title} <span style="color:#8fa8b8;font-weight:400;">'
            f'({language or "text"})</span></div>'
            f'<div style="color:#cfe9f5;font-family:Consolas,monospace;'
            f'white-space:pre-wrap;overflow-wrap:break-word;word-break:break-all;">'
            f'{safe_path}</div>'
            f'<div style="color:#8fa8b8;margin-bottom:6px;">{stat}</div>'
            f'<div style="font-family:Consolas,monospace;font-size:11px;">'
            f'{"".join(rows)}</div></div>'
        )
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_current_session()
        if self.avatar is not None:
            try:
                self.avatar.close()
            except Exception:
                pass
        try:
            self.settings.save()
        except Exception:
            pass
        if hasattr(self, "tts"):
            self.tts.stop()
        if self._listener is not None:
            self._listener.stop()
        event.accept()
