"""Аватар-компаньон J.A.R.V.I.S.

Сферический персонаж: светящаяся сфера-тело, две ладони на коротких
«руках» и две ступни на коротких «ногах». Умеет ходить по экрану,
спать с подушкой, печатать на клавиатуре, показывать предметы и эмоции,
выводить субтитры и «идти» к точке клика вместо курсора.
"""

from __future__ import annotations

import logging
import math
import random
import sys
import threading
import time

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer

from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import QApplication, QWidget

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Палитра и предметы под состояние
# ---------------------------------------------------------------------------

_STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "idle": (0, 200, 235),
    "sleep": (80, 120, 165),
    "listening": (0, 235, 255),
    "thinking": (150, 110, 255),
    "speaking": (0, 235, 180),
    "confirm": (255, 200, 60),
    "working": (255, 160, 40),
    "coding": (130, 170, 255),
    "happy": (0, 240, 150),
    "error": (255, 80, 100),
}

_STATE_OBJECTS: dict[str, str] = {
    "idle": "",
    "sleep": "pillow",
    "listening": "eye",
    "thinking": "search",
    "speaking": "note",
    "confirm": "alert",
    "working": "gear",
    "coding": "keyboard",
    "happy": "star",
    "error": "alert",
}


def _rgb(color: QColor) -> tuple[int, int, int]:
    return color.red(), color.green(), color.blue()


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def _draw_holographic_sphere(
    painter: QPainter,
    cx: float,
    cy: float,
    radius: float,
    squash: float,
    phase: float,
    base: tuple[int, int, int],
    glow: QColor,
    mid: QColor,
    dark: QColor,
    level: float,
) -> None:
    """Рисует сферу из множества маленьких светящихся точек-сфер.

    Эффект голограммы: точки лежат на поверхности сферы, вращаются вокруг
    вертикальной оси; проекция на 2D + перспектива (размер и прозрачность
    зависят от расстояния до наблюдателя).
    """
    import math

    # Параметры сетки: чем больше точек, тем плотнее «голограмма».
    # ~480 точек — компромисс между красотой и производительностью.
    lat_steps = 18   # параллели
    lon_steps = 28   # меридианы

    # Мягкое внешнее свечение (фон под сеткой) — как у старой сферы.
    halo = QRadialGradient(QPointF(cx, cy), radius * 1.55)
    halo.setColorAt(0.0, QColor(*_rgb(glow), 70))
    halo.setColorAt(0.7, QColor(*base, 28))
    halo.setColorAt(1.0, QColor(0, 0, 0, 0))
    painter.setPen(Qt.NoPen)
    painter.setBrush(halo)
    painter.drawEllipse(QPointF(cx, cy), radius * 1.55, radius * 1.55 * squash)

    # Вращение: медленное, вокруг вертикальной оси + лёгкое «дыхание».
    rot = phase * 0.55
    breathe = 1.0 + 0.015 * math.sin(phase * 1.6)

    base_r = max(0.9, radius * 0.042)   # размер точки

    # Собираем точки и сортируем по глубине, чтобы ближние перекрывали дальние.
    dots: list[tuple[float, float, float, float]] = []  # x, y, r, alpha
    for i in range(lat_steps):
        # широта: от -pi/2 до pi/2 (от полюса до полюса)
        lat = -math.pi / 2 + (i + 0.5) * math.pi / lat_steps
        cos_lat = math.cos(lat)
        sin_lat = math.sin(lat)
        for j in range(lon_steps):
            lon = (j / lon_steps) * 2 * math.pi + rot
            # 3D координаты на сфере радиуса 1
            x3 = cos_lat * math.cos(lon)
            y3 = sin_lat
            z3 = cos_lat * math.sin(lon)

            # Проекция: ортографическая + перспектива (z ближе = крупнее).
            # z3 > 0 — точка ближе к наблюдателю.
            persp = 1.0 + 0.18 * z3
            px = cx + x3 * radius * persp * breathe
            py = cy - y3 * radius * squash * persp * breathe

            # Размер: ближние точки крупнее, дальние мельче.
            pr = base_r * persp * (0.7 + 0.3 * (z3 + 1.0) / 2.0)

            # Прозрачность: чем ближе, тем ярче. Добавляем «пульс» от уровня.
            alpha = int(
                110 + 130 * (z3 + 1.0) / 2.0
                + 40 * level * (z3 + 1.0) / 2.0
            )
            alpha = max(30, min(255, alpha))

            dots.append((px, py, pr, alpha))

    # Сортируем от дальних к ближним (по алгебраической z, но z у нас
    # уже учтён в alpha и persp — достаточно отсортировать по alpha).
    dots.sort(key=lambda d: d[3])

    # Рисуем точки: цвет зависит от глубины (ближе = glow, дальше = dark).
    for px, py, pr, alpha in dots:
        # Смешиваем base и glow по прозрачности.
        t = (alpha - 30) / 225.0
        color = _mix(_rgb(mid), _rgb(glow), t)
        color.setAlpha(alpha)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(px, py), pr, pr)

    # Тонкая светящаяся сфера-оболочка поверх сетки (как «стекло»).
    shell = QRadialGradient(QPointF(cx - radius * 0.3, cy - radius * 0.35), radius * 1.6)
    shell.setColorAt(0.0, QColor(255, 255, 255, 30))
    shell.setColorAt(0.5, QColor(*base, 10))
    shell.setColorAt(1.0, QColor(*_rgb(dark), 60))
    painter.setPen(QPen(QColor(*_rgb(glow), 140), max(1.0, 1.5 * (radius / 46.0))))
    painter.setBrush(shell)
    painter.drawEllipse(QPointF(cx, cy), radius * breathe, radius * squash * breathe)

    # Блик на «стекле».
    painter.setBrush(QColor(255, 255, 255, 60))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(
        QPointF(cx - radius * 0.32, cy - radius * 0.40),
        radius * 0.22, radius * 0.14,
    )


# ---------------------------------------------------------------------------
# Рисование персонажа-сферы
# ---------------------------------------------------------------------------

def draw_character(
    painter: QPainter,
    area: QRectF,
    state: str = "idle",
    phase: float = 0.0,
    blink: bool = False,
    *,
    objects: bool = True,
    typing: bool = False,
    level: float = 0.0,
    accent: str = "",
    walking: bool = False,
    gaze: float = 0.0,
    dragging: bool = False,
    pressing: bool = False,
) -> None:
    """Рисует сферу с ладонями и ступнями в области `area`."""
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)

    base = _STATE_COLORS.get(state, _STATE_COLORS["idle"])
    if accent:
        c = QColor(accent)
        if c.isValid():
            base = (c.red(), c.green(), c.blue())
    glow = _mix(base, (255, 255, 255), 0.45)
    mid = _mix(base, (0, 0, 0), 0.25)
    dark = _mix(base, (6, 14, 22), 0.82)

    w, h = area.width(), area.height()
    s = max(0.35, min(w / 190.0, h / 250.0))

    # Компоновка: сфера чуть выше центра, внизу — ноги.
    R = 46 * s
    ground = area.bottom() - 7 * s
    leg_len = 26 * s
    bob = 0.0
    squash = 1.0
    if walking:
        bob = abs(math.sin(phase * 7.5)) * 5 * s
    elif state == "sleep":
        # Во сне персонаж стоит неподвижно: только «z z z» над головой.
        bob = 4 * s
    else:
        bob = math.sin(phase * 1.7) * 1.8 * s

    if dragging:
        squash = 0.92 + 0.06 * math.sin(phase * 12.0)

    cy = ground - leg_len - R * squash + bob
    cx = area.center().x() + (math.sin(phase * 9.0) * 2.2 * s if dragging else 0.0)

    tilt = 0.0
    if walking:
        tilt = math.sin(phase * 7.5) * 4.0
    elif dragging:
        tilt = math.sin(phase * 10.0) * 9.0

    # --- тень ---------------------------------------------------------
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(0, 0, 0, 80))
    shadow_w = R * (0.95 - 0.15 * (bob / max(1.0, 6 * s)))
    painter.drawEllipse(QPointF(area.center().x(), ground + 4 * s), shadow_w, 5 * s)

    # --- ноги и ступни ------------------------------------------------
    leg_pen = QPen(QColor(*base, 215), max(1.6, 3.6 * s))
    leg_pen.setCapStyle(Qt.RoundCap)

    feet: list[tuple[float, float, float]] = []  # x, y, угол
    for sign in (-1, 1):
        if walking:
            step = math.sin(phase * 7.5 + (0 if sign < 0 else math.pi))
            lift = max(0.0, step) * 11 * s
            fwd = step * 8 * s
        elif dragging:
            # при переносе ступни болтаются вниз
            lift = abs(math.sin(phase * 8.0 + sign)) * 4 * s
            fwd = sign * 2 * s
        elif state == "sleep":
            lift = 0.0
            fwd = sign * 9 * s
        else:
            # стоя: ступни слегка притопывают и переступают
            tap = abs(math.sin(phase * 2.3 + (0 if sign < 0 else 1.1)))
            lift = tap * 3.2 * s
            fwd = sign * (11 + 2 * math.sin(phase * 1.4 + sign)) * s

        hip = QPointF(cx + sign * R * 0.34, cy + R * 0.72)
        knee = QPointF(hip.x() + fwd * 0.45, hip.y() + leg_len * 0.55)
        foot = QPointF(hip.x() + fwd, ground - lift)

        painter.setPen(leg_pen)
        painter.drawLine(hip, knee)
        painter.drawLine(knee, foot)
        feet.append((foot.x(), foot.y(), fwd))

    # Ступни-подошвы (эллипсы) рисуем поверх линий ног.
    for fx, fy, fwd in feet:
        painter.setPen(QPen(QColor(*_rgb(glow), 200), max(1.0, 1.8 * s)))
        painter.setBrush(QColor(*_rgb(mid), 235))
        painter.drawEllipse(QPointF(fx, fy), 11 * s, 5 * s)
        # пальцы-штрихи на подошве
        painter.setPen(QPen(QColor(*_rgb(glow), 150), max(0.7, 1.1 * s)))
        for i in range(3):
            painter.drawPoint(QPointF(fx - 4 * s + i * 4 * s, fy - 2 * s))

    # --- сфера-тело ---------------------------------------------------
    painter.save()
    if tilt:
        painter.translate(cx, cy)
        painter.rotate(tilt)
        painter.translate(-cx, -cy)

    # внешнее свечение
    halo = QRadialGradient(QPointF(cx, cy), R * 2.0)
    halo.setColorAt(0.0, QColor(*_rgb(glow), int(70 + 70 * level)))
    halo.setColorAt(0.5, QColor(*base, int(28 + 30 * level)))
    halo.setColorAt(1.0, QColor(0, 0, 0, 0))
    painter.setPen(Qt.NoPen)
    painter.setBrush(halo)
    painter.drawEllipse(QPointF(cx, cy), R * 2.0, R * 2.0)

    # орбита
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QColor(*base, 90), max(0.8, 1.2 * s)))
    painter.drawEllipse(QPointF(cx, cy), R * 1.28, R * 1.28)
    arc_start = int(((phase * 40) % 360) * 16)
    painter.setPen(QPen(QColor(*_rgb(glow), 190), max(1.0, 1.8 * s)))
    painter.drawArc(
        QRectF(cx - R * 1.28, cy - R * 1.28, R * 2.56, R * 2.56), arc_start, 60 * 16
    )

    # сама сфера
    body = QRadialGradient(
        QPointF(cx - R * 0.35, cy - R * 0.42), R * 2.0
    )
    body.setColorAt(0.0, QColor(*_rgb(glow), 255))
    body.setColorAt(0.30, QColor(*base, 250))
    body.setColorAt(0.72, QColor(*_rgb(dark), 252))
    body.setColorAt(1.0, QColor(3, 10, 18, 255))
    painter.setBrush(body)
    painter.setPen(QPen(QColor(*_rgb(glow), 210), max(1.2, 2.0 * s)))
    painter.drawEllipse(QPointF(cx, cy), R, R * squash)

    # блик
    painter.setBrush(QColor(255, 255, 255, 55))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(
        QPointF(cx - R * 0.34, cy - R * 0.42), R * 0.30, R * 0.20
    )

    # --- лицо ---------------------------------------------------------
    eye_dx = R * 0.36
    eye_y = cy - R * 0.14
    eye_r = R * 0.17
    look = max(-1.0, min(1.0, gaze)) * eye_r * 0.6

    painter.setPen(Qt.NoPen)
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        if blink or state == "sleep":
            painter.setPen(QPen(QColor(*_rgb(glow), 240), max(1.2, 2.2 * s)))
            painter.drawLine(
                QPointF(ex - eye_r, eye_y), QPointF(ex + eye_r, eye_y)
            )
            painter.setPen(Qt.NoPen)
        else:
            painter.setBrush(QColor(235, 255, 255, 245))
            painter.drawEllipse(QPointF(ex, eye_y), eye_r, eye_r * 1.08)
            painter.setBrush(QColor(8, 18, 28, 255))
            painter.drawEllipse(
                QPointF(ex + look, eye_y + eye_r * 0.18), eye_r * 0.48, eye_r * 0.52
            )
            painter.setBrush(QColor(235, 255, 255, 245))

    # рот
    mouth_y = cy + R * 0.42
    mouth_w = R * 0.52
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QColor(*_rgb(glow), 235), max(1.2, 2.0 * s)))
    if state == "sleep":
        painter.drawArc(
            QRectF(cx - mouth_w * 0.5, mouth_y - mouth_w * 0.3, mouth_w, mouth_w * 0.7),
            200 * 16, 140 * 16,
        )
    elif state == "speaking":
        openness = 0.35 + 0.65 * abs(math.sin(phase * 9.5)) * (0.4 + 0.6 * level)
        painter.setBrush(QColor(*_rgb(glow), 190))
        painter.drawEllipse(
            QPointF(cx, mouth_y), mouth_w * 0.34, R * 0.20 * openness
        )
    elif state in {"happy", "confirm"}:
        painter.drawArc(
            QRectF(cx - mouth_w * 0.6, mouth_y - mouth_w * 0.55, mouth_w * 1.2, mouth_w),
            200 * 16, 140 * 16,
        )
    elif state == "error":
        painter.drawArc(
            QRectF(cx - mouth_w * 0.6, mouth_y - mouth_w * 0.15, mouth_w * 1.2, mouth_w),
            20 * 16, 140 * 16,
        )
    elif state == "thinking":
        painter.drawEllipse(QPointF(cx, mouth_y), mouth_w * 0.16, mouth_w * 0.16)
    else:
        painter.drawLine(
            QPointF(cx - mouth_w * 0.35, mouth_y), QPointF(cx + mouth_w * 0.35, mouth_y)
        )

    painter.restore()  # конец наклона

    # --- руки и ладони ------------------------------------------------
    arm_pen = QPen(QColor(*base, 215), max(1.6, 3.4 * s))
    arm_pen.setCapStyle(Qt.RoundCap)

    palms: list[QPointF] = []
    for sign in (-1, 1):
        shoulder = QPointF(cx + sign * R * 0.92, cy + R * 0.08)

        if typing:
            tap = math.sin(phase * 17.0 + sign * 1.6) * 2.6 * s
            elbow = QPointF(shoulder.x() + sign * 9 * s, shoulder.y() + 9 * s)
            palm = QPointF(cx + sign * R * 0.42, cy + R * 1.02 + tap)
        elif pressing and sign > 0:
            # правая ладонь вытянута к точке клика
            elbow = QPointF(shoulder.x() + 14 * s, shoulder.y() + 12 * s)
            palm = QPointF(cx + R * 1.55, cy + R * 1.15)
        elif dragging:
            # при переносе ладони беспомощно болтаются
            sway = math.sin(phase * 9.0 + (0 if sign < 0 else 1.4))
            elbow = QPointF(shoulder.x() + sign * 7 * s, shoulder.y() + 10 * s)
            palm = QPointF(
                shoulder.x() + sign * (9 + 2 * sway) * s,
                shoulder.y() + (22 + 3 * sway) * s,
            )
        elif state == "sleep":
            elbow = QPointF(shoulder.x() + sign * 6 * s, shoulder.y() + 8 * s)
            palm = QPointF(cx + sign * R * 0.78, cy + R * 0.72)
        elif state == "working" and sign > 0:
            wave = math.sin(phase * 6.5)
            elbow = QPointF(shoulder.x() + 9 * s, shoulder.y() - 6 * s)
            palm = QPointF(
                shoulder.x() + (15 + 3 * wave) * s, shoulder.y() - (16 + 4 * wave) * s
            )
        elif state in {"speaking", "happy", "confirm"}:
            wave = math.sin(phase * 6.0 + (0 if sign < 0 else 1.3))
            elbow = QPointF(
                shoulder.x() + sign * (7 + 2 * wave) * s, shoulder.y() + 7 * s
            )
            palm = QPointF(
                shoulder.x() + sign * (13 + 4 * wave) * s,
                shoulder.y() + (2 + 7 * wave) * s,
            )
        else:
            idle_sway = math.sin(phase * 1.9 + (0 if sign < 0 else 0.8))
            elbow = QPointF(
                shoulder.x() + sign * 6 * s, shoulder.y() + (10 + 2 * idle_sway) * s
            )
            palm = QPointF(
                shoulder.x() + sign * (9 + 2 * idle_sway) * s,
                shoulder.y() + (21 + 3 * idle_sway) * s,
            )

        painter.setPen(arm_pen)
        painter.drawLine(shoulder, elbow)
        painter.drawLine(elbow, palm)
        palms.append(palm)

    # ладони
    for idx, palm in enumerate(palms):
        sign = -1 if idx == 0 else 1
        painter.setPen(QPen(QColor(*_rgb(glow), 210), max(1.0, 1.6 * s)))
        painter.setBrush(QColor(*_rgb(mid), 235))
        painter.drawEllipse(palm, 7.5 * s, 6.5 * s)
        # пальцы
        painter.setPen(QPen(QColor(*_rgb(glow), 175), max(0.8, 1.3 * s)))
        for i in range(3):
            fx = palm.x() + sign * (2 + i * 1.6) * s
            fy = palm.y() - 4 * s + i * 4 * s
            painter.drawLine(
                QPointF(palm.x() + sign * 3 * s, palm.y() - 4 * s + i * 4 * s),
                QPointF(fx + sign * 3 * s, fy),
            )

    if pressing and len(palms) > 1:
        # «нажатие»: расходящаяся волна от ладони
        pulse = (phase * 2.8) % 1.0
        painter.setBrush(Qt.NoBrush)
        painter.setPen(
            QPen(QColor(*_rgb(glow), int(210 * (1.0 - pulse))), max(1.0, 2.2 * s))
        )
        r = (7 + 13 * pulse) * s
        painter.drawEllipse(palms[1], r, r)

    # --- предметы -----------------------------------------------------

    if objects:
        kind = _STATE_OBJECTS.get(state, "")
        if kind == "keyboard":
            _draw_keyboard(painter, cx, cy + R * 1.12, s, base, glow, phase)
        elif kind == "pillow":
            _draw_pillow(painter, cx, cy + R * 0.92, s, glow, phase)
            _draw_zzz(painter, cx + R * 0.9, cy - R * 0.9, s, glow, phase)
        elif kind:
            palm = palms[1] if len(palms) > 1 else QPointF(cx, cy)
            ox = palm.x() + 13 * s
            oy = palm.y() - 9 * s + math.sin(phase * 2.4) * 2.6 * s
            _draw_object(painter, kind, ox, oy, 8.5 * s, glow, mid, phase)

    painter.restore()


def _draw_keyboard(
    painter: QPainter, x: float, y: float, s: float,
    base: tuple[int, int, int], glow: QColor, phase: float,
) -> None:
    kb_w, kb_h = 58 * s, 16 * s
    rect = QRectF(x - kb_w / 2, y, kb_w, kb_h)
    painter.setBrush(QColor(10, 22, 34, 240))
    painter.setPen(QPen(QColor(*base, 210), max(1.0, 1.6 * s)))
    painter.drawRoundedRect(rect, 3.5 * s, 3.5 * s)

    # подсвечиваем «нажимаемые» клавиши
    painter.setPen(Qt.NoPen)
    for row in range(2):
        for col in range(8):
            kx = rect.left() + 4 * s + col * (kb_w - 8 * s) / 8
            ky = rect.top() + 4 * s + row * 6 * s
            on = (int(phase * 12) + row * 3 + col) % 5 == 0
            painter.setBrush(QColor(*_rgb(glow), 235 if on else 70))
            painter.drawRoundedRect(
                QRectF(kx, ky, 4.2 * s, 4.2 * s), 1.0 * s, 1.0 * s
            )


def _draw_pillow(
    painter: QPainter, x: float, y: float, s: float, glow: QColor, phase: float,
) -> None:
    w, h = 62 * s, 20 * s
    rect = QRectF(x - w / 2, y, w, h)
    painter.save()
    painter.translate(x, y + h / 2)
    painter.rotate(0.0)
    painter.translate(-x, -(y + h / 2))
    painter.setBrush(QColor(226, 238, 248, 240))
    painter.setPen(QPen(QColor(*_rgb(glow), 170), max(1.0, 1.4 * s)))
    painter.drawRoundedRect(rect, h * 0.55, h * 0.55)
    # шов посередине
    painter.setPen(QPen(QColor(150, 175, 195, 200), max(0.8, 1.0 * s)))
    painter.drawLine(
        QPointF(rect.left() + w * 0.18, rect.center().y()),
        QPointF(rect.right() - w * 0.18, rect.center().y()),
    )
    painter.restore()


def _draw_zzz(
    painter: QPainter, x: float, y: float, s: float, glow: QColor, phase: float,
) -> None:
    font = QFont("Segoe UI", max(7, int(9 * s)))
    font.setBold(True)
    painter.setFont(font)
    for i in range(3):
        t = (phase * 0.55 + i * 0.33) % 1.0
        alpha = int(230 * (1.0 - t))
        if alpha <= 0:
            continue
        painter.setPen(QColor(*_rgb(glow), alpha))
        painter.drawText(
            QPointF(x + t * 18 * s, y - t * 26 * s),
            "z" if i % 2 == 0 else "Z",
        )


def _draw_object(
    painter: QPainter,
    kind: str,
    x: float,
    y: float,
    size: float,
    glow: QColor,
    dark: QColor,
    phase: float,
) -> None:
    """Небольшой предмет в ладони."""
    painter.save()
    painter.translate(x, y)
    pen = QPen(QColor(*_rgb(glow), 235), max(1.0, size * 0.24))
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    if kind == "gear":
        painter.save()
        painter.rotate((phase * 90) % 360)
        painter.drawEllipse(QPointF(0, 0), size * 0.6, size * 0.6)
        for i in range(8):
            a = math.radians(i * 45)
            painter.drawLine(
                QPointF(math.cos(a) * size * 0.6, math.sin(a) * size * 0.6),
                QPointF(math.cos(a) * size * 0.9, math.sin(a) * size * 0.9),
            )
        painter.restore()
    elif kind == "search":
        painter.drawEllipse(QPointF(-size * 0.15, -size * 0.15), size * 0.52, size * 0.52)
        painter.drawLine(QPointF(size * 0.26, size * 0.26), QPointF(size * 0.9, size * 0.9))
    elif kind == "note":
        painter.drawEllipse(QPointF(-size * 0.25, size * 0.45), size * 0.28, size * 0.22)
        painter.drawLine(QPointF(size * 0.02, size * 0.45), QPointF(size * 0.02, -size * 0.68))
        painter.drawLine(QPointF(size * 0.02, -size * 0.68), QPointF(size * 0.58, -size * 0.5))
    elif kind == "star":
        pts = []
        for i in range(10):
            a = math.radians(-90 + i * 36)
            r = size * (0.92 if i % 2 == 0 else 0.40)
            pts.append(QPointF(math.cos(a) * r, math.sin(a) * r))
        painter.setBrush(QColor(*_rgb(glow), 215))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF(pts))
    elif kind == "alert":
        painter.drawLine(QPointF(0, -size * 0.82), QPointF(0, size * 0.22))
        painter.setBrush(QColor(*_rgb(glow), 235))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(0, size * 0.6), size * 0.15, size * 0.15)
    elif kind == "eye":
        painter.drawEllipse(QPointF(0, 0), size * 0.82, size * 0.48)
        painter.setBrush(QColor(*_rgb(glow), 235))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(0, 0), size * 0.26, size * 0.26)
    painter.restore()



# ---------------------------------------------------------------------------
# Окно аватара на рабочем столе
# ---------------------------------------------------------------------------

BASE_W = 190.0
BASE_H = 250.0
# Сверху холста есть «воздух» над сферой — обрезаем его, чтобы окно
# не занимало лишнее место на экране.
TOP_CROP = 80.0


class AvatarWindow(QWidget):
    """Отдельное окно-компаньон поверх всех программ."""

    def __init__(self, settings=None, parent=None) -> None:
        super().__init__(parent)
        self._settings = settings

        flags = (
            Qt.FramelessWindowHint
            | Qt.Tool
            | Qt.WindowStaysOnTopHint
            | Qt.NoDropShadowWindowHint
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.OpenHandCursor)
        self.setWindowTitle("J.A.R.V.I.S.")

        # настройки
        self._scale = float(getattr(settings, "avatar_scale", 1.0) or 1.0)
        self._speed = float(getattr(settings, "avatar_speed", 1.0) or 1.0)
        self._opacity = float(getattr(settings, "avatar_opacity", 0.95) or 0.95)
        self._objects = bool(getattr(settings, "avatar_objects", True))
        self._accent = str(getattr(settings, "avatar_accent", "") or "")
        self._subtitles = bool(getattr(settings, "avatar_subtitles", True))
        self._wander = bool(getattr(settings, "avatar_wander", True))
        self._on_top = bool(getattr(settings, "avatar_always_on_top", True))

        # состояние персонажа
        self._state = "sleep"
        self._typing = False
        self._level = 0.0
        self._phase = 0.0
        self._blink = False
        self._blink_until = 0.0
        self._next_blink = time.monotonic() + random.uniform(2.5, 6.0)
        self._gaze = 0.0
        self._press_until = 0.0

        # движение
        self._pos = QPointF(0.0, 0.0)      # центр сферы в экранных координатах
        self._target = QPointF(0.0, 0.0)
        self._moving = False
        self._pause_until = 0.0
        self._history: list[QPointF] = []
        self._initialized = False

        # перетаскивание
        self._dragging = False
        self._drag_offset = QPoint()

        # запрос «иди и нажми» из фонового потока
        self._lock = threading.Lock()
        self._pending_cursor: tuple[float, float, threading.Event] | None = None
        self._cursor_event: threading.Event | None = None
        self._state_before_cursor: str = ""
        self._cursor_walk = False

        # субтитры
        self._subtitle = ""
        self._subtitle_until = 0.0

        self._last_tick = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self.setWindowOpacity(max(0.2, min(1.0, self._opacity)))
        self._resize_for_subtitle()

    # ------------------------------------------------------------------
    # Геометрия
    # ------------------------------------------------------------------
    def _base_size(self) -> tuple[int, int]:
        return (
            max(110, int(BASE_W * self._scale)),
            max(95, int((BASE_H - TOP_CROP) * self._scale)),
        )

    def _char_metrics(self) -> tuple[int, int, float, float, float]:
        """(ширина, высота, масштаб, центр сферы X, центр сферы Y)."""
        w, h = self._base_size()
        s = max(0.35, min(w / BASE_W, h / (BASE_H - TOP_CROP)))
        return w, h, s, w / 2.0, (BASE_H - 79.0) * s - TOP_CROP * s

    def _screen_rect(self) -> QRect:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return QRect(0, 0, 1280, 720)
        return screen.availableGeometry()

    def _apply_position(self, body: QPointF) -> None:
        w, h, _s, cx_off, cy_off = self._char_metrics()
        self._pos = QPointF(body)
        self.move(int(round(body.x() - cx_off)), int(round(body.y() - cy_off)))

    def _sync_from_window(self) -> None:
        _w, _h, _s, cx_off, cy_off = self._char_metrics()
        self._pos = QPointF(self.x() + cx_off, self.y() + cy_off)

    # ------------------------------------------------------------------
    # Блуждание
    # ------------------------------------------------------------------
    def _pick_target(self) -> QPointF:
        rect = self._screen_rect()
        w, h, s, _cx, _cy = self._char_metrics()
        left = rect.left() + w * 0.55
        right = rect.right() - w * 0.55
        top = rect.top() + (h - 79.0 * s) + 8.0
        bottom = rect.bottom() - 90.0 * s
        if right <= left:
            right = left + 40
        if bottom <= top:
            bottom = top + 40

        best = None
        best_score = -1.0
        for _ in range(14):
            cand = QPointF(
                random.uniform(left, right), random.uniform(top, bottom)
            )
            near = min(
                (math.hypot(cand.x() - p.x(), cand.y() - p.y()) for p in self._history),
                default=1e9,
            )
            if near > rect.width() * 0.35:
                return cand
            if near > best_score:
                best_score = near
                best = cand
        return best or QPointF((left + right) / 2, (top + bottom) / 2)

    def _start_walk(self, target: QPointF) -> None:
        self._target = QPointF(target)
        self._moving = True
        self._pause_until = 0.0

    def _arrived(self, now: float) -> None:
        self._moving = False
        self._pos = QPointF(self._target)
        self._apply_position(self._pos)
        self._history.append(QPointF(self._target))
        if len(self._history) > 6:
            self._history.pop(0)
        if self._state == "sleep":
            self._pause_until = now + random.uniform(11.0, 26.0)
        else:
            self._pause_until = now + random.uniform(1.2, 5.0)

    # ------------------------------------------------------------------
    # Внешний API
    # ------------------------------------------------------------------
    def set_state(self, state: str) -> None:
        state = state or "idle"
        if state == self._state:
            return
        self._state = state
        now = time.monotonic()
        if state == "sleep" and not self._cursor_walk:
            # Во сне персонаж замирает на месте и больше не блуждает.
            self._moving = False
            self._target = QPointF(self._pos)
            self._gaze = 0.0
        elif not self._moving and not self._dragging:
            self._pause_until = min(self._pause_until, now + 0.4)
        self.update()

    def set_typing(self, typing: bool) -> None:
        typing = bool(typing)
        if typing != self._typing:
            self._typing = typing
            self.update()

    def set_level(self, value: float) -> None:
        self._level = max(0.0, min(1.0, float(value or 0.0)))

    def speak(self, text: str) -> None:
        self.show_subtitle(text)

    def show_subtitle(self, text: str, seconds: float | None = None) -> None:
        if not self._subtitles:
            return
        text = " ".join(str(text or "").split())
        if not text:
            return
        if seconds is None:
            seconds = max(3.0, min(14.0, len(text) / 13.0))
        self._subtitle = text
        self._subtitle_until = time.monotonic() + max(1.5, float(seconds))
        self._resize_for_subtitle()
        self.update()

    def request_walk(self, x: float, y: float, timeout: float = 15.0) -> bool:
        """Потокобезопасно: дойти до экранной точки (x, y) и «нажать»."""
        event = threading.Event()
        with self._lock:
            self._pending_cursor = (float(x), float(y), event)
        return bool(event.wait(timeout))

    def apply_settings(self, settings) -> None:
        self._settings = settings
        self._speed = float(getattr(settings, "avatar_speed", 1.0) or 1.0)
        self._objects = bool(getattr(settings, "avatar_objects", True))
        self._accent = str(getattr(settings, "avatar_accent", "") or "")
        self._subtitles = bool(getattr(settings, "avatar_subtitles", True))
        self._wander = bool(getattr(settings, "avatar_wander", True))
        self._opacity = float(getattr(settings, "avatar_opacity", 0.95) or 0.95)
        self.setWindowOpacity(max(0.2, min(1.0, self._opacity)))

        on_top = bool(getattr(settings, "avatar_always_on_top", True))
        if on_top != self._on_top:
            self._on_top = on_top
            flags = self.windowFlags()
            if on_top:
                flags |= Qt.WindowStaysOnTopHint
            else:
                flags &= ~Qt.WindowStaysOnTopHint
            self.setWindowFlags(flags)
            if self.isVisible():
                self.show()

        scale = float(getattr(settings, "avatar_scale", 1.0) or 1.0)
        if abs(scale - self._scale) > 1e-3:
            body = QPointF(self._pos)
            self._scale = scale
            self._resize_for_subtitle()
            if self._initialized:
                self._apply_position(body)

        if not self._subtitles:
            self._subtitle = ""
            self._subtitle_until = 0.0
            self._resize_for_subtitle()

        if not self._wander:
            self._moving = False
            self._target = QPointF(self._pos)

        self.update()

    # ------------------------------------------------------------------
    # Цикл анимации
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        try:
            now = time.monotonic()
            dt = min(0.12, max(0.0, now - self._last_tick))
            self._last_tick = now
            self._phase += dt

            # моргание
            if now >= self._next_blink:
                self._blink = True
                self._blink_until = now + 0.12
                self._next_blink = now + random.uniform(2.4, 6.5)
            elif self._blink and now >= self._blink_until:
                self._blink = False

            # запрос «иди к курсору»
            request = None
            with self._lock:
                if self._pending_cursor is not None:
                    request = self._pending_cursor
                    self._pending_cursor = None
            if request is not None:
                px, py, event = request
                w, h, s, _cx, _cy = self._char_metrics()
                del w, h
                R = 46.0 * s
                target = QPointF(px - 1.55 * R, py - 1.15 * R)
                rect = self._screen_rect()
                target.setX(min(max(target.x(), rect.left() + 60 * s), rect.right() - 60 * s))
                target.setY(min(max(target.y(), rect.top() + 80 * s), rect.bottom() - 80 * s))
                self._cursor_event = event
                self._cursor_walk = True
                # Если аватар спал — на время перехода «просыпается».
                self._state_before_cursor = self._state
                if self._state == "sleep":
                    self._state = "idle"
                self._start_walk(target)

            # движение
            # Во сне персонаж стоит на месте: ни шага, ни блуждания.
            frozen = self._state == "sleep" and not self._cursor_walk
            if frozen:
                self._moving = False
            if self._moving and not self._dragging and not frozen:
                dx = self._target.x() - self._pos.x()
                dy = self._target.y() - self._pos.y()
                dist = math.hypot(dx, dy)
                speed = max(40.0, 165.0 * self._speed)
                step = speed * dt
                if dist <= max(step, 2.0):
                    self._arrived(now)
                    if self._cursor_walk:
                        self._cursor_walk = False
                        self._press_until = now + 0.55
                        self._set_click_through(True)
                        QTimer.singleShot(600, lambda: self._set_click_through(False))
                        event = getattr(self, "_cursor_event", None)
                        self._cursor_event = None
                        if event is not None:
                            event.set()
                        self._pause_until = now + 0.5
                        prev = getattr(self, "_state_before_cursor", None)
                        self._state_before_cursor = None
                        if prev:
                            self._state = prev
                else:
                    self._pos = QPointF(
                        self._pos.x() + dx / dist * step,
                        self._pos.y() + dy / dist * step,
                    )
                    self._apply_position(self._pos)
                    self._gaze = max(-1.0, min(1.0, dx / 90.0))
            elif not self._dragging and not frozen:
                self._gaze *= 0.92
                if self._wander and now >= self._pause_until:
                    self._start_walk(self._pick_target())

            # субтитры
            if self._subtitle and now >= self._subtitle_until:
                self._subtitle = ""
                self._resize_for_subtitle()

            self.update()
        except Exception:
            log.debug("avatar tick failed", exc_info=True)

    # ------------------------------------------------------------------
    # События мыши (перетаскивание)
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._moving = False
            self._press_until = 0.0
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            self.setCursor(Qt.ClosedHandCursor)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self._sync_from_window()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.OpenHandCursor)
            self._sync_from_window()
            # после переноса сразу продолжаем жить своей жизнью
            self._pause_until = 0.0
            if self._wander and self._state != "sleep":
                self._start_walk(self._pick_target())
            else:
                self._target = QPointF(self._pos)
            self.update()
        super().mouseReleaseEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            self._resize_for_subtitle()
            target = self._pick_target()
            self._apply_position(target)
            self._target = QPointF(target)
            self._moving = False
            self._pause_until = time.monotonic() + random.uniform(0.6, 2.0)

    def closeEvent(self, event) -> None:
        try:
            self._timer.stop()
        except Exception:
            pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Отрисовка
    # ------------------------------------------------------------------
    def _pressing(self) -> bool:
        return time.monotonic() < self._press_until

    def _subtitle_font(self) -> QFont:
        font = QFont("Segoe UI", max(7, int(9.5 * self._scale)))
        return font

    def _subtitle_height(self) -> int:
        if not self._subtitle:
            return 0
        w, _h = self._base_size()
        metrics = QFontMetrics(self._subtitle_font())
        rect = metrics.boundingRect(
            QRect(0, 0, max(40, w - 26), 10000),
            Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignVCenter,
            self._subtitle,
        )
        return int(rect.height() + 20)

    def _resize_for_subtitle(self) -> None:
        w, h = self._base_size()
        extra = self._subtitle_height() if self._subtitles else 0
        new_h = h + extra
        if self.width() != w or self.height() != new_h:
            self.resize(w, new_h)

    def paintEvent(self, event) -> None:
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            w, h, s, _cx, _cy = self._char_metrics()

            state = self._state
            if state == "sleep" and self._moving:
                state = "idle"

            painter.save()
            painter.translate(0.0, -TOP_CROP * s)
            draw_character(
                painter,
                QRectF(0, 0, BASE_W * s, BASE_H * s),
                state,
                self._phase,
                self._blink,
                objects=self._objects,
                typing=self._typing,
                level=self._level,
                accent=self._accent,
                walking=self._moving,
                gaze=self._gaze,
                dragging=self._dragging,
                pressing=self._pressing(),
            )
            painter.restore()
            self._paint_subtitle(painter, w, h)
            painter.end()
        except Exception:
            log.debug("avatar paint failed", exc_info=True)

    def _paint_subtitle(self, painter: QPainter, w: int, h: int) -> None:
        if not self._subtitle or not self._subtitles:
            return
        extra = self._subtitle_height()
        if extra <= 0:
            return
        rect = QRectF(5.0, h + 1.0, w - 10.0, extra - 5.0)
        base = _STATE_COLORS.get(self._state, _STATE_COLORS["idle"])
        if self._accent:
            c = QColor(self._accent)
            if c.isValid():
                base = (c.red(), c.green(), c.blue())
        glow = _mix(base, (255, 255, 255), 0.5)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(6, 16, 26, 218))
        painter.drawRoundedRect(rect, 9, 9)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(*_rgb(glow), 200), 1.2))
        painter.drawRoundedRect(rect, 9, 9)

        painter.setFont(self._subtitle_font())
        painter.setPen(QColor(228, 245, 255, 245))
        painter.drawText(
            rect.adjusted(8, 6, -8, -6),
            Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignVCenter,
            self._subtitle,
        )

    # ------------------------------------------------------------------
    # Пропускать клики насквозь в момент «нажатия»
    # ------------------------------------------------------------------
    def _set_click_through(self, on: bool) -> None:
        if not sys.platform.startswith("win"):
            return
        try:
            import ctypes

            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00080000
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if on:
                style |= WS_EX_TRANSPARENT | WS_EX_LAYERED
            else:
                style &= ~WS_EX_TRANSPARENT
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:
            log.debug("click-through toggle failed", exc_info=True)


# ---------------------------------------------------------------------------
# Превью в настройках
# ---------------------------------------------------------------------------

class AvatarPreview(QWidget):
    """Живое превью аватара для вкладки настроек."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state = "idle"
        self._objects = True
        self._accent = ""
        self._phase = 0.0
        self._blink = False
        self._next_blink = time.monotonic() + 3.0
        self._blink_until = 0.0
        self.setMinimumSize(170, 300)
        self._timer = QTimer(self)
        self._timer.setInterval(45)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_state(self, state: str) -> None:
        self._state = state or "idle"
        self.update()

    def set_objects(self, enabled: bool) -> None:
        self._objects = bool(enabled)
        self.update()

    def set_accent(self, accent: str) -> None:
        self._accent = accent or ""
        self.update()

    def _tick(self) -> None:
        self._phase += 0.045
        now = time.monotonic()
        if now >= self._next_blink:
            self._blink = True
            self._blink_until = now + 0.12
            self._next_blink = now + random.uniform(2.5, 6.0)
        elif self._blink and now >= self._blink_until:
            self._blink = False
        self.update()

    def paintEvent(self, event) -> None:
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            panel = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            painter.setPen(QPen(QColor(60, 90, 120, 160), 1.0))
            painter.setBrush(QColor(8, 18, 28, 210))
            painter.drawRoundedRect(panel, 12, 12)

            inner = panel.adjusted(6, 10, -6, -10)
            visible_h = BASE_H - TOP_CROP
            s = max(
                0.25,
                min(inner.width() / BASE_W, inner.height() / visible_h),
            )
            tx = inner.center().x() - BASE_W * s / 2.0
            ty = inner.center().y() - visible_h * s / 2.0 + TOP_CROP * s

            painter.save()
            painter.setClipRect(inner)
            painter.translate(tx, ty)
            draw_character(
                painter,
                QRectF(0, 0, BASE_W * s, BASE_H * s),
                self._state,
                self._phase,
                self._blink,
                objects=self._objects,
                accent=self._accent,
            )
            painter.restore()
            painter.end()
        except Exception:
            log.debug("avatar preview paint failed", exc_info=True)
