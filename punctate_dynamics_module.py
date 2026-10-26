# -*- coding: utf-8 -*-
"""
LinkNex Module 5 - Punctate Organelle Dynamics
==============================================

Workflow
--------
1. Open one/multiple time-lapse TIFF files OR a folder and browse the raw time-lapse in the main Working View.
2. Split each time-lapse into a same-name folder and build a single viewer time-lapse stack.
3. Generate binary masks and build a mask stack for fast frame browsing.
4. Manually review/correct masks in the main Working View.
5. Run latest saved mask -> PCD -> instance -> dynamic tracking/network, then build an instance stack.
6. Display the shared background + mask / instance / dynamic overlay from the prebuilt stacks.
7. Export exactly that view as tif_instance_overlay_tracks.mp4.

The cross-frame tracker never uses per-frame RGB as temporal identity.
"""

from __future__ import annotations

import colorsys
import contextlib
import json
import math
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import tifffile

from PySide6.QtCore import Qt, QObject, QPointF, QRectF, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QBrush, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from backends import dynamic_organelle_backend as dynamic_backend
from backends import image_to_mask_backend as mask_backend


NAVY = "#071D49"
DEEP_BLUE = "#003B70"
US_BLUE = "#0067B1"
CYAN = "#00AEEF"
OFFWHITE = "#F6F8FB"
TEXT = "#10243E"
MUTED = "#66788A"
BORDER = "#C9D4E2"
GREEN = "#1E9E5A"
RED = "#D94B4B"
WHITE = "#FFFFFF"

SUPPORTED_TIFFS = {".tif", ".tiff"}
INSTANCE_OVERLAY_ALPHA = 0.40
MASK_OVERLAY_ALPHA = 0.35


def natural_key(text: str):
    import re
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(text))]


def deterministic_track_rgb(track_id: int) -> Tuple[int, int, int]:
    hue = (int(track_id) * 0.618033988749895) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return tuple(int(round(v * 255)) for v in rgb)


def read_numeric_txt(path: Path) -> np.ndarray:
    last_error = None
    for delimiter in (",", None, "\t"):
        try:
            arr = np.loadtxt(path, delimiter=delimiter)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return np.asarray(arr)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read numeric TXT: {path}\n{last_error}")


def read_frame_2d(path: Path) -> np.ndarray:
    arr = np.asarray(tifffile.imread(path))
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return np.mean(arr[..., :3].astype(np.float32), axis=-1)
    while arr.ndim > 2:
        arr = np.max(arr, axis=0)
    return np.asarray(arr)


def normalize_gray_to_rgb(image: np.ndarray, display_range=None) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        gray = np.zeros(arr.shape, dtype=np.uint8)
    else:
        if display_range is None:
            lo = float(np.nanpercentile(arr[finite], 1.0))
            hi = float(np.nanpercentile(arr[finite], 99.7))
        else:
            lo, hi = map(float, display_range)
        if hi <= lo:
            hi = lo + 1.0
        gray = np.clip((arr - lo) / (hi - lo), 0, 1)
        gray = np.rint(gray * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def rgb_array_to_pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)



class DropPathListWidget(QListWidget):
    """QListWidget that accepts dragged TIFF files and folders."""

    pathsDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DropOnly)
        self.setToolTip("Drag TIFF file(s) or a folder here")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
            if any(p.is_dir() or (p.is_file() and p.suffix.lower() in SUPPORTED_TIFFS) for p in paths):
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        paths = [p for p in paths if p.is_dir() or (p.is_file() and p.suffix.lower() in SUPPORTED_TIFFS)]
        if paths:
            self.pathsDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


@dataclass
class DynamicsDataset:
    source_path: Path
    name: str
    frame_dir: Optional[Path] = None
    frames: List[Path] = field(default_factory=list)

    @property
    def ready_dir(self) -> Optional[Path]:
        return self.frame_dir

    @property
    def processed_dir(self) -> Optional[Path]:
        return None if self.frame_dir is None else self.frame_dir / "processed_output"


class FunctionWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, func: Callable[[], object]):
        super().__init__()
        self.func = func

    @Slot()
    def run(self):
        try:
            result = self.func()
        except Exception:
            self.error.emit(traceback.format_exc())
            return
        self.finished.emit(result)


class DynamicLogWorker(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, func: Callable[[Callable[[str], None]], object]):
        super().__init__()
        self.func = func

    @Slot()
    def run(self):
        try:
            result = self.func(self.progress.emit)
        except Exception:
            self.error.emit(traceback.format_exc())
            return
        self.finished.emit(result)


class _DynamicLogStream:
    """Mirror backend stdout/stderr to the popup and a UTF-8 log file."""
    def __init__(self, emit_line: Callable[[str], None], file_handle):
        self.emit_line = emit_line
        self.file_handle = file_handle

    def write(self, text):
        if text is None:
            return 0
        raw = str(text)
        if not raw:
            return 0
        clean = raw.replace("\r", "\n")
        self.file_handle.write(clean)
        self.file_handle.flush()
        for part in clean.split("\n"):
            line = part.strip()
            if line:
                self.emit_line(line)
        return len(raw)

    def flush(self):
        self.file_handle.flush()

    def isatty(self):
        return False


class DynamicLogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dynamic Analysis Log")
        self.resize(760, 480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self.state_label = QLabel("Dynamic analysis is running...")
        self.state_label.setStyleSheet(f"font-weight:700; color:{DEEP_BLUE};")
        layout.addWidget(self.state_label)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(5000)
        layout.addWidget(self.output, 1)

    @Slot(str)
    def append_line(self, text: str):
        self.output.appendPlainText(str(text))
        bar = self.output.verticalScrollBar()
        bar.setValue(bar.maximum())

    def mark_complete(self):
        self.state_label.setText("Dynamic analysis completed. Closing log...")
        self.state_label.setStyleSheet(f"font-weight:700; color:{GREEN};")

    def mark_failed(self):
        self.state_label.setText("Dynamic analysis failed. Log kept open for review.")
        self.state_label.setStyleSheet(f"font-weight:700; color:{RED};")




ROI_COLOR = QColor("#FFB000")
ROI_SELECTED_COLOR = QColor("#00AEEF")
ROI_HANDLE_PX = 10.0


@dataclass
class ROIRecord:
    name: str
    shape_type: str
    item: QGraphicsItem


class _EditableROIBase:
    """Mixin shared by editable ROI graphics items."""

    def _roi_setup(self, name: str, on_changed: Optional[Callable[[], None]] = None):
        self.roi_name = str(name)
        self._on_changed = on_changed
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(20)
        self._label = QGraphicsSimpleTextItem(self.roi_name, self)
        self._label.setBrush(QBrush(ROI_COLOR))
        self._label.setZValue(30)
        self._refresh_roi_style()
        self._update_label_position()

    def _refresh_roi_style(self):
        selected = bool(self.isSelected())
        color = ROI_SELECTED_COLOR if selected else ROI_COLOR
        pen = QPen(color, 2.0)
        pen.setCosmetic(True)
        self.setPen(pen)
        if hasattr(self, "_label"):
            self._label.setBrush(QBrush(color))

    def _update_label_position(self):
        try:
            bounds = self.boundingRect()
            self._label.setPos(bounds.left(), bounds.top() - 18)
        except Exception:
            pass

    def set_roi_name(self, name: str):
        self.roi_name = str(name)
        self._label.setText(self.roi_name)
        self._update_label_position()
        self._emit_changed()

    def _emit_changed(self):
        self._update_label_position()
        if callable(self._on_changed):
            self._on_changed()

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._refresh_roi_style()
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._emit_changed()
        return result

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._emit_changed()


class EditableRectROI(_EditableROIBase, QGraphicsRectItem):
    def __init__(self, rect: QRectF, name: str, on_changed=None):
        QGraphicsRectItem.__init__(self, rect.normalized())
        self._resize_corner = None
        self._roi_setup(name, on_changed)

    def _corner_points(self):
        r = self.rect()
        return [r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]

    def mousePressEvent(self, event):
        p = event.pos()
        corners = self._corner_points()
        dists = [math.hypot(p.x() - c.x(), p.y() - c.y()) for c in corners]
        idx = int(np.argmin(dists)) if dists else -1
        if idx >= 0 and dists[idx] <= ROI_HANDLE_PX:
            self._resize_corner = idx
            event.accept()
            return
        self._resize_corner = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize_corner is None:
            super().mouseMoveEvent(event)
            return
        p = event.pos()
        r = self.rect()
        opposite = [r.bottomRight(), r.bottomLeft(), r.topLeft(), r.topRight()][self._resize_corner]
        new_rect = QRectF(opposite, p).normalized()
        if new_rect.width() >= 2 and new_rect.height() >= 2:
            self.setRect(new_rect)
            self._emit_changed()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._resize_corner is not None:
            self._resize_corner = None
            self._emit_changed()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paint(self, painter, option, widget=None):
        QGraphicsRectItem.paint(self, painter, option, widget)
        if self.isSelected():
            painter.setPen(QPen(ROI_SELECTED_COLOR, 1.0))
            painter.setBrush(QBrush(QColor("#FFFFFF")))
            s = 6.0
            for c in self._corner_points():
                painter.drawRect(QRectF(c.x() - s / 2, c.y() - s / 2, s, s))

    def scene_rect(self) -> QRectF:
        return self.mapRectToScene(self.rect()).boundingRect()


class EditableCircleROI(_EditableROIBase, QGraphicsEllipseItem):
    def __init__(self, center: QPointF, radius: float, name: str, on_changed=None):
        r = max(1.0, float(radius))
        QGraphicsEllipseItem.__init__(self, QRectF(-r, -r, 2 * r, 2 * r))
        self.setPos(center)
        self._resizing = False
        self._roi_setup(name, on_changed)

    def radius(self) -> float:
        return float(self.rect().width() / 2.0)

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0.0, 0.0))

    def mousePressEvent(self, event):
        p = event.pos()
        dist = math.hypot(p.x(), p.y())
        if abs(dist - self.radius()) <= ROI_HANDLE_PX:
            self._resizing = True
            event.accept()
            return
        self._resizing = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._resizing:
            super().mouseMoveEvent(event)
            return
        p = event.pos()
        r = max(1.0, math.hypot(p.x(), p.y()))
        self.setRect(QRectF(-r, -r, 2 * r, 2 * r))
        self._emit_changed()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            self._emit_changed()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paint(self, painter, option, widget=None):
        QGraphicsEllipseItem.paint(self, painter, option, widget)
        if self.isSelected():
            r = self.radius()
            painter.setPen(QPen(ROI_SELECTED_COLOR, 1.0))
            painter.setBrush(QBrush(QColor("#FFFFFF")))
            s = 7.0
            painter.drawEllipse(QRectF(r - s / 2, -s / 2, s, s))


class EditablePolygonROI(_EditableROIBase, QGraphicsPolygonItem):
    def __init__(self, points: Sequence[QPointF], name: str, on_changed=None):
        QGraphicsPolygonItem.__init__(self, QPolygonF(list(points)))
        self._vertex_index = None
        self._roi_setup(name, on_changed)

    def mousePressEvent(self, event):
        p = event.pos()
        poly = self.polygon()
        dists = [math.hypot(p.x() - poly[i].x(), p.y() - poly[i].y()) for i in range(poly.count())]
        idx = int(np.argmin(dists)) if dists else -1
        if idx >= 0 and dists[idx] <= ROI_HANDLE_PX:
            self._vertex_index = idx
            event.accept()
            return
        self._vertex_index = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._vertex_index is None:
            super().mouseMoveEvent(event)
            return
        poly = self.polygon()
        poly.replace(self._vertex_index, event.pos())
        self.setPolygon(poly)
        self._emit_changed()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._vertex_index is not None:
            self._vertex_index = None
            self._emit_changed()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paint(self, painter, option, widget=None):
        QGraphicsPolygonItem.paint(self, painter, option, widget)
        if self.isSelected():
            painter.setPen(QPen(ROI_SELECTED_COLOR, 1.0))
            painter.setBrush(QBrush(QColor("#FFFFFF")))
            s = 6.0
            poly = self.polygon()
            for i in range(poly.count()):
                p = poly[i]
                painter.drawEllipse(QRectF(p.x() - s / 2, p.y() - s / 2, s, s))

    def scene_points(self) -> List[QPointF]:
        poly = self.polygon()
        return [self.mapToScene(poly[i]) for i in range(poly.count())]


class DynamicsCanvas(QGraphicsView):
    brushStroke = Signal(int, int, bool)
    brushStrokeFinished = Signal()
    roiCreated = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setZValue(0)
        self._scene.addItem(self._pixmap_item)
        self.setBackgroundBrush(QColor("#111111"))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._editing = False
        self._erase = False
        self._right_panning = False
        self._last_pan_pos = None
        self._image_shape = None

        self._roi_mode: Optional[str] = None
        self._roi_start: Optional[QPointF] = None
        self._roi_polygon_points: List[QPointF] = []
        self._roi_preview: Optional[QGraphicsItem] = None

    def set_rgb(self, rgb: np.ndarray):
        self._image_shape = rgb.shape[:2]
        pix = rgb_array_to_pixmap(rgb)
        self._pixmap_item.setPixmap(pix)
        self._scene.setSceneRect(0, 0, pix.width(), pix.height())
        if self.transform().m11() == 1.0:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def add_roi_item(self, item: QGraphicsItem):
        self._scene.addItem(item)

    def remove_roi_item(self, item: QGraphicsItem):
        if item.scene() is self._scene:
            self._scene.removeItem(item)

    def clear_roi_items(self):
        for item in list(self._scene.items()):
            if isinstance(item, (EditableRectROI, EditableCircleROI, EditablePolygonROI)):
                self._scene.removeItem(item)

    def set_mask_editing(self, enabled: bool, erase: bool = False):
        self._editing = bool(enabled)
        self._erase = bool(erase)
        if enabled:
            self.set_roi_draw_mode(None)
        # ROI graphics sit above the image and can otherwise intercept mouse
        # events. Keep them visible, but temporarily disable interaction while
        # painting the mask.
        for item in list(self._scene.items()):
            if isinstance(item, (EditableRectROI, EditableCircleROI, EditablePolygonROI)):
                item.setEnabled(not enabled)
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_roi_draw_mode(self, mode: Optional[str]):
        self._clear_roi_preview()
        self._roi_mode = mode
        self._roi_start = None
        self._roi_polygon_points = []
        if mode is not None:
            self._editing = False
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _scene_point_from_view(self, pos) -> QPointF:
        return self.mapToScene(int(pos.x()), int(pos.y()))

    def _point_in_image(self, p: QPointF) -> bool:
        if self._image_shape is None:
            return False
        h, w = self._image_shape
        return 0 <= p.x() < w and 0 <= p.y() < h

    def _clear_roi_preview(self):
        if self._roi_preview is not None and self._roi_preview.scene() is self._scene:
            self._scene.removeItem(self._roi_preview)
        self._roi_preview = None

    def _preview_pen(self):
        pen = QPen(ROI_SELECTED_COLOR, 2.0, Qt.DashLine)
        pen.setCosmetic(True)
        return pen

    def _update_polygon_preview(self, cursor: Optional[QPointF] = None):
        self._clear_roi_preview()
        if not self._roi_polygon_points:
            return
        path = QPainterPath(self._roi_polygon_points[0])
        for p in self._roi_polygon_points[1:]:
            path.lineTo(p)
        if cursor is not None:
            path.lineTo(cursor)
        self._roi_preview = QGraphicsPathItem(path)
        self._roi_preview.setPen(self._preview_pen())
        self._roi_preview.setZValue(40)
        self._scene.addItem(self._roi_preview)

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._right_panning = True
            self._last_pan_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._roi_mode is not None:
            p = self._scene_point_from_view(event.position())
            if not self._point_in_image(p):
                return
            if self._roi_mode in ("rectangle", "circle"):
                self._roi_start = p
                self._clear_roi_preview()
                if self._roi_mode == "rectangle":
                    preview = QGraphicsRectItem(QRectF(p, p))
                else:
                    preview = QGraphicsEllipseItem(QRectF(p, p))
                preview.setPen(self._preview_pen())
                preview.setZValue(40)
                self._roi_preview = preview
                self._scene.addItem(preview)
            elif self._roi_mode == "polygon":
                self._roi_polygon_points.append(p)
                self._update_polygon_preview()
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._editing:
            self._emit_brush(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._right_panning and self._last_pan_pos is not None:
            delta = event.position() - self._last_pan_pos
            self._last_pan_pos = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return

        if self._roi_mode is not None:
            p = self._scene_point_from_view(event.position())
            if self._roi_mode == "rectangle" and self._roi_start is not None and isinstance(self._roi_preview, QGraphicsRectItem):
                self._roi_preview.setRect(QRectF(self._roi_start, p).normalized())
                event.accept()
                return
            if self._roi_mode == "circle" and self._roi_start is not None and isinstance(self._roi_preview, QGraphicsEllipseItem):
                radius = math.hypot(p.x() - self._roi_start.x(), p.y() - self._roi_start.y())
                r = QRectF(self._roi_start.x() - radius, self._roi_start.y() - radius, 2 * radius, 2 * radius)
                self._roi_preview.setRect(r)
                event.accept()
                return
            if self._roi_mode == "polygon" and self._roi_polygon_points:
                self._update_polygon_preview(p)
                event.accept()
                return

        if self._editing and (event.buttons() & Qt.LeftButton):
            self._emit_brush(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._right_panning:
            self._right_panning = False
            self._last_pan_pos = None
            self.setCursor(Qt.CrossCursor if (self._editing or self._roi_mode) else Qt.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._roi_mode in ("rectangle", "circle") and self._roi_start is not None:
            p = self._scene_point_from_view(event.position())
            start = self._roi_start
            mode = self._roi_mode
            self._roi_start = None
            self._clear_roi_preview()
            if mode == "rectangle":
                rect = QRectF(start, p).normalized()
                if rect.width() >= 3 and rect.height() >= 3:
                    self.roiCreated.emit("rectangle", rect)
            else:
                radius = math.hypot(p.x() - start.x(), p.y() - start.y())
                if radius >= 2:
                    self.roiCreated.emit("circle", (start, float(radius)))
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._editing:
            self.brushStrokeFinished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self._roi_mode == "polygon":
            p = self._scene_point_from_view(event.position())
            if self._point_in_image(p):
                if not self._roi_polygon_points or math.hypot(
                    p.x() - self._roi_polygon_points[-1].x(),
                    p.y() - self._roi_polygon_points[-1].y(),
                ) > 1.0:
                    self._roi_polygon_points.append(p)
            if len(self._roi_polygon_points) >= 3:
                points = list(self._roi_polygon_points)
                self._roi_polygon_points = []
                self._clear_roi_preview()
                self.roiCreated.emit("polygon", points)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._roi_mode is not None:
            self._roi_polygon_points = []
            self._roi_start = None
            self._clear_roi_preview()
            self.set_roi_draw_mode(None)
            event.accept()
            return
        super().keyPressEvent(event)

    def _emit_brush(self, pos):
        scene_pos = self.mapToScene(int(pos.x()), int(pos.y()))
        x, y = int(round(scene_pos.x())), int(round(scene_pos.y()))
        if self._image_shape is None:
            return
        h, w = self._image_shape
        if 0 <= x < w and 0 <= y < h:
            self.brushStroke.emit(x, y, not self._erase)


class PunctateDynamicsWidget(QWidget):
    backRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.datasets: List[DynamicsDataset] = []
        self.current_dataset_index = -1
        self.current_frame_index = 0
        self.view_mode = "preview"  # preview | mask | dynamic
        self.overlay_visible = True
        self.current_mask: Optional[np.ndarray] = None
        self.current_mask_path: Optional[Path] = None
        self.current_mask_dirty = False
        self._threads: List[Tuple[QThread, FunctionWorker]] = []
        self._tracked_cache = {}
        self._display_range_cache = {}

        # Lightweight LRU caches keep slider browsing responsive without loading
        # an entire long time-lapse into RAM. 2400x2400 TIFFs are read once, then
        # reused while the user moves around nearby frames.
        self._frame_image_cache = OrderedDict()
        self._mask_image_cache = OrderedDict()
        self._numeric_txt_cache = OrderedDict()
        self._source_timepoint_cache = OrderedDict()
        self._external_timepoint_cache = OrderedDict()
        self._external_display_range_cache = {}
        self._viewer_stack_cache = {}
        self._cache_max_items = 10

        # Optional second-channel time-lapse used as the display/export background.
        # Stored per dataset so batch-loaded datasets can each keep their own channel.
        self._external_background_paths: Dict[str, Path] = {}
        self._external_background_active: Dict[str, bool] = {}

        # Debounce expensive image refresh while the slider is actively dragged.
        self._frame_refresh_timer = QTimer(self)
        self._frame_refresh_timer.setSingleShot(True)
        self._frame_refresh_timer.setInterval(35)
        self._frame_refresh_timer.timeout.connect(self._commit_frame_change)

        self.rois: List[ROIRecord] = []
        self._roi_counter = 1
        self._loading_rois = False
        self._build_ui()
        self._apply_style()
        self._configure_module5_controls()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        home = QPushButton("← Home")
        home.clicked.connect(self.backRequested.emit)
        title = QLabel("Punctate Organelle Dynamics")
        title.setObjectName("DynTitle")
        top.addWidget(home)
        top.addStretch(1)
        top.addWidget(title)
        top.addStretch(1)
        root.addLayout(top)

        main = QHBoxLayout()
        main.setSpacing(10)

        # Left panel -----------------------------------------------------
        # Keep only data selection and processing scope. All time-lapse viewing
        # happens in the single central Working View.
        left = QFrame()
        left.setObjectName("SidePanel")
        left.setFixedWidth(255)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(7)

        ll.addWidget(self._section("INPUT / TIME-LAPSE"))
        self.open_files_btn = QPushButton("Open TIFF File(s)")
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_files_btn.clicked.connect(self.open_files)
        self.open_folder_btn.clicked.connect(self.open_folder)
        ll.addWidget(self.open_files_btn)
        ll.addWidget(self.open_folder_btn)

        drop_hint = QLabel("Drag TIFF file(s) or a folder into the list below")
        drop_hint.setWordWrap(True)
        drop_hint.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        ll.addWidget(drop_hint)

        self.dataset_list = DropPathListWidget()
        self.dataset_list.currentRowChanged.connect(self._dataset_changed)
        self.dataset_list.pathsDropped.connect(self._paths_dropped)
        ll.addWidget(self.dataset_list, 1)

        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Current dataset", "All loaded datasets"])
        ll.addWidget(QLabel("Processing scope"))
        ll.addWidget(self.scope_combo)

        main.addWidget(left)

        # Center ---------------------------------------------------------
        center = QVBoxLayout()
        view_buttons = QHBoxLayout()
        self.original_view_btn = QPushButton("Image + Instance")
        self.mask_view_btn = QPushButton("Mask Review")
        self.dynamic_view_btn = QPushButton("Dynamic Overlay")
        self.overlay_toggle_btn = QPushButton("Overlay ON")
        self.overlay_toggle_btn.setCheckable(True)
        self.overlay_toggle_btn.setChecked(True)
        self.overlay_toggle_btn.setMaximumWidth(105)
        self.original_view_btn.clicked.connect(lambda: self.set_view_mode("preview"))
        self.mask_view_btn.clicked.connect(lambda: self.set_view_mode("mask"))
        self.dynamic_view_btn.clicked.connect(lambda: self.set_view_mode("dynamic"))
        self.overlay_toggle_btn.clicked.connect(self._toggle_overlay_visibility)
        view_buttons.addWidget(self.original_view_btn)
        view_buttons.addWidget(self.mask_view_btn)
        view_buttons.addWidget(self.dynamic_view_btn)
        view_buttons.addWidget(self.overlay_toggle_btn)
        center.addLayout(view_buttons)

        self.canvas = DynamicsCanvas()
        self.canvas.roiCreated.connect(self._roi_created)
        self.canvas.scene().selectionChanged.connect(self._scene_roi_selection_changed)
        center.addWidget(self.canvas, 1)

        self.frame_label = QLabel("Frame: - / -")
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._frame_changed)
        center.addWidget(self.frame_label)
        center.addWidget(self.frame_slider)

        main.addLayout(center, 1)

        # Right panel ----------------------------------------------------
        right = QFrame()
        right.setObjectName("SidePanel")
        right.setFixedWidth(290)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(7, 7, 7, 7)
        rl.setSpacing(4)

        rl.addWidget(self._section("MASK GENERATION"))
        params1 = QHBoxLayout()
        self.block_size = QSpinBox(); self.block_size.setRange(3, 199); self.block_size.setSingleStep(2); self.block_size.setValue(37)
        self.adaptive_c = QDoubleSpinBox(); self.adaptive_c.setRange(-100, 100); self.adaptive_c.setDecimals(2); self.adaptive_c.setValue(12)
        self.mask_threshold = QSpinBox(); self.mask_threshold.setRange(0, 255); self.mask_threshold.setValue(50)
        params1.addWidget(QLabel("Block")); params1.addWidget(self.block_size)
        params1.addWidget(QLabel("C")); params1.addWidget(self.adaptive_c)
        rl.addLayout(params1)
        p2 = QHBoxLayout(); p2.addWidget(QLabel("Threshold")); p2.addWidget(self.mask_threshold); rl.addLayout(p2)
        self.generate_mask_btn = QPushButton("Generate Mask"); self.generate_mask_btn.setMaximumWidth(135)
        self.generate_mask_btn.clicked.connect(self.generate_masks)
        rl.addWidget(self.generate_mask_btn)

        # Mask overlay opacity is fixed for consistent QC display.

        rl.addWidget(self._section("DYNAMIC ANALYSIS"))
        grid1 = QHBoxLayout()
        self.dt_seconds = QDoubleSpinBox(); self.dt_seconds.setRange(0.001, 9999); self.dt_seconds.setDecimals(3); self.dt_seconds.setValue(2.0)
        self.xy_um = QDoubleSpinBox(); self.xy_um.setRange(0.0001, 100); self.xy_um.setDecimals(6); self.xy_um.setValue(0.057292); self.xy_um.valueChanged.connect(self._update_roi_info)
        grid1.addWidget(QLabel("Δt s")); grid1.addWidget(self.dt_seconds)
        grid1.addWidget(QLabel("XY µm")); grid1.addWidget(self.xy_um)
        rl.addLayout(grid1)
        grid2 = QHBoxLayout()
        self.z_um = QDoubleSpinBox(); self.z_um.setRange(0.0001, 100); self.z_um.setDecimals(6); self.z_um.setValue(0.20)
        self.track_radius = QDoubleSpinBox(); self.track_radius.setRange(0.01, 100); self.track_radius.setDecimals(3); self.track_radius.setValue(1.0)
        grid2.addWidget(QLabel("Z µm")); grid2.addWidget(self.z_um)
        grid2.addWidget(QLabel("Track R µm")); grid2.addWidget(self.track_radius)
        rl.addLayout(grid2)
        grid3 = QHBoxLayout()
        self.network_dist = QDoubleSpinBox(); self.network_dist.setRange(0.001, 100); self.network_dist.setDecimals(3); self.network_dist.setValue(0.20)
        grid3.addWidget(QLabel("Network dist µm")); grid3.addWidget(self.network_dist)
        rl.addLayout(grid3)
        self.run_dynamic_btn = QPushButton("Run Dynamic Analysis"); self.run_dynamic_btn.setMaximumWidth(155)
        self.run_dynamic_btn.clicked.connect(self.run_dynamic)
        rl.addWidget(self.run_dynamic_btn)

        rl.addWidget(self._section("DISPLAY / EXPORT"))
        self.background_combo = QComboBox()
        self.background_combo.addItems(["Original image", "Black", "White", "Gray"])
        self.background_combo.setMaximumWidth(185)
        # activated fires for an explicit user choice, including re-selecting the
        # current item; that is useful for switching away from an external TIFF.
        self.background_combo.activated.connect(self._background_combo_selected)
        self.background_tif_btn = QPushButton("Choose Background Image")
        self.background_tif_btn.setMaximumWidth(185)
        self.background_tif_btn.setToolTip("Load a second-channel time-lapse (for example ER) as the synchronized background")
        self.background_tif_btn.clicked.connect(self.open_background_tif)
        rl.addWidget(self.background_combo)
        rl.addWidget(self.background_tif_btn)
        self.show_tracks = QCheckBox("Show trajectory tails"); self.show_tracks.setChecked(True); self.show_tracks.stateChanged.connect(self.refresh_view)
        rl.addWidget(self.show_tracks)
        trail = QHBoxLayout(); self.trail_length = QSpinBox(); self.trail_length.setRange(0, 9999); self.trail_length.setValue(10); self.trail_length.valueChanged.connect(self.refresh_view); trail.addWidget(QLabel("Trail frames (0=all)")); trail.addWidget(self.trail_length); rl.addLayout(trail)
        self.export_btn = QPushButton("Export Overlay MP4"); self.export_btn.setMaximumWidth(155)
        self.export_btn.clicked.connect(self.export_video)
        rl.addWidget(self.export_btn)

        rl.addWidget(self._section("ROI DYNAMICS"))
        roi_draw_row = QHBoxLayout()
        self.roi_polygon_btn = QPushButton("Free ROI")
        self.roi_rect_btn = QPushButton("Rectangle")
        self.roi_circle_btn = QPushButton("Circle")
        for b in (self.roi_polygon_btn, self.roi_rect_btn, self.roi_circle_btn):
            b.setCheckable(True)
        self.roi_polygon_btn.clicked.connect(lambda checked: self._set_roi_draw_mode("polygon" if checked else None))
        self.roi_rect_btn.clicked.connect(lambda checked: self._set_roi_draw_mode("rectangle" if checked else None))
        self.roi_circle_btn.clicked.connect(lambda checked: self._set_roi_draw_mode("circle" if checked else None))
        roi_draw_row.addWidget(self.roi_polygon_btn)
        roi_draw_row.addWidget(self.roi_rect_btn)
        roi_draw_row.addWidget(self.roi_circle_btn)
        rl.addLayout(roi_draw_row)

        self.roi_list = QListWidget()
        self.roi_list.setMaximumHeight(95)
        self.roi_list.currentRowChanged.connect(self._roi_list_selection_changed)
        self.roi_list.itemDoubleClicked.connect(lambda _item: self.rename_selected_roi())
        rl.addWidget(self.roi_list)

        roi_manage_row = QHBoxLayout()
        self.roi_rename_btn = QPushButton("Rename")
        self.roi_delete_btn = QPushButton("Delete")
        self.roi_clear_btn = QPushButton("Clear")
        self.roi_rename_btn.clicked.connect(self.rename_selected_roi)
        self.roi_delete_btn.clicked.connect(self.delete_selected_roi)
        self.roi_clear_btn.clicked.connect(self.clear_rois)
        roi_manage_row.addWidget(self.roi_rename_btn)
        roi_manage_row.addWidget(self.roi_delete_btn)
        roi_manage_row.addWidget(self.roi_clear_btn)
        rl.addLayout(roi_manage_row)

        self.roi_info = QLabel("No ROI selected")
        self.roi_info.setWordWrap(True)
        self.roi_info.setStyleSheet("color: #44576A; font-size: 11px;")
        rl.addWidget(self.roi_info)

        self.export_roi_csv_btn = QPushButton("Export ROI Dynamics CSV"); self.export_roi_csv_btn.setMaximumWidth(185)
        self.export_roi_csv_btn.clicked.connect(self.export_roi_dynamics_csv)
        rl.addWidget(self.export_roi_csv_btn)

        rl.addStretch(1)

        main.addWidget(right)
        root.addLayout(main, 1)

    def _configure_module5_controls(self):
        """Direct numeric entry, no spin arrows, and consistent button feedback."""
        for box in self.findChildren(QAbstractSpinBox):
            box.setButtonSymbols(QAbstractSpinBox.NoButtons)

        for button in self.findChildren(QPushButton):
            button.setProperty("flash", False)
            button.setProperty("busy", False)
            button.pressed.connect(lambda b=button: self._flash_button(b))

    @staticmethod
    def _repolish(widget):
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    def _flash_button(self, button: QPushButton):
        if bool(button.property("busy")):
            return
        button.setProperty("flash", True)
        self._repolish(button)

        def reset():
            if not bool(button.property("busy")):
                button.setProperty("flash", False)
                self._repolish(button)

        QTimer.singleShot(220, reset)

    def _set_button_busy(self, button: Optional[QPushButton], busy: bool, running_text: str = "Running..."):
        if button is None:
            return
        if busy:
            if button.property("baseText") is None:
                button.setProperty("baseText", button.text())
            button.setProperty("flash", False)
            button.setProperty("busy", True)
            button.setText(running_text)
            button.setEnabled(False)
        else:
            base = button.property("baseText")
            if base is not None:
                button.setText(str(base))
            button.setProperty("busy", False)
            button.setProperty("flash", False)
            button.setEnabled(True)
        self._repolish(button)

    def _section(self, text):
        lab = QLabel(text); lab.setObjectName("Section"); return lab

    def _apply_style(self):
        self.setStyleSheet(f"""
            QWidget {{ font-family: 'Segoe UI'; font-size: 12px; color: {TEXT}; }}
            QFrame#SidePanel {{ background: #F8FAFD; border: 1px solid {BORDER}; border-radius: 8px; }}
            QLabel#DynTitle {{ font-size: 19px; font-weight: 800; color: {NAVY}; }}
            QLabel#Section {{ color: {DEEP_BLUE}; font-weight: 900; letter-spacing: 0.8px; padding-top: 5px; }}
            QPushButton {{ background: white; border: 1px solid {BORDER}; border-radius: 5px; padding: 5px 8px; min-height: 18px; }}
            QPushButton:hover {{ border: 1px solid {US_BLUE}; background: #F2F8FD; }}
            QPushButton:checked {{ background: {US_BLUE}; color: white; border-color: {US_BLUE}; }}
            QPushButton[flash="true"] {{ background: {CYAN}; color: white; border-color: {CYAN}; }}
            QPushButton[busy="true"] {{ background: {GREEN}; color: white; border-color: {GREEN}; font-weight: 600; }}
            QPushButton[busy="true"]:disabled {{ background: {GREEN}; color: white; border-color: {GREEN}; }}
            QListWidget, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: white; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 5px; }}
            QComboBox QAbstractItemView {{ background: white; color: {TEXT}; selection-background-color: #E8F3FB; selection-color: {TEXT}; border: 1px solid {BORDER}; }}
        """)


    # ------------------------------------------------------------------ ROI drawing / editing
    def _set_roi_draw_mode(self, mode: Optional[str]):
        buttons = {
            "polygon": self.roi_polygon_btn,
            "rectangle": self.roi_rect_btn,
            "circle": self.roi_circle_btn,
        }
        for key, btn in buttons.items():
            btn.blockSignals(True)
            btn.setChecked(mode == key)
            btn.blockSignals(False)
        self.canvas.set_mask_editing(False)
        self.canvas.set_roi_draw_mode(mode)
        if mode == "polygon":
            self.log("Free ROI: click vertices; double-click to finish. Drag vertices later to edit.")
        elif mode == "rectangle":
            self.log("Rectangle ROI: drag to draw; drag a corner later to resize; drag inside to move.")
        elif mode == "circle":
            self.log("Circle ROI: drag from center to radius; drag the edge later to resize; drag inside to move.")

    def _next_roi_name(self) -> str:
        existing = {r.name for r in self.rois}
        while True:
            name = f"ROI_{self._roi_counter:02d}"
            self._roi_counter += 1
            if name not in existing:
                return name

    def _roi_created(self, shape_type: str, geometry):
        name = self._next_roi_name()
        if shape_type == "rectangle":
            item = EditableRectROI(geometry, name, self._roi_geometry_changed)
        elif shape_type == "circle":
            center, radius = geometry
            item = EditableCircleROI(center, radius, name, self._roi_geometry_changed)
        elif shape_type == "polygon":
            item = EditablePolygonROI(geometry, name, self._roi_geometry_changed)
        else:
            return
        self.canvas.add_roi_item(item)
        record = ROIRecord(name=name, shape_type=shape_type, item=item)
        self.rois.append(record)
        self.roi_list.addItem(name)
        self.roi_list.setCurrentRow(len(self.rois) - 1)
        item.setSelected(True)
        self._set_roi_draw_mode(None)
        self._save_rois_for_current_dataset()
        self._update_roi_info()

    def _roi_geometry_changed(self):
        if self._loading_rois:
            return
        self._save_rois_for_current_dataset()
        self._update_roi_info()

    def _roi_list_selection_changed(self, row: int):
        if self._loading_rois:
            return
        for i, rec in enumerate(self.rois):
            rec.item.setSelected(i == row)
        self._update_roi_info()

    def _scene_roi_selection_changed(self):
        if self._loading_rois:
            return
        selected = self.canvas.scene().selectedItems()
        row = -1
        for i, rec in enumerate(self.rois):
            if rec.item in selected:
                row = i
                break
        if self.roi_list.currentRow() != row:
            self.roi_list.blockSignals(True)
            self.roi_list.setCurrentRow(row)
            self.roi_list.blockSignals(False)
        self._update_roi_info()

    def _selected_roi(self) -> Optional[ROIRecord]:
        row = self.roi_list.currentRow()
        if 0 <= row < len(self.rois):
            return self.rois[row]
        return None

    def rename_selected_roi(self):
        rec = self._selected_roi()
        if rec is None:
            return
        value, ok = QInputDialog.getText(self, "Rename ROI", "ROI name:", text=rec.name)
        if not ok:
            return
        name = value.strip()
        if not name:
            return
        if any(r is not rec and r.name == name for r in self.rois):
            QMessageBox.warning(self, "Duplicate name", f"An ROI named '{name}' already exists.")
            return
        rec.name = name
        if hasattr(rec.item, "set_roi_name"):
            rec.item.set_roi_name(name)
        self.roi_list.currentItem().setText(name)
        self._save_rois_for_current_dataset()
        self._update_roi_info()

    def delete_selected_roi(self):
        row = self.roi_list.currentRow()
        if not (0 <= row < len(self.rois)):
            return
        rec = self.rois.pop(row)
        self.canvas.remove_roi_item(rec.item)
        self.roi_list.takeItem(row)
        if self.rois:
            self.roi_list.setCurrentRow(min(row, len(self.rois) - 1))
        self._save_rois_for_current_dataset()
        self._update_roi_info()

    def clear_rois(self):
        if not self.rois:
            return
        answer = QMessageBox.question(self, "Clear ROIs", "Delete all ROIs for this dataset?")
        if answer != QMessageBox.Yes:
            return
        self.canvas.clear_roi_items()
        self.rois.clear()
        self.roi_list.clear()
        self._roi_counter = 1
        self._save_rois_for_current_dataset()
        self._update_roi_info()

    def _roi_geometry(self, rec: ROIRecord) -> Dict[str, object]:
        if rec.shape_type == "rectangle":
            r = rec.item.scene_rect()
            return {
                "x": float(r.x()), "y": float(r.y()),
                "width": float(r.width()), "height": float(r.height()),
            }
        if rec.shape_type == "circle":
            c = rec.item.scene_center()
            return {"cx": float(c.x()), "cy": float(c.y()), "radius": float(rec.item.radius())}
        points = rec.item.scene_points()
        return {"points": [[float(p.x()), float(p.y())] for p in points]}

    def _serialize_roi(self, rec: ROIRecord) -> Dict[str, object]:
        return {"name": rec.name, "shape_type": rec.shape_type, "geometry": self._roi_geometry(rec)}

    def _roi_json_path(self, ds: Optional[DynamicsDataset] = None) -> Optional[Path]:
        ds = ds or self.current_dataset()
        if ds is None or ds.processed_dir is None:
            return None
        return ds.processed_dir / "roi_definitions.json"

    def _save_rois_for_current_dataset(self):
        if self._loading_rois:
            return
        path = self._roi_json_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "coordinate_system": "image_pixels_xy",
            "rois": [self._serialize_roi(r) for r in self.rois],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_rois_for_current_dataset(self):
        self._loading_rois = True
        try:
            self.canvas.clear_roi_items()
            self.rois.clear()
            self.roi_list.clear()
            self._roi_counter = 1
            path = self._roi_json_path()
            if path is None or not path.exists():
                return
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return
            for entry in payload.get("rois", []):
                name = str(entry.get("name") or self._next_roi_name())
                shape_type = str(entry.get("shape_type", ""))
                g = entry.get("geometry", {})
                if shape_type == "rectangle":
                    item = EditableRectROI(
                        QRectF(float(g["x"]), float(g["y"]), float(g["width"]), float(g["height"])),
                        name, self._roi_geometry_changed,
                    )
                elif shape_type == "circle":
                    item = EditableCircleROI(
                        QPointF(float(g["cx"]), float(g["cy"])), float(g["radius"]),
                        name, self._roi_geometry_changed,
                    )
                elif shape_type == "polygon":
                    points = [QPointF(float(x), float(y)) for x, y in g.get("points", [])]
                    if len(points) < 3:
                        continue
                    item = EditablePolygonROI(points, name, self._roi_geometry_changed)
                else:
                    continue
                self.canvas.add_roi_item(item)
                self.rois.append(ROIRecord(name=name, shape_type=shape_type, item=item))
                self.roi_list.addItem(name)
            numeric = []
            for rec in self.rois:
                if rec.name.startswith("ROI_"):
                    try:
                        numeric.append(int(rec.name.split("_")[-1]))
                    except Exception:
                        pass
            self._roi_counter = max(numeric, default=0) + 1
            if self.rois:
                self.roi_list.setCurrentRow(0)
                self.rois[0].item.setSelected(True)
        finally:
            self._loading_rois = False
            self._update_roi_info()

    @staticmethod
    def _polygon_area_perimeter(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
        if len(points) < 3:
            return 0.0, 0.0
        arr = np.asarray(points, dtype=float)
        x, y = arr[:, 0], arr[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        diff = arr - np.roll(arr, -1, axis=0)
        perimeter = float(np.sum(np.linalg.norm(diff, axis=1)))
        return area, perimeter

    def _roi_shape_metrics_from_definition(self, definition: Dict[str, object], xy_um: float) -> Dict[str, object]:
        shape = definition["shape_type"]
        g = definition["geometry"]
        row = {
            "roi_name": definition["name"],
            "roi_shape": shape,
            "roi_center_x_px": np.nan,
            "roi_center_y_px": np.nan,
            "roi_area_px2": np.nan,
            "roi_area_um2": np.nan,
            "roi_perimeter_px": np.nan,
            "roi_perimeter_um": np.nan,
            "roi_width_px": np.nan,
            "roi_height_px": np.nan,
            "roi_width_um": np.nan,
            "roi_height_um": np.nan,
            "roi_radius_px": np.nan,
            "roi_radius_um": np.nan,
            "roi_diameter_px": np.nan,
            "roi_diameter_um": np.nan,
            "roi_vertex_count": np.nan,
        }
        if shape == "rectangle":
            w, h = float(g["width"]), float(g["height"])
            x, y = float(g["x"]), float(g["y"])
            area = w * h
            per = 2.0 * (w + h)
            row.update({
                "roi_center_x_px": x + w / 2.0,
                "roi_center_y_px": y + h / 2.0,
                "roi_area_px2": area, "roi_area_um2": area * xy_um * xy_um,
                "roi_perimeter_px": per, "roi_perimeter_um": per * xy_um,
                "roi_width_px": w, "roi_height_px": h,
                "roi_width_um": w * xy_um, "roi_height_um": h * xy_um,
                "roi_vertex_count": 4,
            })
        elif shape == "circle":
            r = float(g["radius"])
            area = math.pi * r * r
            per = 2.0 * math.pi * r
            row.update({
                "roi_center_x_px": float(g["cx"]), "roi_center_y_px": float(g["cy"]),
                "roi_area_px2": area, "roi_area_um2": area * xy_um * xy_um,
                "roi_perimeter_px": per, "roi_perimeter_um": per * xy_um,
                "roi_width_px": 2 * r, "roi_height_px": 2 * r,
                "roi_width_um": 2 * r * xy_um, "roi_height_um": 2 * r * xy_um,
                "roi_radius_px": r, "roi_radius_um": r * xy_um,
                "roi_diameter_px": 2 * r, "roi_diameter_um": 2 * r * xy_um,
            })
        elif shape == "polygon":
            pts = [(float(x), float(y)) for x, y in g.get("points", [])]
            area, per = self._polygon_area_perimeter(pts)
            arr = np.asarray(pts, dtype=float)
            if len(arr):
                xmin, ymin = np.min(arr, axis=0)
                xmax, ymax = np.max(arr, axis=0)
                w, h = xmax - xmin, ymax - ymin
                # True polygon area centroid (shoelace formula).  The old
                # vertex-average is only correct for some symmetric polygons.
                if len(arr) >= 3:
                    x = arr[:, 0]
                    y = arr[:, 1]
                    x_next = np.roll(x, -1)
                    y_next = np.roll(y, -1)
                    cross = x * y_next - x_next * y
                    signed_area2 = float(np.sum(cross))
                    if abs(signed_area2) > 1e-12:
                        center = np.asarray([
                            np.sum((x + x_next) * cross) / (3.0 * signed_area2),
                            np.sum((y + y_next) * cross) / (3.0 * signed_area2),
                        ], dtype=float)
                    else:
                        center = np.mean(arr, axis=0)
                else:
                    center = np.mean(arr, axis=0)
            else:
                center = (np.nan, np.nan); w = h = np.nan
            row.update({
                "roi_center_x_px": float(center[0]), "roi_center_y_px": float(center[1]),
                "roi_area_px2": area, "roi_area_um2": area * xy_um * xy_um,
                "roi_perimeter_px": per, "roi_perimeter_um": per * xy_um,
                "roi_width_px": w, "roi_height_px": h,
                "roi_width_um": w * xy_um, "roi_height_um": h * xy_um,
                "roi_vertex_count": len(pts),
            })
        return row

    def _update_roi_info(self, *_args):
        if not hasattr(self, "roi_info"):
            return
        rec = self._selected_roi()
        if rec is None:
            self.roi_info.setText("No ROI selected")
            return
        definition = self._serialize_roi(rec)
        m = self._roi_shape_metrics_from_definition(definition, float(self.xy_um.value()) if hasattr(self, "xy_um") else 1.0)
        if rec.shape_type == "circle":
            detail = f"radius={m['roi_radius_px']:.1f} px ({m['roi_radius_um']:.3f} µm)"
        elif rec.shape_type == "rectangle":
            detail = f"W×H={m['roi_width_px']:.1f}×{m['roi_height_px']:.1f} px"
        else:
            detail = f"vertices={int(m['roi_vertex_count'])}"
        self.roi_info.setText(
            f"{rec.name} · {rec.shape_type}\n"
            f"area={m['roi_area_px2']:.1f} px² ({m['roi_area_um2']:.3f} µm²) · "
            f"perimeter={m['roi_perimeter_px']:.1f} px · {detail}"
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
            if any(p.is_dir() or (p.is_file() and p.suffix.lower() in SUPPORTED_TIFFS) for p in paths):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        valid = [p for p in paths if p.is_dir() or (p.is_file() and p.suffix.lower() in SUPPORTED_TIFFS)]
        if valid:
            self._paths_dropped(valid)
            event.acceptProposedAction()
        else:
            event.ignore()

    # ------------------------------------------------------------------ Loading/splitting
    def log(self, text: str):
        # Module 5 no longer keeps a permanent Status panel.
        # Non-dynamic messages remain available in the console.
        print(str(text))

    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open time-lapse TIFF file(s)", "", "TIFF (*.tif *.tiff)")
        if not files:
            return
        for f in files:
            self._add_dataset(Path(f))

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open folder containing TIFF files")
        if not folder:
            return
        self._add_folder(Path(folder), show_warning=True)

    def _paths_dropped(self, paths):
        """Handle file/folder drag-and-drop into the left list."""
        added_before = self.dataset_list.count()
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                self._add_folder(path, show_warning=False)
            elif path.is_file() and path.suffix.lower() in SUPPORTED_TIFFS:
                self._add_dataset(path)
        if self.dataset_list.count() > added_before:
            # Select the first newly added item so it appears immediately.
            self.dataset_list.setCurrentRow(added_before)

    def _add_folder(self, folder: Path, show_warning: bool = False):
        folder = Path(folder)
        paths = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_TIFFS],
            key=lambda p: natural_key(p.name),
        )
        if not paths:
            if show_warning:
                QMessageBox.warning(self, "No TIFF", "No TIFF files were found directly inside this folder.")
            else:
                self.log(f"[DROP] No TIFF files directly inside: {folder}")
            return

        # If this is already a split time-series folder (name_01.tif, name_02.tif...),
        # load the folder as ONE dataset instead of treating every frame separately.
        import re
        stripped = [re.sub(r"_\d+$", "", p.stem) for p in paths]
        if len(paths) >= 2 and len(set(stripped)) == 1:
            ds = DynamicsDataset(source_path=folder, name=folder.name, frame_dir=folder, frames=paths)
            if not any(d.source_path.resolve() == folder.resolve() for d in self.datasets):
                self.datasets.append(ds)
                item = QListWidgetItem(folder.name)
                item.setToolTip(str(folder))
                item.setData(Qt.UserRole, str(folder))
                self.dataset_list.addItem(item)
                if self.dataset_list.count() == 1:
                    self.dataset_list.setCurrentRow(0)
            return

        # Otherwise treat every TIFF directly in the dropped/opened folder as a source dataset.
        for path in paths:
            self._add_dataset(path)

    def _add_dataset(self, path: Path):
        path = Path(path)
        if any(ds.source_path.resolve() == path.resolve() for ds in self.datasets):
            return
        ds = DynamicsDataset(source_path=path, name=path.stem)
        split_candidate = path.parent / path.stem
        if split_candidate.exists() and split_candidate.is_dir():
            frames = sorted([p for p in split_candidate.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_TIFFS], key=lambda p: natural_key(p.name))
            if frames:
                ds.frame_dir = split_candidate
                ds.frames = frames
        self.datasets.append(ds)
        item = QListWidgetItem(path.name if path.is_file() else ds.name)
        item.setToolTip(str(path))
        item.setData(Qt.UserRole, str(path))
        self.dataset_list.addItem(item)
        if self.dataset_list.count() == 1:
            self.dataset_list.setCurrentRow(0)

    def _scope_datasets(self) -> List[DynamicsDataset]:
        if self.scope_combo.currentIndex() == 1:
            return list(self.datasets)
        ds = self.current_dataset()
        return [] if ds is None else [ds]

    def current_dataset(self) -> Optional[DynamicsDataset]:
        if 0 <= self.current_dataset_index < len(self.datasets):
            return self.datasets[self.current_dataset_index]
        return None

    def _dataset_changed(self, row: int):
        self._save_current_mask_if_dirty()
        self.current_dataset_index = row
        self.current_frame_index = 0
        ds = self.current_dataset()

        # One main slider handles the raw source before Split and all stacked
        # results after Split. There is no separate Source Preview anymore.
        if ds is None:
            n_main = 0
        elif ds.frames:
            n_main = len(ds.frames)
        else:
            n_main = self._source_frame_count(ds)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, n_main - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(n_main > 0)
        self.frame_slider.blockSignals(False)
        self._update_frame_label()

        if ds is not None and self._external_background_path(ds) is not None:
            self.background_tif_btn.setText("Background Image ✓")
            self.background_tif_btn.setToolTip(str(self._external_background_path(ds)))
        else:
            self.background_tif_btn.setText("Choose Background Image")
            self.background_tif_btn.setToolTip("Load a second-channel time-lapse (for example ER) as the synchronized background")

        self._load_mask_for_current()
        self._load_rois_for_current_dataset()
        self.refresh_view()

    def _frame_changed(self, value: int):
        # Keep slider movement itself cheap. The label updates immediately, while
        # disk/cache work is debounced so a fast drag does not render every stop.
        self._save_current_mask_if_dirty()
        self.current_frame_index = int(value)
        self._update_frame_label()
        self._frame_refresh_timer.start()

    def _commit_frame_change(self):
        self._load_mask_for_current()
        self.refresh_view()

    def _update_frame_label(self):
        ds = self.current_dataset()
        if ds is None:
            n = 0
        elif ds.frames:
            n = len(ds.frames)
        else:
            n = self._source_frame_count(ds)
        self.frame_label.setText(f"Frame: {self.current_frame_index + 1 if n else '-'} / {n if n else '-'}")

    @staticmethod
    def _path_cache_key(path: Path) -> Tuple[str, int, int]:
        path = Path(path)
        try:
            stat = path.stat()
            return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        except Exception:
            return (str(path), 0, 0)

    def _lru_get(self, cache: OrderedDict, key):
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    def _lru_put(self, cache: OrderedDict, key, value):
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > self._cache_max_items:
            cache.popitem(last=False)
        return value

    def _cached_frame_2d(self, path: Path) -> np.ndarray:
        key = self._path_cache_key(path)
        cached = self._lru_get(self._frame_image_cache, key)
        if cached is not None:
            return cached
        return self._lru_put(self._frame_image_cache, key, read_frame_2d(path))

    def _cached_numeric_txt(self, path: Path) -> np.ndarray:
        key = self._path_cache_key(path)
        cached = self._lru_get(self._numeric_txt_cache, key)
        if cached is not None:
            return cached
        return self._lru_put(self._numeric_txt_cache, key, read_numeric_txt(path))

    def _cached_mask_2d(self, path: Path) -> np.ndarray:
        key = self._path_cache_key(path)
        cached = self._lru_get(self._mask_image_cache, key)
        if cached is not None:
            return cached.copy()
        mask = np.asarray(tifffile.imread(path))
        mask = np.squeeze(mask)
        while mask.ndim > 2:
            mask = np.max(mask, axis=0)
        mask = (mask != 0).astype(np.uint8) * 255
        self._lru_put(self._mask_image_cache, key, mask)
        return mask.copy()

    def _viewer_stack_path(self, ds: DynamicsDataset, kind: str) -> Optional[Path]:
        if ds.processed_dir is None:
            return None
        names = {
            "original": "viewer_original_stack.tif",
            "mask": "viewer_mask_stack.tif",
            "instance": "viewer_instance_rgb_stack.tif",
        }
        name = names.get(str(kind))
        return None if name is None else ds.processed_dir / name

    def _invalidate_viewer_stack(self, path: Optional[Path] = None):
        if path is None:
            self._viewer_stack_cache.clear()
            return
        try:
            key = str(Path(path).resolve())
        except Exception:
            key = str(path)
        self._viewer_stack_cache.pop(key, None)

    def _open_viewer_stack(self, path: Path):
        path = Path(path)
        if not path.exists():
            return None
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        cached = self._viewer_stack_cache.get(key)
        if cached is not None:
            return cached
        try:
            arr = tifffile.memmap(path, mode="r+")
        except Exception:
            # Fallback for externally generated/non-memmappable stacks.
            arr = np.asarray(tifffile.imread(path))
        self._viewer_stack_cache[key] = arr
        return arr

    @staticmethod
    def _new_tiff_memmap(path: Path, shape, dtype):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        nbytes = int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)
        return tifffile.memmap(
            path,
            shape=tuple(shape),
            dtype=np.dtype(dtype),
            bigtiff=(nbytes >= 3_800_000_000),
        )

    def _build_original_stack(self, ds: DynamicsDataset) -> Optional[Path]:
        if not ds.frames or ds.processed_dir is None:
            return None
        first = np.asarray(read_frame_2d(ds.frames[0]))
        if first.ndim != 2:
            raise ValueError(f"Viewer original stack expects 2D frames, got {first.shape}")
        stack_path = self._viewer_stack_path(ds, "original")
        self._invalidate_viewer_stack(stack_path)
        mm = self._new_tiff_memmap(stack_path, (len(ds.frames),) + first.shape, first.dtype)
        for i, frame in enumerate(ds.frames):
            arr = np.asarray(read_frame_2d(frame))
            if arr.shape != first.shape:
                raise ValueError(f"Frame size mismatch while stacking: {frame.name}: {arr.shape} != {first.shape}")
            mm[i] = arr.astype(first.dtype, copy=False)
        mm.flush()
        del mm
        self._invalidate_viewer_stack(stack_path)
        return stack_path

    def _build_mask_stack(self, ds: DynamicsDataset) -> Optional[Path]:
        if not ds.frames or ds.processed_dir is None:
            return None
        paths = [ds.processed_dir / f"{f.stem}_mask.tif" for f in ds.frames]
        if not all(p.exists() for p in paths):
            return None
        first = np.asarray(tifffile.imread(paths[0]))
        first = np.squeeze(first)
        while first.ndim > 2:
            first = np.max(first, axis=0)
        first = (first != 0).astype(np.uint8) * 255
        stack_path = self._viewer_stack_path(ds, "mask")
        self._invalidate_viewer_stack(stack_path)
        mm = self._new_tiff_memmap(stack_path, (len(paths),) + first.shape, np.uint8)
        for i, path in enumerate(paths):
            arr = np.asarray(tifffile.imread(path))
            arr = np.squeeze(arr)
            while arr.ndim > 2:
                arr = np.max(arr, axis=0)
            if arr.shape != first.shape:
                raise ValueError(f"Mask size mismatch while stacking: {path.name}: {arr.shape} != {first.shape}")
            mm[i] = (arr != 0).astype(np.uint8) * 255
        mm.flush()
        del mm
        self._invalidate_viewer_stack(stack_path)
        return stack_path

    def _build_instance_stack(self, ds: DynamicsDataset) -> Optional[Path]:
        if not ds.frames or ds.processed_dir is None:
            return None
        original = self._read_original_timepoint(ds, 0)
        if original is None:
            return None
        h, w = original.shape[:2]
        tracked = self._tracked_df(ds, force=True)
        stack_path = self._viewer_stack_path(ds, "instance")
        self._invalidate_viewer_stack(stack_path)
        mm = self._new_tiff_memmap(stack_path, (len(ds.frames), h, w, 3), np.uint8)
        mm[:] = 0

        for i, frame in enumerate(ds.frames):
            voxel_path = ds.processed_dir / f"{frame.stem}_voxel_instance.txt"
            if not voxel_path.exists():
                continue
            try:
                arr = read_numeric_txt(voxel_path)
            except Exception:
                continue
            if arr.ndim != 2 or arr.shape[1] < 6:
                continue
            xyz = np.asarray(arr[:, :3], dtype=float)
            local_rgb = np.clip(np.rint(arr[:, 3:6]), 0, 255).astype(np.uint8)
            x = np.rint(xyz[:, 0]).astype(np.int64)
            y = np.rint(xyz[:, 1]).astype(np.int64)
            valid = (
                np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1])
                & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            )
            x, y, colors = x[valid], y[valid], local_rgb[valid].copy()

            if tracked is not None and not tracked.empty:
                rows = tracked[tracked["frame_name"].astype(str) == frame.stem]
                if not rows.empty:
                    mapping = {}
                    for r in rows.itertuples(index=False):
                        vals = [getattr(r, "voxel_r", np.nan), getattr(r, "voxel_g", np.nan), getattr(r, "voxel_b", np.nan)]
                        if np.all(np.isfinite(vals)):
                            mapping[tuple(int(v) for v in vals)] = int(r.track_id)
                    for local_color, tid in mapping.items():
                        keep = np.all(colors == np.asarray(local_color, dtype=np.uint8)[None, :], axis=1)
                        if np.any(keep):
                            colors[keep] = np.asarray(deterministic_track_rgb(tid), dtype=np.uint8)
            mm[i, y, x] = colors

        mm.flush()
        del mm
        self._invalidate_viewer_stack(stack_path)
        return stack_path

    def _read_stack_frame(self, ds: DynamicsDataset, kind: str, index: int) -> Optional[np.ndarray]:
        path = self._viewer_stack_path(ds, kind)
        if path is None or not path.exists():
            return None
        stack = self._open_viewer_stack(path)
        if stack is None or getattr(stack, "ndim", 0) < 3 or len(stack) == 0:
            return None
        idx = min(max(0, int(index)), len(stack) - 1)
        return np.asarray(stack[idx])

    def _read_original_timepoint(self, ds: DynamicsDataset, index: int) -> Optional[np.ndarray]:
        if ds.frames:
            stacked = self._read_stack_frame(ds, "original", index)
            if stacked is not None:
                return stacked
            idx = min(max(0, int(index)), len(ds.frames) - 1)
            return self._cached_frame_2d(ds.frames[idx])
        return self._read_source_timepoint(ds, index)

    def _dataset_key(self, ds: DynamicsDataset) -> str:
        try:
            return str(ds.source_path.resolve())
        except Exception:
            return str(ds.source_path)

    def _source_frame_count(self, ds: Optional[DynamicsDataset]) -> int:
        if ds is None:
            return 0
        if ds.source_path.is_dir():
            return len(ds.frames)
        if not ds.source_path.is_file():
            return 0
        try:
            with tifffile.TiffFile(ds.source_path) as tif:
                if len(tif.pages) > 1:
                    return len(tif.pages)
                shape = tuple(tif.series[0].shape) if tif.series else ()
            if len(shape) <= 2:
                return 1
            # Common unsplit time-lapse: T,Y,X or T,Z,Y,X.
            return int(shape[0])
        except Exception:
            try:
                arr = np.asarray(tifffile.imread(ds.source_path))
                arr = np.squeeze(arr)
                return 1 if arr.ndim <= 2 else int(arr.shape[0])
            except Exception:
                return 0

    def _read_source_timepoint(self, ds: DynamicsDataset, index: int) -> Optional[np.ndarray]:
        idx = max(0, int(index))
        source_key = self._dataset_key(ds)
        cache_key = (source_key, idx)
        cached = self._lru_get(self._source_timepoint_cache, cache_key)
        if cached is not None:
            return cached

        if ds.source_path.is_dir():
            if not ds.frames:
                return None
            idx = min(idx, len(ds.frames) - 1)
            arr = self._cached_frame_2d(ds.frames[idx])
            return self._lru_put(self._source_timepoint_cache, cache_key, arr)

        path = ds.source_path
        if not path.is_file():
            return None
        try:
            with tifffile.TiffFile(path) as tif:
                if len(tif.pages) > 1 and idx < len(tif.pages):
                    arr = np.asarray(tif.pages[idx].asarray())
                else:
                    arr = np.asarray(tif.asarray())
                    arr = np.squeeze(arr)
                    if arr.ndim > 2:
                        idx = min(idx, arr.shape[0] - 1)
                        arr = np.asarray(arr[idx])
            arr = np.squeeze(arr)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                arr = np.mean(arr[..., :3].astype(np.float32), axis=-1)
            while arr.ndim > 2:
                arr = np.max(arr, axis=0)
            return self._lru_put(self._source_timepoint_cache, cache_key, np.asarray(arr))
        except Exception:
            return None

    def _split_many(self, targets: Sequence[DynamicsDataset]):
        messages = []
        for ds in targets:
            source = ds.source_path
            if source.is_dir() and ds.frames:
                stack_path = self._viewer_stack_path(ds, "original")
                if stack_path is None or not stack_path.exists():
                    stack_path = self._build_original_stack(ds)
                messages.append(f"{ds.name}: already split ({len(ds.frames)} frame(s)) -> {source}")
                if stack_path is not None:
                    messages.append(f"{ds.name}: viewer original stack -> {stack_path}")
                continue
            arr = np.asarray(tifffile.imread(source))
            arr = np.squeeze(arr)

            if arr.ndim == 2:
                frames = [arr]
            elif arr.ndim == 3:
                if arr.shape[-1] in (3, 4):
                    frames = [np.mean(arr[..., :3].astype(np.float32), axis=-1)]
                else:
                    frames = [arr[i] for i in range(arr.shape[0])]
            elif arr.ndim == 4:
                # Typical T,Z,Y,X -> one 2D frame per timepoint by Z-max.
                if arr.shape[-1] in (3, 4):
                    frames = [np.mean(arr[t, ..., :3].astype(np.float32), axis=-1) for t in range(arr.shape[0])]
                else:
                    frames = [np.max(arr[t], axis=0) for t in range(arr.shape[0])]
            else:
                raise ValueError(f"Unsupported time-lapse shape {arr.shape} for {source}")

            out_dir = source.parent / source.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            width = max(2, len(str(len(frames))))
            for old in out_dir.glob(f"{source.stem}_*.tif"):
                try: old.unlink()
                except Exception: pass
            saved = []
            for i, frame in enumerate(frames, start=1):
                out = out_dir / f"{source.stem}_{i:0{width}d}.tif"
                tifffile.imwrite(out, np.asarray(frame))
                saved.append(out)
            ds.frame_dir = out_dir
            ds.frames = saved
            (out_dir / "processed_output").mkdir(parents=True, exist_ok=True)
            stack_path = self._build_original_stack(ds)
            messages.append(f"{ds.name}: split into {len(saved)} frame(s) -> {out_dir}")
            if stack_path is not None:
                messages.append(f"{ds.name}: viewer original stack -> {stack_path}")
        return messages

    def _after_split(self, messages):
        for m in messages:
            self.log(m)
        self._dataset_changed(self.current_dataset_index)

    # ------------------------------------------------------------------ Mask generation/review
    def generate_masks(self):
        """Auto-split unsplit time-lapses, then generate masks in one action."""
        targets = self._scope_datasets()
        if not targets:
            return
        block = int(self.block_size.value())
        if block % 2 == 0:
            block += 1
        c_val = float(self.adaptive_c.value())
        threshold = int(self.mask_threshold.value())
        self._run_threaded(
            lambda: self._prepare_and_generate_masks_many(targets, block, c_val, threshold),
            self._after_mask_generation,
            button=self.generate_mask_btn,
            running_text="Generating Masks...",
        )

    def _prepare_and_generate_masks_many(self, targets, block, c_val, threshold):
        """Split only datasets that need it, then run mask generation."""
        messages = []
        needs_split = [ds for ds in targets if not ds.frames]
        if needs_split:
            messages.extend(self._split_many(needs_split))
        messages.extend(self._generate_masks_many(targets, block, c_val, threshold))
        return messages

    def _generate_masks_many(self, targets, block, c_val, threshold):
        out_messages = []
        for ds in targets:
            proc = ds.frame_dir / "processed_output"
            proc.mkdir(parents=True, exist_ok=True)
            for frame in ds.frames:
                image = mask_backend.read_tif(frame)
                enhanced = mask_backend.enhance_adaptive(image, block_size=block, C=c_val)
                mask = mask_backend.convert_enhanced_image_to_binary_mask(enhanced, threshold=threshold)
                save_path = proc / f"{frame.stem}_mask.tif"
                # Save a 2D mask for a 2D time frame; backend accepts it and adds Z=1 later.
                mask_out = np.squeeze(mask)
                tifffile.imwrite(save_path, mask_out.astype(np.uint8))
            stack_path = self._build_mask_stack(ds)
            out_messages.append(f"{ds.name}: generated {len(ds.frames)} mask(s)")
            if stack_path is not None:
                out_messages.append(f"{ds.name}: viewer mask stack -> {stack_path}")
        return out_messages

    def _after_mask_generation(self, messages):
        for m in messages:
            self.log(m)
        self._mask_image_cache.clear()
        self._numeric_txt_cache.clear()
        # Auto-splitting may have changed frame_dir/frames. Rebind the single
        # main slider to the prepared stack before entering Mask Review.
        if self.current_dataset_index >= 0:
            self._dataset_changed(self.current_dataset_index)
        self.set_view_mode("mask")
        self._load_mask_for_current()
        self.refresh_view()

    def _toggle_overlay_visibility(self, checked: bool):
        """Temporarily hide/show analysis overlays without changing the active view mode."""
        self.overlay_visible = bool(checked)
        self.overlay_toggle_btn.setText("Overlay ON" if self.overlay_visible else "Overlay OFF")
        self.canvas.set_mask_editing(False)
        self.refresh_view()

    def set_view_mode(self, mode: str):
        self._save_current_mask_if_dirty()
        self.view_mode = mode
        ds = self.current_dataset()
        if mode == "mask":
            if ds is not None and ds.frames and ds.processed_dir is not None:
                stack_path = self._viewer_stack_path(ds, "mask")
                mask_paths = [ds.processed_dir / f"{f.stem}_mask.tif" for f in ds.frames]
                if (stack_path is None or not stack_path.exists()) and mask_paths and all(p.exists() for p in mask_paths):
                    try:
                        self._build_mask_stack(ds)
                    except Exception:
                        pass
            self._load_mask_for_current()
        if mode in ("preview", "dynamic") and ds is not None and ds.frames and ds.processed_dir is not None:
            if mode == "dynamic":
                self._load_tracked_current_dataset(force=True)
            stack_path = self._viewer_stack_path(ds, "instance")
            tracked_path = ds.processed_dir / "tables" / "instances_tracked_all_frames.csv"
            if (stack_path is None or not stack_path.exists()) and tracked_path.exists():
                try:
                    self._build_instance_stack(ds)
                except Exception:
                    pass
        self.canvas.set_mask_editing(False)
        self.refresh_view()

    def _current_frame_path(self) -> Optional[Path]:
        """Return the current split frame; raw unsplit data are shown directly in the main viewer."""
        ds = self.current_dataset()
        if ds is None or not ds.frames:
            return None
        idx = min(max(0, self.current_frame_index), len(ds.frames) - 1)
        return ds.frames[idx]

    def _load_mask_for_current(self):
        self.current_mask = None
        self.current_mask_path = None
        self.current_mask_dirty = False
        ds = self.current_dataset(); frame = self._current_frame_path()
        if ds is None or frame is None or ds.processed_dir is None:
            return
        path = ds.processed_dir / f"{frame.stem}_mask.tif"
        if not path.exists():
            return
        stacked = self._read_stack_frame(ds, "mask", self.current_frame_index)
        if stacked is not None:
            self.current_mask = (np.asarray(stacked) != 0).astype(np.uint8) * 255
        else:
            self.current_mask = self._cached_mask_2d(path)
        self.current_mask_path = path

    def _save_current_mask_if_dirty(self):
        if self.current_mask_dirty and self.current_mask is not None and self.current_mask_path is not None:
            tifffile.imwrite(self.current_mask_path, self.current_mask.astype(np.uint8))
            self.current_mask_dirty = False
            self._update_mask_stack_frame()
            self.log(f"Auto-saved mask: {self.current_mask_path.name}")

    def _update_mask_stack_frame(self):
        ds = self.current_dataset()
        if ds is None or self.current_mask is None:
            return
        path = self._viewer_stack_path(ds, "mask")
        if path is None or not path.exists():
            return
        stack = self._open_viewer_stack(path)
        if stack is None or self.current_frame_index >= len(stack):
            return
        try:
            stack[self.current_frame_index] = (self.current_mask != 0).astype(np.uint8) * 255
            if hasattr(stack, "flush"):
                stack.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------ Dynamic backend
    def run_dynamic(self):
        targets = self._scope_datasets()
        if not targets:
            return
        self._save_current_mask_if_dirty()
        for ds in targets:
            if not ds.frames or ds.processed_dir is None:
                QMessageBox.warning(self, "No prepared frames", f"Generate Mask first so LinkNex can prepare the time-lapse:\n{ds.name}")
                return
            missing = [f.name for f in ds.frames if not (ds.processed_dir / f"{f.stem}_mask.tif").exists()]
            if missing:
                preview = "\n".join(missing[:8])
                suffix = "\n..." if len(missing) > 8 else ""
                QMessageBox.warning(
                    self,
                    "Missing masks",
                    f"Generate masks before dynamic analysis for {ds.name}.\n\nMissing {len(missing)} frame(s):\n{preview}{suffix}",
                )
                return

        dt = float(self.dt_seconds.value())
        xy = float(self.xy_um.value())
        z = float(self.z_um.value())
        radius = float(self.track_radius.value())
        netdist = float(self.network_dist.value())
        self._run_dynamic_with_log(targets, dt, xy, z, radius, netdist)

    def _run_dynamic_many(self, targets, dt, xy, z, radius, netdist, progress_cb):
        messages = []
        # Global settings in the user-provided backend are set immediately before each run.
        dynamic_backend.SEGMENTATION_DIR_NAME = "processed_output"
        dynamic_backend.TRACKING_OBJECT_TYPE = "puncta"
        dynamic_backend.DT_SECONDS = dt
        dynamic_backend.VOXEL_SIZE_ZYX = (z, xy, xy)
        dynamic_backend.TRACKING_PROFILES["puncta"]["search_radius_um"] = radius
        dynamic_backend.NETWORK_DISTANCE_UM = netdist
        dynamic_backend.DATASET_NAME = "puncta"
        dynamic_backend.SAVE_TRACKED_POINTCLOUDS = True
        dynamic_backend.SAVE_FRAME_NETWORK_PNGS = False
        dynamic_backend.SAVE_TRAJECTORY_PNGS = False
        dynamic_backend.SAVE_NETWORK_TIME_SERIES = False

        for dataset_index, ds in enumerate(targets, start=1):
            ds.processed_dir.mkdir(parents=True, exist_ok=True)
            log_path = ds.processed_dir / "dynamic_analysis.log"
            with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
                stream = _DynamicLogStream(progress_cb, log_file)
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    print("=" * 78)
                    print(f"LinkNex Dynamic Analysis | dataset {dataset_index}/{len(targets)}: {ds.name}")
                    print(f"Frames: {len(ds.frames)}")
                    print(f"Input folder: {ds.frame_dir}")
                    print(f"Output folder: {ds.processed_dir}")
                    print(f"Persistent surface PCD folder: {ds.processed_dir / 'tracked_surface_pointclouds'}")
                    print("=" * 78)

                    print("[STAGE 1/4] Clearing stale downstream outputs...")
                    self._clear_downstream_outputs(ds)
                    print("[STAGE 1/4] Done.")

                    print("[STAGE 2/4] Reviewed mask -> voxel / mesh / surface PCD")
                    dynamic_backend.run_reviewed_mask_to_pcd(ds.frame_dir)
                    print("[STAGE 2/4] Done.")

                    print("[STAGE 3/4] Instance segmentation + morphology")
                    dynamic_backend.run_instance_segmentation(ds.frame_dir)
                    print("[STAGE 3/4] Done.")

                    print("[STAGE 4/4] Persistent tracking + temporal network")
                    dynamic_backend.process_cell(ds.frame_dir)
                    print("[STAGE 4/4] Done.")

                    print("[VIEWER STACK] Building instance time-lapse stack...")
                    instance_stack = self._build_instance_stack(ds)
                    if instance_stack is not None:
                        print(f"[VIEWER STACK] Saved: {instance_stack}")

                    tables_dir = ds.processed_dir / "tables"
                    if tables_dir.exists():
                        for saved in sorted(tables_dir.glob("*.csv"), key=lambda p: natural_key(p.name)):
                            print(f"[OUTPUT TABLE] {saved}")
                    surface_dir = ds.processed_dir / "tracked_surface_pointclouds"
                    if surface_dir.exists():
                        count = len(list(surface_dir.glob("*_pcd_tracked.txt")))
                        print(f"[OUTPUT PCD] {count} tracked surface PCD file(s) -> {surface_dir}")
                    print(f"[LOG SAVED] {log_path}")
                    print(f"[COMPLETE] {ds.name}")

            messages.append(f"{ds.name}: dynamic analysis finished | log: {log_path}")
        return messages

    def _run_dynamic_with_log(self, targets, dt, xy, z, radius, netdist):
        self._set_button_busy(self.run_dynamic_btn, True, "Running Dynamic...")
        dialog = DynamicLogDialog(self)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._dynamic_log_dialog = dialog

        thread = QThread(self)
        worker = DynamicLogWorker(
            lambda emit: self._run_dynamic_many(targets, dt, xy, z, radius, netdist, emit)
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(dialog.append_line)

        def finished(result):
            try:
                self._after_dynamic(result)
                dialog.mark_complete()
                QTimer.singleShot(900, dialog.accept)
            finally:
                self._set_button_busy(self.run_dynamic_btn, False)
                thread.quit()

        def failed(tb):
            self._set_button_busy(self.run_dynamic_btn, False)
            dialog.append_line(tb)
            dialog.mark_failed()
            QMessageBox.critical(self, "Dynamic analysis failed", tb.splitlines()[-1] if tb else "Unknown error")
            thread.quit()

        worker.finished.connect(finished)
        worker.error.connect(failed)
        thread.finished.connect(worker.deleteLater)

        def cleanup_thread():
            self._threads = [(t, w) for (t, w) in self._threads if t is not thread]
            thread.deleteLater()

        thread.finished.connect(cleanup_thread)
        self._threads.append((thread, worker))
        thread.start()

    def _clear_downstream_outputs(self, ds: DynamicsDataset):
        """Remove derived files so corrected masks always propagate downstream."""
        if ds.processed_dir is None or not ds.processed_dir.exists():
            return
        proc = ds.processed_dir
        derived_suffixes = (
            "_voxel.txt", "_mesh.obj", "_pcd.txt",
            "_voxel_instance.txt", "_pcd_instance.txt", "_centers.txt",
        )
        for frame in ds.frames:
            for suffix in derived_suffixes:
                path = proc / f"{frame.stem}{suffix}"
                if path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass
        instance_stack = self._viewer_stack_path(ds, "instance")
        if instance_stack is not None and instance_stack.exists():
            try:
                self._invalidate_viewer_stack(instance_stack)
                instance_stack.unlink()
            except Exception:
                pass
        for name in ["morphology_cell_mean_all.csv"]:
            path = proc / name
            if path.exists():
                try: path.unlink()
                except Exception: pass
        import shutil
        for name in ["tables", "figures", "tracked_surface_pointclouds", "tracked_voxel_pointclouds", "tracking_videos"]:
            path = proc / name
            if path.exists() and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    def _after_dynamic(self, messages):
        self._tracked_cache.clear()
        for m in messages: self.log(m)
        self.set_view_mode("dynamic")


    # ------------------------------------------------------------------ ROI quantitative dynamics
    @staticmethod
    def _roi_mask_from_definition(definition: Dict[str, object], image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        shape = definition["shape_type"]
        g = definition["geometry"]
        if shape == "rectangle":
            x0 = int(round(float(g["x"])))
            y0 = int(round(float(g["y"])))
            # cv2.rectangle includes both endpoints. Subtract one pixel from
            # the far edge so an integer W×H ROI rasterizes to W×H pixels
            # instead of (W+1)×(H+1).
            x1 = int(round(float(g["x"]) + float(g["width"]))) - 1
            y1 = int(round(float(g["y"]) + float(g["height"]))) - 1
            if x1 >= x0 and y1 >= y0:
                cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
        elif shape == "circle":
            center = (int(round(float(g["cx"]))), int(round(float(g["cy"]))))
            radius = max(1, int(round(float(g["radius"]))))
            cv2.circle(mask, center, radius, 255, thickness=-1)
        elif shape == "polygon":
            pts = np.asarray(g.get("points", []), dtype=float)
            if len(pts) >= 3:
                pts = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [pts], 255)
        return mask != 0

    @staticmethod
    def _safe_cv(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.nan
        mean = float(np.mean(arr))
        if abs(mean) < 1e-12:
            return np.nan
        return float(np.std(arr, ddof=0) / mean)

    @staticmethod
    def _instance_projected_metrics(
        points_xyz: np.ndarray,
        image: np.ndarray,
        xy_um: float,
        z_um: float,
    ) -> Dict[str, float]:
        points = np.asarray(points_xyz, dtype=float)
        if len(points) == 0:
            return {}
        xy = np.rint(points[:, :2]).astype(np.int64)
        xy = np.unique(xy, axis=0)
        h, w = image.shape[:2]
        valid = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        xy = xy[valid]
        if len(xy) == 0:
            return {}

        x = xy[:, 0]
        y = xy[:, 1]
        area_px = float(len(xy))
        area_um2 = float(area_px * xy_um * xy_um)
        voxel_count = float(len(points))
        volume_um3 = float(voxel_count * xy_um * xy_um * z_um)
        eq_diam_um = float(2.0 * math.sqrt(max(area_um2, 0.0) / math.pi)) if area_um2 > 0 else 0.0

        xmin, xmax = int(np.min(x)), int(np.max(x))
        ymin, ymax = int(np.min(y)), int(np.max(y))
        bbox_w_px = float(xmax - xmin + 1)
        bbox_h_px = float(ymax - ymin + 1)
        aspect_ratio = float(max(bbox_w_px, bbox_h_px) / max(1.0, min(bbox_w_px, bbox_h_px)))

        crop = np.zeros((int(bbox_h_px) + 2, int(bbox_w_px) + 2), dtype=np.uint8)
        crop[y - ymin + 1, x - xmin + 1] = 255
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter_px = float(sum(cv2.arcLength(c, True) for c in contours)) if contours else np.nan

        # Shape ratios must use area and perimeter from the same geometric
        # representation.  The previous code mixed pixel-count area with an
        # OpenCV contour perimeter, which can yield circularity > 1 for small
        # raster objects.  Keep pixel-count area for projected-area reporting,
        # but use contour area for circularity/solidity.
        contour_area_px2 = float(sum(cv2.contourArea(c) for c in contours)) if contours else np.nan
        circularity = (
            float(4.0 * math.pi * contour_area_px2 / (perimeter_px * perimeter_px))
            if np.isfinite(contour_area_px2) and contour_area_px2 > 1e-8
            and np.isfinite(perimeter_px) and perimeter_px > 1e-8
            else np.nan
        )
        if contours:
            contour_points = np.vstack(contours)
            hull = cv2.convexHull(contour_points) if len(contour_points) >= 3 else None
        else:
            hull = None
        hull_area_px = float(cv2.contourArea(hull)) if hull is not None else np.nan
        solidity = (
            float(contour_area_px2 / hull_area_px)
            if np.isfinite(contour_area_px2) and contour_area_px2 >= 0
            and np.isfinite(hull_area_px) and hull_area_px > 1e-8
            else np.nan
        )

        vals = np.asarray(image[y, x], dtype=float)
        vals = vals[np.isfinite(vals)]
        return {
            "projected_area_px2": area_px,
            "projected_area_um2": area_um2,
            "voxel_count": voxel_count,
            "volume_um3": volume_um3,
            "equivalent_diameter_um": eq_diam_um,
            "bbox_width_um": bbox_w_px * xy_um,
            "bbox_height_um": bbox_h_px * xy_um,
            "aspect_ratio": aspect_ratio,
            "perimeter_um": perimeter_px * xy_um if np.isfinite(perimeter_px) else np.nan,
            "circularity": circularity,
            "solidity": solidity,
            "intensity_mean": float(np.mean(vals)) if len(vals) else np.nan,
            "intensity_median": float(np.median(vals)) if len(vals) else np.nan,
            "intensity_max": float(np.max(vals)) if len(vals) else np.nan,
            "intensity_min": float(np.min(vals)) if len(vals) else np.nan,
            "integrated_intensity": float(np.sum(vals)) if len(vals) else np.nan,
        }

    @staticmethod
    def _track_speed_lookup(tracked: pd.DataFrame, dt_seconds: float) -> Dict[Tuple[int, int], Tuple[float, float]]:
        lookup: Dict[Tuple[int, int], Tuple[float, float]] = {}
        if tracked is None or tracked.empty:
            return lookup
        for tid, g in tracked.groupby("track_id"):
            g = g.sort_values("frame")
            frames = g["frame"].to_numpy(int)
            xyz = g[["centroid_x_um", "centroid_y_um", "centroid_z_um"]].to_numpy(float)
            for i in range(1, len(g)):
                delta_frames = max(1, int(frames[i] - frames[i - 1]))
                disp = float(np.linalg.norm(xyz[i] - xyz[i - 1]))
                speed = disp / max(1e-12, delta_frames * dt_seconds)
                lookup[(int(frames[i]), int(tid))] = (disp, speed)
        return lookup

    def export_roi_dynamics_csv(self):
        ds = self.current_dataset()
        if ds is None or not ds.frames or ds.processed_dir is None:
            QMessageBox.warning(self, "No prepared data", "Generate Mask first so LinkNex can prepare the time-lapse.")
            return
        if not self.rois:
            QMessageBox.warning(self, "No ROI", "Draw at least one ROI first.")
            return

        # ROI morphology/intensity/count only requires instance segmentation.
        # Tracking is optional and, when available, adds speed/entry/exit fields.
        has_instance = any(
            (ds.processed_dir / f"{frame.stem}_voxel_instance.txt").exists()
            for frame in ds.frames
        )
        if not has_instance:
            QMessageBox.warning(
                self,
                "No instance results",
                "No *_voxel_instance.txt files were found. Run instance/dynamic analysis first.",
            )
            return

        self._save_rois_for_current_dataset()
        roi_defs = [self._serialize_roi(r) for r in self.rois]
        xy_um = float(self.xy_um.value())
        z_um = float(self.z_um.value())
        dt_seconds = float(self.dt_seconds.value())
        self._run_threaded(
            lambda: self._export_roi_dynamics_worker(ds, roi_defs, xy_um, z_um, dt_seconds),
            self._after_roi_csv,
            button=self.export_roi_csv_btn,
            running_text="Exporting ROI CSV...",
        )

    def _export_roi_dynamics_worker(
        self,
        ds: DynamicsDataset,
        roi_defs: Sequence[Dict[str, object]],
        xy_um: float,
        z_um: float,
        dt_seconds: float,
    ) -> Path:
        tracked = self._tracked_df(ds, force=True)
        # Tracking is optional for ROI export. If absent, morphology/intensity/count
        # are still exported; track-related fields remain empty/NaN/0.
        speed_lookup = self._track_speed_lookup(tracked, dt_seconds) if tracked is not None else {}
        rows_out: List[Dict[str, object]] = []

        for frame_index, frame_path in enumerate(ds.frames):
            image = np.asarray(read_frame_2d(frame_path), dtype=float)
            h, w = image.shape[:2]
            if tracked is not None and not tracked.empty:
                frame_rows = tracked[tracked["frame_name"].astype(str) == frame_path.stem].copy()
                if frame_rows.empty:
                    frame_rows = tracked[tracked["frame"].astype(int) == int(frame_index)].copy()
            else:
                frame_rows = pd.DataFrame()

            voxel_path = ds.processed_dir / f"{frame_path.stem}_voxel_instance.txt"
            group_map = {}
            if voxel_path.exists():
                arr = read_numeric_txt(voxel_path)
                if arr.shape[1] >= 6:
                    colors = np.rint(arr[:, 3:6]).astype(np.int32)
                    unique, inverse = np.unique(colors, axis=0, return_inverse=True)
                    for idx, color in enumerate(unique):
                        key = tuple(int(v) for v in color)
                        if key == (0, 0, 0):
                            continue
                        group_map[key] = arr[np.flatnonzero(inverse == idx), :3]

            for definition in roi_defs:
                roi_mask = self._roi_mask_from_definition(definition, (h, w))
                shape_metrics = self._roi_shape_metrics_from_definition(definition, xy_um)
                roi_pixel_count = int(np.count_nonzero(roi_mask))
                roi_area_um2_mask = float(roi_pixel_count * xy_um * xy_um)
                roi_vals = image[roi_mask]
                roi_vals = roi_vals[np.isfinite(roi_vals)]

                instance_metrics = []
                overlap_instance_count = 0
                overlap_union = np.zeros((h, w), dtype=bool)
                ids_in_roi = []
                speeds = []
                displacements = []
                new_track_count = 0
                center_x = float(shape_metrics.get("roi_center_x_px", np.nan))
                center_y = float(shape_metrics.get("roi_center_y_px", np.nan))
                centroid_distances_um = []

                # Match tracked rows by frame-local voxel RGB when tracking exists.
                tracked_by_color = {}
                if frame_rows is not None and not frame_rows.empty:
                    for r in frame_rows.itertuples(index=False):
                        if all(np.isfinite([r.voxel_r, r.voxel_g, r.voxel_b])):
                            tracked_by_color[(int(r.voxel_r), int(r.voxel_g), int(r.voxel_b))] = r

                # Iterate the actual segmented instances, not the tracking table.
                # This lets ROI CSV export work immediately after instance segmentation.
                for color, points in group_map.items():
                    if points is None or len(points) == 0:
                        continue

                    xy = np.unique(np.rint(points[:, :2]).astype(np.int64), axis=0)
                    valid = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
                    xy = xy[valid]
                    if len(xy) == 0:
                        continue
                    inside = roi_mask[xy[:, 1], xy[:, 0]]
                    if np.any(inside):
                        overlap_instance_count += 1
                        in_xy = xy[inside]
                        overlap_union[in_xy[:, 1], in_xy[:, 0]] = True

                    tracked_row = tracked_by_color.get(color)
                    if tracked_row is not None:
                        centroid_x = float(tracked_row.centroid_x)
                        centroid_y = float(tracked_row.centroid_y)
                    else:
                        centroid_x = float(np.mean(points[:, 0]))
                        centroid_y = float(np.mean(points[:, 1]))

                    cx = int(round(centroid_x))
                    cy = int(round(centroid_y))
                    centroid_inside = 0 <= cx < w and 0 <= cy < h and bool(roi_mask[cy, cx])
                    if not centroid_inside:
                        continue

                    metrics = self._instance_projected_metrics(points, image, xy_um, z_um)
                    if not metrics:
                        continue

                    if tracked_row is not None and np.isfinite(getattr(tracked_row, "track_id", np.nan)):
                        tid = int(tracked_row.track_id)
                        metrics["track_id"] = tid
                        ids_in_roi.append(tid)
                        if getattr(tracked_row, "tracking_status", "") == "new":
                            new_track_count += 1
                        disp_speed = speed_lookup.get((int(tracked_row.frame), tid))
                        if disp_speed is not None:
                            displacements.append(float(disp_speed[0]))
                            speeds.append(float(disp_speed[1]))
                    else:
                        metrics["track_id"] = np.nan

                    instance_metrics.append(metrics)
                    if np.isfinite(center_x) and np.isfinite(center_y):
                        centroid_distances_um.append(
                            float(math.hypot(centroid_x - center_x, centroid_y - center_y) * xy_um)
                        )

                def vals(key):
                    return np.asarray([m.get(key, np.nan) for m in instance_metrics], dtype=float)

                def mean(key):
                    v = vals(key); v = v[np.isfinite(v)]
                    return float(np.mean(v)) if len(v) else np.nan

                def median(key):
                    v = vals(key); v = v[np.isfinite(v)]
                    return float(np.median(v)) if len(v) else np.nan

                def total(key):
                    v = vals(key); v = v[np.isfinite(v)]
                    return float(np.sum(v)) if len(v) else 0.0

                organelle_count = len(instance_metrics)
                overlap_pixels = int(np.count_nonzero(overlap_union & roi_mask))
                row = dict(shape_metrics)
                row.update({
                    "dataset": ds.name,
                    "frame": int(frame_index),
                    "timepoint": int(frame_index + 1),
                    "frame_name": frame_path.stem,
                    "time_seconds": float(frame_index * dt_seconds),
                    "roi_membership_rule": "instance_centroid_inside_roi",
                    "roi_mask_pixel_count": roi_pixel_count,
                    "roi_mask_area_um2": roi_area_um2_mask,
                    "roi_intensity_mean": float(np.mean(roi_vals)) if len(roi_vals) else np.nan,
                    "roi_intensity_median": float(np.median(roi_vals)) if len(roi_vals) else np.nan,
                    "roi_intensity_min": float(np.min(roi_vals)) if len(roi_vals) else np.nan,
                    "roi_intensity_max": float(np.max(roi_vals)) if len(roi_vals) else np.nan,
                    "roi_intensity_std": float(np.std(roi_vals)) if len(roi_vals) else np.nan,
                    "roi_integrated_intensity": float(np.sum(roi_vals)) if len(roi_vals) else np.nan,
                    "organelle_count": int(organelle_count),
                    "organelle_overlap_count": int(overlap_instance_count),
                    "organelle_density_per_100_um2": float(organelle_count / roi_area_um2_mask * 100.0) if roi_area_um2_mask > 0 else np.nan,
                    "organelle_overlap_pixels": overlap_pixels,
                    "organelle_area_fraction_in_roi": float(overlap_pixels / roi_pixel_count) if roi_pixel_count > 0 else np.nan,
                    "track_ids_in_roi": ";".join(str(v) for v in sorted(set(ids_in_roi))),
                    "new_track_count_in_roi": int(new_track_count),
                    "total_projected_area_um2": total("projected_area_um2"),
                    "mean_projected_area_um2": mean("projected_area_um2"),
                    "median_projected_area_um2": median("projected_area_um2"),
                    "projected_area_cv": self._safe_cv(vals("projected_area_um2")),
                    "total_voxel_count": total("voxel_count"),
                    "mean_voxel_count": mean("voxel_count"),
                    "total_volume_um3": total("volume_um3"),
                    "mean_volume_um3": mean("volume_um3"),
                    "mean_equivalent_diameter_um": mean("equivalent_diameter_um"),
                    "median_equivalent_diameter_um": median("equivalent_diameter_um"),
                    "mean_bbox_width_um": mean("bbox_width_um"),
                    "mean_bbox_height_um": mean("bbox_height_um"),
                    "mean_aspect_ratio": mean("aspect_ratio"),
                    "mean_perimeter_um": mean("perimeter_um"),
                    "mean_circularity": mean("circularity"),
                    "mean_solidity": mean("solidity"),
                    "mean_instance_intensity": mean("intensity_mean"),
                    "median_instance_intensity": median("intensity_mean"),
                    "max_instance_intensity": float(np.nanmax(vals("intensity_max"))) if organelle_count and np.any(np.isfinite(vals("intensity_max"))) else np.nan,
                    "total_instance_integrated_intensity": total("integrated_intensity"),
                    "instance_intensity_cv": self._safe_cv(vals("intensity_mean")),
                    "mean_track_displacement_um": float(np.mean(displacements)) if displacements else np.nan,
                    "mean_track_speed_um_per_s": float(np.mean(speeds)) if speeds else np.nan,
                    "max_track_speed_um_per_s": float(np.max(speeds)) if speeds else np.nan,
                    "mean_centroid_distance_to_roi_center_um": float(np.mean(centroid_distances_um)) if centroid_distances_um else np.nan,
                })
                rows_out.append(row)

        df_out = pd.DataFrame(rows_out)
        if not df_out.empty:
            df_out["roi_track_entry_count"] = 0
            df_out["roi_track_exit_count"] = 0
            df_out["roi_track_persistent_count"] = 0
            for roi_name, g in df_out.groupby("roi_name", sort=False):
                idxs = g.sort_values("frame").index.tolist()
                previous_ids = set()
                first_frame = True
                for idx in idxs:
                    text_ids = str(df_out.at[idx, "track_ids_in_roi"] or "")
                    current_ids = {int(v) for v in text_ids.split(";") if v.strip().isdigit()}
                    if first_frame:
                        # The first observation establishes baseline occupancy;
                        # tracks already present are not biological "entries".
                        df_out.at[idx, "roi_track_entry_count"] = 0
                        df_out.at[idx, "roi_track_exit_count"] = 0
                        df_out.at[idx, "roi_track_persistent_count"] = 0
                        first_frame = False
                    else:
                        df_out.at[idx, "roi_track_entry_count"] = int(len(current_ids - previous_ids))
                        df_out.at[idx, "roi_track_exit_count"] = int(len(previous_ids - current_ids))
                        df_out.at[idx, "roi_track_persistent_count"] = int(len(current_ids & previous_ids))
                    previous_ids = current_ids
                for col in ["organelle_count", "roi_intensity_mean", "mean_projected_area_um2", "mean_track_speed_um_per_s"]:
                    values = pd.to_numeric(df_out.loc[idxs, col], errors="coerce")
                    df_out.loc[idxs, f"{col}_change_from_previous"] = values.diff().to_numpy()

        out_dir = ds.processed_dir / "ROI_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "roi_dynamic_metrics.csv"
        df_out.to_csv(out_path, index=False)
        self._save_roi_summary_plots(df_out, out_dir)
        return out_path

    @staticmethod
    def _save_roi_summary_plots(df_out: pd.DataFrame, out_dir: Path) -> None:
        """Save only ROI organelle-count and ROI mean-intensity time courses.

        Files are written directly into ROI_results; no additional figure
        subfolder is created.  Multiple ROIs are drawn as separate lines.
        """
        if df_out is None or df_out.empty:
            return

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        x_col = "time_seconds" if "time_seconds" in df_out.columns else "timepoint"
        x_label = "Time (s)" if x_col == "time_seconds" else "Time point"

        plot_specs = [
            ("organelle_count", "Organelle count", "roi_organelle_count.png"),
            ("roi_intensity_mean", "Mean intensity", "roi_mean_intensity.png"),
        ]

        for metric, y_label, filename in plot_specs:
            if metric not in df_out.columns:
                continue
            fig = Figure(figsize=(6.4, 4.6), dpi=180)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            groups = list(df_out.groupby("roi_name", sort=False)) if "roi_name" in df_out.columns else [("ROI", df_out)]
            for roi_name, g in groups:
                g = g.sort_values("frame") if "frame" in g.columns else g
                xv = pd.to_numeric(g[x_col], errors="coerce").to_numpy(dtype=float)
                yv = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(xv) & np.isfinite(yv)
                if np.any(valid):
                    ax.plot(
                        xv[valid], yv[valid], marker="o", markersize=4.5,
                        linewidth=2.0, label=str(roi_name),
                    )

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
            ax.tick_params(axis="both", width=2.0, length=5)
            if len(groups) > 1:
                ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")

    def _after_roi_csv(self, path: Path):
        self.log(f"Saved ROI dynamics CSV: {path}")
        QMessageBox.information(
            self,
            "ROI CSV saved",
            f"ROI dynamics results saved to:\n\n{path.parent}\n\n"
            "Files:\n"
            "- roi_dynamic_metrics.csv\n"
            "- roi_organelle_count.png\n"
            "- roi_mean_intensity.png",
        )

    # ------------------------------------------------------------------ Rendering
    def _dataset_display_range(self, ds: DynamicsDataset):
        key = str(ds.frame_dir.resolve()) if ds.frame_dir is not None else self._dataset_key(ds)
        if key in self._display_range_cache:
            return self._display_range_cache[key]
        n = len(ds.frames) if ds.frames else self._source_frame_count(ds)
        if n <= 0:
            return None
        indices = np.unique(np.linspace(0, n - 1, min(12, n), dtype=int))
        lows, highs = [], []
        for idx in indices:
            arr = self._read_original_timepoint(ds, int(idx))
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float32)
            finite = np.isfinite(arr)
            if np.any(finite):
                lows.append(float(np.nanpercentile(arr[finite], 1.0)))
                highs.append(float(np.nanpercentile(arr[finite], 99.7)))
        if not lows:
            value = (0.0, 1.0)
        else:
            value = (float(np.median(lows)), float(np.median(highs)))
            if value[1] <= value[0]:
                value = (value[0], value[0] + 1.0)
        self._display_range_cache[key] = value
        return value

    def open_background_tif(self):
        ds = self.current_dataset()
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open second-channel background time-lapse TIFF",
            "",
            "TIFF (*.tif *.tiff)",
        )
        if not path_str:
            return
        path = Path(path_str)

        # Validate spatial registration against the current main-view time-lapse when possible.
        if ds is not None:
            try:
                ref = self._read_original_timepoint(ds, 0)
                if ref is None:
                    raise ValueError("Could not read the current dataset")
                bg = self._read_external_timepoint(path, 0)
                if bg is None or tuple(bg.shape[:2]) != tuple(ref.shape[:2]):
                    QMessageBox.warning(
                        self,
                        "Background size mismatch",
                        f"Background TIFF first frame is {None if bg is None else bg.shape[:2]}, "
                        f"but the analysis frame is {ref.shape[:2]}.\n"
                        "Use a spatially registered channel with the same image dimensions.",
                    )
                    return
            except Exception as exc:
                QMessageBox.warning(self, "Background TIFF", f"Could not read background TIFF:\n{exc}")
                return

        if ds is not None:
            key = self._dataset_key(ds)
        else:
            key = "__global__"
        self._external_background_paths[key] = path
        self._external_background_active[key] = True
        self.background_tif_btn.setToolTip(str(path))
        self.background_tif_btn.setText("Background Image ✓")
        self.refresh_view()

    def _background_combo_selected(self, _index: int):
        ds = self.current_dataset()
        key = self._dataset_key(ds) if ds is not None else "__global__"
        self._external_background_active[key] = False
        self.refresh_view()

    def _external_background_path(self, ds: Optional[DynamicsDataset]) -> Optional[Path]:
        key = self._dataset_key(ds) if ds is not None else "__global__"
        if not self._external_background_active.get(key, False):
            return None
        path = self._external_background_paths.get(key)
        if path is not None:
            return path
        if key != "__global__" and self._external_background_active.get("__global__", False):
            return self._external_background_paths.get("__global__")
        return None

    def _external_timepoint_count(self, path: Path) -> int:
        try:
            with tifffile.TiffFile(path) as tif:
                if len(tif.pages) > 1:
                    return len(tif.pages)
                shape = tuple(tif.series[0].shape) if tif.series else ()
            return 1 if len(shape) <= 2 else int(shape[0])
        except Exception:
            return 1

    def _read_external_timepoint(self, path: Path, index: int) -> Optional[np.ndarray]:
        idx = max(0, int(index))
        count = max(1, self._external_timepoint_count(path))
        idx = min(idx, count - 1)
        key = (str(Path(path).resolve()), idx)
        cached = self._lru_get(self._external_timepoint_cache, key)
        if cached is not None:
            return cached
        try:
            with tifffile.TiffFile(path) as tif:
                if len(tif.pages) > 1 and idx < len(tif.pages):
                    arr = np.asarray(tif.pages[idx].asarray())
                else:
                    arr = np.asarray(tif.asarray())
                    arr = np.squeeze(arr)
                    if arr.ndim > 2:
                        arr = np.asarray(arr[min(idx, arr.shape[0] - 1)])
            arr = np.squeeze(arr)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                arr = np.mean(arr[..., :3].astype(np.float32), axis=-1)
            while arr.ndim > 2:
                arr = np.max(arr, axis=0)
            return self._lru_put(self._external_timepoint_cache, key, np.asarray(arr))
        except Exception:
            return None

    def _external_display_range(self, path: Path) -> Tuple[float, float]:
        key = str(Path(path).resolve())
        if key in self._external_display_range_cache:
            return self._external_display_range_cache[key]
        n = max(1, self._external_timepoint_count(path))
        indices = np.unique(np.linspace(0, n - 1, min(12, n), dtype=int))
        lows, highs = [], []
        for idx in indices:
            arr = self._read_external_timepoint(path, int(idx))
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float32)
            finite = np.isfinite(arr)
            if np.any(finite):
                lows.append(float(np.nanpercentile(arr[finite], 1.0)))
                highs.append(float(np.nanpercentile(arr[finite], 99.7)))
        if lows:
            value = (float(np.median(lows)), float(np.median(highs)))
            if value[1] <= value[0]:
                value = (value[0], value[0] + 1.0)
        else:
            value = (0.0, 1.0)
        self._external_display_range_cache[key] = value
        return value

    def _background_rgb(self, image_rgb: np.ndarray, ds: Optional[DynamicsDataset] = None, frame_index: int = 0, settings=None) -> np.ndarray:
        if settings is None:
            mode = self.background_combo.currentText()
            external_path = self._external_background_path(ds)
            if external_path is not None:
                mode = "External TIFF"
        else:
            mode = settings["background"]
            raw = settings.get("external_tif")
            external_path = Path(raw) if raw else None

        if mode == "Original image":
            return image_rgb.copy()
        if mode == "External TIFF":
            if external_path is None or not Path(external_path).exists():
                return image_rgb.copy()
            external = self._read_external_timepoint(Path(external_path), int(frame_index))
            if external is None or tuple(external.shape[:2]) != tuple(image_rgb.shape[:2]):
                return image_rgb.copy()
            return normalize_gray_to_rgb(external, self._external_display_range(Path(external_path)))

        if mode == "Black":
            color = (0, 0, 0)
        elif mode == "White":
            color = (255, 255, 255)
        else:
            color = (128, 128, 128)
        out = np.empty_like(image_rgb)
        out[:] = np.asarray(color, dtype=np.uint8)
        return out

    def _tracked_df(self, ds: DynamicsDataset, force=False) -> Optional[pd.DataFrame]:
        if ds.processed_dir is None:
            return None
        key = str(ds.processed_dir.resolve())
        if not force and key in self._tracked_cache:
            return self._tracked_cache[key]
        path = ds.processed_dir / "tables" / "instances_tracked_all_frames.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        self._tracked_cache[key] = df
        return df

    def _load_tracked_current_dataset(self, force=False):
        ds = self.current_dataset()
        return None if ds is None else self._tracked_df(ds, force=force)

    def _apply_instance_overlay_auto(self, base: np.ndarray, ds: DynamicsDataset, frame: Path) -> np.ndarray:
        """
        Default preview overlay. If a voxel-instance file exists, overlay it on
        the original image. Tracking is optional: persistent track colors are
        used when available; otherwise the frame-local instance RGB is used.
        """
        if ds.processed_dir is None:
            return base

        stacked_rgb = self._read_stack_frame(ds, "instance", self.current_frame_index)
        if stacked_rgb is not None and stacked_rgb.ndim == 3 and stacked_rgb.shape[-1] >= 3:
            colors = np.asarray(stacked_rgb[..., :3], dtype=np.uint8)
            mask = np.any(colors != 0, axis=2)
            if np.any(mask):
                result = base.copy()
                blended = result[mask].astype(np.float32) * (1.0 - INSTANCE_OVERLAY_ALPHA) + colors[mask].astype(np.float32) * INSTANCE_OVERLAY_ALPHA
                result[mask] = np.clip(blended, 0, 255).astype(np.uint8)
                return result
            return base

        voxel_path = ds.processed_dir / f"{frame.stem}_voxel_instance.txt"
        if not voxel_path.exists():
            return base
        try:
            arr = self._cached_numeric_txt(voxel_path)
        except Exception:
            return base
        if arr.ndim != 2 or arr.shape[1] < 6:
            return base

        result = base.copy()
        xyz = np.asarray(arr[:, :3], dtype=float)
        local_rgb = np.clip(np.rint(arr[:, 3:6]), 0, 255).astype(np.uint8)
        x = np.rint(xyz[:, 0]).astype(np.int64)
        y = np.rint(xyz[:, 1]).astype(np.int64)
        valid = (
            np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1])
            & (x >= 0) & (x < result.shape[1])
            & (y >= 0) & (y < result.shape[0])
        )
        if not np.any(valid):
            return result

        x = x[valid]
        y = y[valid]
        colors = local_rgb[valid].copy()

        # If tracking exists for this frame, replace frame-local colors with
        # persistent track colors. Otherwise retain instance segmentation colors.
        tracked = self._tracked_df(ds)
        if tracked is not None and not tracked.empty:
            rows = tracked[tracked["frame_name"].astype(str) == frame.stem]
            if not rows.empty:
                rgb_to_track = {}
                for r in rows.itertuples(index=False):
                    vals = [getattr(r, "voxel_r", np.nan), getattr(r, "voxel_g", np.nan), getattr(r, "voxel_b", np.nan)]
                    if np.all(np.isfinite(vals)):
                        rgb_to_track[tuple(int(v) for v in vals)] = int(r.track_id)
                if rgb_to_track:
                    for local_color, tid in rgb_to_track.items():
                        keep = np.all(colors == np.asarray(local_color, dtype=np.uint8)[None, :], axis=1)
                        if np.any(keep):
                            colors[keep] = np.asarray(deterministic_track_rgb(tid), dtype=np.uint8)

        overlay = result.copy()
        overlay[y, x] = colors
        mask = np.zeros(result.shape[:2], dtype=bool)
        mask[y, x] = True
        alpha = INSTANCE_OVERLAY_ALPHA
        if np.any(mask):
            blended = result[mask].astype(np.float32) * (1.0 - alpha) + overlay[mask].astype(np.float32) * alpha
            result[mask] = np.clip(blended, 0, 255).astype(np.uint8)
        return result

    def _apply_dynamic_overlay(self, base: np.ndarray, ds: DynamicsDataset, frame: Path, tracked: pd.DataFrame, settings=None, frame_index: Optional[int] = None) -> np.ndarray:
        result = base.copy()
        rows = tracked[tracked["frame_name"].astype(str) == frame.stem]
        if rows.empty:
            return result

        # Instance pixels come from one prebuilt RGB time-lapse stack. This avoids
        # reparsing the per-frame voxel-instance TXT whenever the slider moves.
        stack_index = self.current_frame_index if frame_index is None else int(frame_index)
        stacked_rgb = self._read_stack_frame(ds, "instance", stack_index)
        if stacked_rgb is not None and stacked_rgb.ndim == 3 and stacked_rgb.shape[-1] >= 3:
            colors = np.asarray(stacked_rgb[..., :3], dtype=np.uint8)
            mask = np.any(colors != 0, axis=2)
            if np.any(mask):
                blended = result[mask].astype(np.float32) * (1.0 - INSTANCE_OVERLAY_ALPHA) + colors[mask].astype(np.float32) * INSTANCE_OVERLAY_ALPHA
                result[mask] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            # Backward-compatible fallback for datasets analyzed before viewer stacks existed.
            proc = ds.processed_dir
            voxel_path = proc / f"{frame.stem}_voxel_instance.txt"
            if voxel_path.exists():
                arr = self._cached_numeric_txt(voxel_path)
                if arr.shape[1] >= 6:
                    rgb = np.rint(arr[:, 3:6]).astype(int)
                    mapping = {}
                    for r in rows.itertuples(index=False):
                        if all(np.isfinite([r.voxel_r, r.voxel_g, r.voxel_b])):
                            mapping[(int(r.voxel_r), int(r.voxel_g), int(r.voxel_b))] = int(r.track_id)
                    overlay = result.copy()
                    mask = np.zeros(result.shape[:2], dtype=bool)
                    for color, tid in mapping.items():
                        keep = np.all(rgb == np.asarray(color)[None, :], axis=1)
                        if not np.any(keep):
                            continue
                        x = np.rint(arr[keep, 0]).astype(int)
                        y = np.rint(arr[keep, 1]).astype(int)
                        valid = (x >= 0) & (x < result.shape[1]) & (y >= 0) & (y < result.shape[0])
                        x, y = x[valid], y[valid]
                        overlay[y, x] = deterministic_track_rgb(tid)
                        mask[y, x] = True
                    if np.any(mask):
                        blended = result[mask].astype(np.float32) * (1.0 - INSTANCE_OVERLAY_ALPHA) + overlay[mask].astype(np.float32) * INSTANCE_OVERLAY_ALPHA
                        result[mask] = np.clip(blended, 0, 255).astype(np.uint8)

        show_tracks = self.show_tracks.isChecked() if settings is None else bool(settings["show_tracks"])
        if show_tracks:
            current_frame_value = int(rows["frame"].iloc[0])
            current_ids = sorted(set(rows["track_id"].astype(int).tolist()))
            trail_n = int(self.trail_length.value()) if settings is None else int(settings["trail_length"])
            for tid in current_ids:
                g = tracked[(tracked["track_id"].astype(int) == tid) & (tracked["frame"].astype(int) <= current_frame_value)].sort_values("frame")
                if trail_n > 0:
                    g = g[g["frame"].astype(int) >= current_frame_value - trail_n + 1]
                if len(g) < 2:
                    continue
                pts = np.rint(g[["centroid_x", "centroid_y"]].to_numpy(float)).astype(int)
                color = deterministic_track_rgb(tid)
                for i in range(len(pts) - 1):
                    p1 = tuple(int(v) for v in pts[i]); p2 = tuple(int(v) for v in pts[i + 1])
                    cv2.line(result, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
        return result

    def render_current_rgb(self) -> Optional[np.ndarray]:
        ds = self.current_dataset()
        if ds is None:
            return None

        image = self._read_original_timepoint(ds, self.current_frame_index)
        if image is None:
            return None
        img_rgb = normalize_gray_to_rgb(image, self._dataset_display_range(ds))
        base = self._background_rgb(
            img_rgb,
            ds=ds,
            frame_index=self.current_frame_index,
        )

        # Before Split, the main viewer is the raw source time-lapse. Analysis
        # overlays only become available after split-frame processing.
        if not ds.frames or not self.overlay_visible:
            return base

        frame = self._current_frame_path()
        if frame is None:
            return base

        if self.view_mode == "preview":
            return self._apply_instance_overlay_auto(base, ds, frame)

        if self.view_mode == "mask":
            result = base.copy()
            if self.current_mask is not None:
                m = self.current_mask != 0
                color = np.array([0, 174, 239], dtype=np.uint8)
                alpha = MASK_OVERLAY_ALPHA
                result[m] = np.clip(
                    result[m].astype(np.float32) * (1 - alpha)
                    + color.astype(np.float32) * alpha,
                    0,
                    255,
                ).astype(np.uint8)
            return result

        if self.view_mode == "dynamic":
            tracked = self._tracked_df(ds)
            if tracked is None:
                return base
            return self._apply_dynamic_overlay(base, ds, frame, tracked)

        return base

    def refresh_view(self, *_args):
        rgb = self.render_current_rgb()
        if rgb is not None:
            self.canvas.set_rgb(rgb)
        else:
            placeholder = np.zeros((720, 960, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Open a time-lapse TIFF to start",
                (220, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (180, 190, 205),
                2,
                cv2.LINE_AA,
            )
            self.canvas.set_rgb(placeholder)

    # ------------------------------------------------------------------ Video export
    def export_video(self):
        ds = self.current_dataset()
        if ds is None or not ds.frames or ds.processed_dir is None:
            return
        tracked = self._tracked_df(ds, force=True)
        if tracked is None:
            QMessageBox.warning(self, "No dynamic results", "Run Dynamic Analysis first.")
            return
        external_path = self._external_background_path(ds)
        settings = {
            "background": "External TIFF" if external_path is not None else self.background_combo.currentText(),
            "external_tif": str(external_path) if external_path is not None else "",
            "show_tracks": self.show_tracks.isChecked(),
            "trail_length": int(self.trail_length.value()),
        }
        self._run_threaded(lambda: self._export_video_worker(ds, settings), self._after_export, button=self.export_btn, running_text="Exporting MP4...")

    def _export_video_worker(self, ds: DynamicsDataset, settings):
        tracked = self._tracked_df(ds, force=True)
        if tracked is None:
            raise FileNotFoundError("instances_tracked_all_frames.csv not found")
        display_range = self._dataset_display_range(ds)
        first_gray = self._read_original_timepoint(ds, 0)
        if first_gray is None:
            raise FileNotFoundError("Could not read original viewer stack")
        first = normalize_gray_to_rgb(first_gray, display_range)
        h, w = first.shape[:2]
        out_dir = ds.processed_dir / "tracking_videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "tif_instance_overlay_tracks.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Could not open MP4 writer. Please ensure OpenCV video codecs are available.")
        try:
            for frame_index, frame in enumerate(ds.frames):
                gray = self._read_original_timepoint(ds, frame_index)
                if gray is None:
                    raise FileNotFoundError(f"Could not read viewer stack frame {frame_index + 1}")
                img_rgb = normalize_gray_to_rgb(gray, display_range)
                if img_rgb.shape[:2] != (h, w):
                    raise ValueError(f"Frame size changed: {frame.name} is {img_rgb.shape[:2]}, expected {(h, w)}")
                base = self._background_rgb(img_rgb, ds=ds, frame_index=frame_index, settings=settings)
                rendered = self._apply_dynamic_overlay(base, ds, frame, tracked, settings=settings, frame_index=frame_index)
                writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return out_path

    def _after_export(self, path):
        self.log(f"Saved video: {path}")
        QMessageBox.information(self, "Video saved", str(path))

    # ------------------------------------------------------------------ Thread utility
    def _run_threaded(
        self,
        func: Callable[[], object],
        on_finished: Callable[[object], None],
        button: Optional[QPushButton] = None,
        running_text: str = "Running...",
    ):
        self._set_button_busy(button, True, running_text)
        thread = QThread(self)
        worker = FunctionWorker(func)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def finished(result):
            try:
                on_finished(result)
            finally:
                self._set_button_busy(button, False)
                thread.quit()

        def failed(tb):
            self._set_button_busy(button, False)
            self.log(tb)
            QMessageBox.critical(self, "Operation failed", tb.splitlines()[-1] if tb else "Unknown error")
            thread.quit()

        worker.finished.connect(finished)
        worker.error.connect(failed)
        thread.finished.connect(worker.deleteLater)

        def cleanup_thread():
            self._threads = [(t, w) for (t, w) in self._threads if t is not thread]
            thread.deleteLater()

        thread.finished.connect(cleanup_thread)
        self._threads.append((thread, worker))
        thread.start()
