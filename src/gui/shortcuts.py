"""
Keyboard Shortcuts Panel - Display and manage shortcuts.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGridLayout, QDialog, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QFont, QShortcut
from loguru import logger


class ShortcutItem(QFrame):
    """Single shortcut display item."""
    
    def __init__(self, key: str, description: str, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        
        # Key badge
        key_label = QLabel(key)
        key_label.setStyleSheet("""
            background: #30363d;
            color: #e6edf3;
            padding: 4px 8px;
            border-radius: 4px;
            font-family: monospace;
            font-weight: 600;
        """)
        key_label.setFixedWidth(80)
        layout.addWidget(key_label)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #8b949e;")
        layout.addWidget(desc_label, 1)


class ShortcutsDialog(QDialog):
    """Dialog showing all keyboard shortcuts."""
    
    def __init__(self, shortcuts: dict, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("⌨️ Keyboard Shortcuts")
        self.setMinimumSize(400, 500)
        self.setStyleSheet("""
            QDialog {
                background: #161b22;
            }
            QLabel {
                color: #e6edf3;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        
        # Header
        header = QLabel("⌨️ Keyboard Shortcuts")
        header.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(header)
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(4)
        
        # Add shortcuts by category
        for category, items in shortcuts.items():
            # Category header
            cat_label = QLabel(category)
            cat_label.setStyleSheet("""
                font-weight: 600;
                color: #58a6ff;
                margin-top: 12px;
                margin-bottom: 4px;
            """)
            content_layout.addWidget(cat_label)
            
            for key, desc in items.items():
                item = ShortcutItem(key, desc)
                content_layout.addWidget(item)
        
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)


class ShortcutManager:
    """
    Manages keyboard shortcuts for the application.
    """
    
    SHORTCUTS = {
        "Playback": {
            "Space": "Start / Stop processing",
            "R": "Toggle recording",
            "Esc": "Stop all",
        },
        "View": {
            "1": "Switch to Cameras tab",
            "2": "Switch to 3D Scene tab",
            "3": "Switch to Analytics tab",
            "F": "Toggle fullscreen",
            "H": "Toggle heatmap panel",
            "A": "Toggle alerts panel",
        },
        "Camera": {
            "+": "Zoom in",
            "-": "Zoom out",
            "←→↑↓": "Pan view",
            "0": "Reset view",
        },
        "Audio": {
            "M": "Mute/unmute audio",
            "L": "Toggle localization overlay",
        },
        "System": {
            "Ctrl+,": "Open Settings",
            "Ctrl+Q": "Quit application",
            "Ctrl+S": "Save session",
            "Ctrl+E": "Export data",
            "?": "Show shortcuts",
        }
    }
    
    def __init__(self, main_window):
        self.main_window = main_window
        self._shortcuts: list[QShortcut] = []
        
        self._setup_shortcuts()
        logger.info("ShortcutManager initialized")
    
    def _setup_shortcuts(self):
        """Setup all keyboard shortcuts."""
        mw = self.main_window
        
        # Playback
        self._add_shortcut("Space", self._toggle_playback)
        self._add_shortcut("R", self._toggle_recording)
        self._add_shortcut("Escape", self._stop_all)
        
        # View
        self._add_shortcut("1", lambda: mw.main_tabs.setCurrentIndex(0))
        self._add_shortcut("2", lambda: mw.main_tabs.setCurrentIndex(1))
        self._add_shortcut("3", lambda: mw.main_tabs.setCurrentIndex(2))
        self._add_shortcut("F", self._toggle_fullscreen)
        self._add_shortcut("H", lambda: mw.heatmap_dock.setVisible(
            not mw.heatmap_dock.isVisible()))
        self._add_shortcut("A", lambda: mw.alert_dock.setVisible(
            not mw.alert_dock.isVisible()))
        
        # Audio
        self._add_shortcut("M", self._toggle_mute)
        
        # System
        self._add_shortcut("Ctrl+Q", mw.close)
        self._add_shortcut("?", self.show_shortcuts_dialog)
    
    def _add_shortcut(self, key: str, callback):
        """Add a keyboard shortcut."""
        shortcut = QShortcut(QKeySequence(key), self.main_window)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)
    
    def _toggle_playback(self):
        """Toggle start/stop."""
        mw = self.main_window
        if mw.worker and mw.worker.is_running:
            mw._on_stop()
        else:
            mw._on_start()
    
    def _toggle_recording(self):
        """Toggle recording."""
        self.main_window._toggle_recording()
    
    def _stop_all(self):
        """Stop all operations."""
        self.main_window._on_stop()
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        mw = self.main_window
        if mw.isFullScreen():
            mw.showNormal()
        else:
            mw.showFullScreen()
    
    def _toggle_mute(self):
        """Toggle audio mute."""
        logger.info("Audio mute toggled")
    
    def show_shortcuts_dialog(self):
        """Show shortcuts dialog."""
        dialog = ShortcutsDialog(self.SHORTCUTS, self.main_window)
        dialog.exec()
