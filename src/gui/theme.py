"""
Optimized modern theme for PyQt6 GUI - Clean and spacious layout.
"""

DARK_THEME = """
QMainWindow {
    background-color: #0d1117;
}

QWidget {
    background-color: transparent;
    color: #e6edf3;
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
    font-size: 12px;
}

/* Dock Widgets - Key for layout */
QDockWidget {
    font-size: 13px;
    font-weight: 600;
    titlebar-close-icon: url(close.png);
    titlebar-normal-icon: url(undock.png);
}

QDockWidget::title {
    background: #161b22;
    padding: 8px 12px;
    border-bottom: 1px solid #30363d;
}

QDockWidget::close-button, QDockWidget::float-button {
    padding: 4px;
}

/* Frames - Less border, more breathing room */
QFrame {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}

QLabel {
    color: #e6edf3;
    background: transparent;
    border: none;
    padding: 2px;
}

QLabel#title {
    font-size: 20px;
    font-weight: 600;
    color: #ffffff;
}

QLabel#panel-title {
    font-size: 14px;
    font-weight: 600;
    color: #58a6ff;
    padding: 6px 0;
    margin-bottom: 8px;
}

/* Buttons - Cleaner, less busy */
QPushButton {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 16px;
    color: #e6edf3;
    font-weight: 500;
    min-width: 70px;
}

QPushButton:hover {
    background: #30363d;
    border-color: #8b949e;
}

QPushButton:pressed {
    background: #161b22;
}

QPushButton#primary {
    background: #238636;
    border: 1px solid #238636;
    color: white;
}

QPushButton#primary:hover {
    background: #2ea043;
}

QPushButton#danger {
    background: #da3633;
    border: 1px solid #da3633;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 4px;
    background: #30363d;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    background: #58a6ff;
    border-radius: 7px;
}

QSlider::sub-page:horizontal {
    background: #58a6ff;
    border-radius: 2px;
}

/* Progress Bars */
QProgressBar {
    background: #21262d;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}

QProgressBar::chunk {
    background: #238636;
    border-radius: 3px;
}

/* Scroll Areas - Thinner scrollbars */
QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #30363d;
    border-radius: 3px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #484f58;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 6px;
}

QScrollBar::handle:horizontal {
    background: #30363d;
    border-radius: 3px;
    min-width: 20px;
}

/* Combo Boxes */
QComboBox {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    min-width: 100px;
}

QComboBox::drop-down {
    border: none;
    width: 18px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #8b949e;
}

QComboBox QAbstractItemView {
    background: #21262d;
    border: 1px solid #30363d;
    selection-background-color: #388bfd26;
}

/* Checkboxes */
QCheckBox {
    spacing: 6px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #30363d;
    background: #21262d;
}

QCheckBox::indicator:checked {
    background: #238636;
    border-color: #238636;
}

/* Tab Widget - Cleaner tabs */
QTabWidget::pane {
    border: none;
    background: transparent;
}

QTabBar::tab {
    background: transparent;
    border: none;
    padding: 10px 16px;
    margin-right: 2px;
    color: #8b949e;
}

QTabBar::tab:selected {
    color: #e6edf3;
    border-bottom: 2px solid #58a6ff;
}

QTabBar::tab:hover:!selected {
    color: #e6edf3;
}

/* Group Boxes */
QGroupBox {
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 500;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #58a6ff;
}

/* List Widgets */
QListWidget {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px;
}

QListWidget::item {
    padding: 6px 8px;
    border-radius: 4px;
    margin: 1px;
}

QListWidget::item:selected {
    background: #388bfd26;
}

QListWidget::item:hover:!selected {
    background: #21262d;
}

/* Spin Boxes */
QSpinBox, QDoubleSpinBox {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 5px 8px;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    width: 14px;
    border: none;
    background: transparent;
}

/* Line Edit */
QLineEdit {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e6edf3;
}

QLineEdit:focus {
    border-color: #58a6ff;
}

/* Status Bar */
QStatusBar {
    background: #161b22;
    border-top: 1px solid #30363d;
    color: #8b949e;
    font-size: 11px;
    padding: 4px 8px;
}

/* Menu Bar */
QMenuBar {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 4px;
}

QMenuBar::item {
    padding: 6px 12px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background: #30363d;
}

QMenu {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}

QMenu::item:selected {
    background: #388bfd26;
}

/* Tool Bar */
QToolBar {
    background: #161b22;
    border: none;
    padding: 4px 8px;
    spacing: 8px;
}

QToolButton {
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    color: #e6edf3;
}

QToolButton:hover {
    background: #30363d;
}

/* Table Widget */
QTableWidget {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    gridline-color: #21262d;
}

QTableWidget::item {
    padding: 6px;
}

QHeaderView::section {
    background: #161b22;
    border: none;
    border-bottom: 1px solid #30363d;
    padding: 8px;
    font-weight: 600;
}
"""


ACCENT_COLORS = {
    "primary": "#58a6ff",
    "success": "#238636", 
    "warning": "#d29922",
    "danger": "#da3633",
    "info": "#58a6ff",
    "audio": "#f778ba",
    "video": "#3fb950",
    "fusion": "#a371f7"
}
