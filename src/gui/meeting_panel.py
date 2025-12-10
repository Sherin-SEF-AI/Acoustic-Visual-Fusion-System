"""
Meeting Analytics Panel - GUI for conversation intelligence.
"""

import time
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGridLayout, QScrollArea, QProgressBar, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont
import numpy as np
from loguru import logger


class PieChartWidget(QWidget):
    """Real-time pie chart for talk distribution."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self._data: list = []
        
    def set_data(self, data: list):
        """Set pie chart data. Each item: {name, value, color}"""
        self._data = data
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate drawing area
        size = min(self.width(), self.height()) - 40
        x = (self.width() - size) // 2
        y = (self.height() - size) // 2
        
        if not self._data:
            painter.setPen(QPen(QColor("#8b949e")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data")
            return
        
        total = sum(d.get("value", 0) for d in self._data)
        if total == 0:
            return
        
        start_angle = 0
        for item in self._data:
            value = item.get("value", 0)
            color = QColor(item.get("color", "#58a6ff"))
            
            span_angle = int((value / total) * 360 * 16)
            
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("#0d1117"), 2))
            painter.drawPie(x, y, size, size, start_angle, span_angle)
            
            start_angle += span_angle
        
        # Draw center circle (donut effect)
        inner_size = int(size * 0.5)
        inner_x = x + (size - inner_size) // 2
        inner_y = y + (size - inner_size) // 2
        painter.setBrush(QBrush(QColor("#0d1117")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(inner_x, inner_y, inner_size, inner_size)


class ParticipantCard(QFrame):
    """Card showing participant statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: #21262d;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Header with name and speaking indicator
        header = QHBoxLayout()
        self.name_label = QLabel("Participant")
        self.name_label.setStyleSheet("font-weight: 600; font-size: 14px;")
        header.addWidget(self.name_label)
        
        self.speaking_indicator = QLabel("●")
        self.speaking_indicator.setStyleSheet("color: #30363d; font-size: 16px;")
        header.addWidget(self.speaking_indicator)
        header.addStretch()
        layout.addLayout(header)
        
        # Stats row
        stats = QHBoxLayout()
        
        self.talk_time_label = QLabel("0:00")
        self.talk_time_label.setStyleSheet("color: #58a6ff; font-size: 12px;")
        stats.addWidget(QLabel("🗣️"))
        stats.addWidget(self.talk_time_label)
        stats.addSpacing(12)
        
        self.turns_label = QLabel("0 turns")
        self.turns_label.setStyleSheet("color: #8b949e; font-size: 12px;")
        stats.addWidget(self.turns_label)
        stats.addStretch()
        
        layout.addLayout(stats)
        
        # Progress bar for talk percentage
        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setStyleSheet("""
            QProgressBar {
                border: none;
                background: #30363d;
                height: 6px;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: #58a6ff;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress)
    
    def update_data(self, name: str, talk_time: float, 
                    talk_percentage: float, turns: int,
                    is_speaking: bool = False):
        """Update card with participant data."""
        self.name_label.setText(name)
        
        mins = int(talk_time // 60)
        secs = int(talk_time % 60)
        self.talk_time_label.setText(f"{mins}:{secs:02d}")
        
        self.turns_label.setText(f"{turns} turns")
        self.progress.setValue(int(talk_percentage))
        
        if is_speaking:
            self.speaking_indicator.setStyleSheet("color: #3fb950; font-size: 16px;")
        else:
            self.speaking_indicator.setStyleSheet("color: #30363d; font-size: 16px;")


class MomentumWidget(QWidget):
    """Shows conversation momentum as a gauge."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(120, 80)
        self._momentum = 0.5
        self._label = "Neutral"
    
    def set_momentum(self, score: float, label: str):
        self._momentum = max(0, min(1, score))
        self._label = label
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background arc
        center_x = self.width() // 2
        center_y = self.height() - 10
        radius = min(self.width(), self.height()) - 20
        
        # Draw background arc
        painter.setPen(QPen(QColor("#30363d"), 8))
        painter.drawArc(
            center_x - radius // 2, center_y - radius // 2,
            radius, radius,
            0 * 16, 180 * 16
        )
        
        # Draw momentum arc
        color = QColor("#238636") if self._momentum > 0.5 else QColor("#d29922")
        if self._momentum < 0.3:
            color = QColor("#da3633")
        
        painter.setPen(QPen(color, 8))
        angle = int(self._momentum * 180 * 16)
        painter.drawArc(
            center_x - radius // 2, center_y - radius // 2,
            radius, radius,
            180 * 16, -angle
        )
        
        # Label
        painter.setPen(QColor("#e6edf3"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(
            self.rect(), 
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            self._label
        )


class MeetingPanel(QFrame):
    """Meeting analytics panel with conversation intelligence."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # Update timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_display)
        self._timer.start(1000)  # Update every second
        
        logger.info("MeetingPanel initialized")
    
    def _setup_ui(self):
        self.setStyleSheet("""
            QFrame {
                background: #161b22;
                border-radius: 8px;
            }
            QLabel {
                color: #e6edf3;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("📊 Meeting Analytics")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header.addWidget(title)
        
        self.duration_label = QLabel("00:00")
        self.duration_label.setStyleSheet("color: #8b949e;")
        header.addWidget(self.duration_label)
        header.addStretch()
        
        self.balance_label = QLabel("Balance: --")
        self.balance_label.setStyleSheet("color: #58a6ff;")
        header.addWidget(self.balance_label)
        layout.addLayout(header)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Pie chart
        chart_frame = QFrame()
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        chart_label = QLabel("Talk Distribution")
        chart_label.setStyleSheet("font-weight: 500; color: #8b949e;")
        chart_layout.addWidget(chart_label)
        
        self.pie_chart = PieChartWidget()
        chart_layout.addWidget(self.pie_chart)
        
        splitter.addWidget(chart_frame)
        
        # Right: Participants
        participants_frame = QFrame()
        participants_layout = QVBoxLayout(participants_frame)
        participants_layout.setContentsMargins(0, 0, 0, 0)
        
        participants_label = QLabel("Participants")
        participants_label.setStyleSheet("font-weight: 500; color: #8b949e;")
        participants_layout.addWidget(participants_label)
        
        # Scroll area for participant cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        
        self.participants_container = QWidget()
        self.participants_layout = QVBoxLayout(self.participants_container)
        self.participants_layout.setContentsMargins(0, 0, 0, 0)
        self.participants_layout.setSpacing(8)
        self.participants_layout.addStretch()
        
        scroll.setWidget(self.participants_container)
        participants_layout.addWidget(scroll)
        
        splitter.addWidget(participants_frame)
        layout.addWidget(splitter, 1)
        
        # Bottom: Momentum and stats
        bottom = QHBoxLayout()
        
        # Momentum gauge
        momentum_frame = QFrame()
        momentum_layout = QVBoxLayout(momentum_frame)
        momentum_layout.setContentsMargins(8, 8, 8, 8)
        momentum_label = QLabel("Momentum")
        momentum_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        momentum_layout.addWidget(momentum_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.momentum_widget = MomentumWidget()
        momentum_layout.addWidget(self.momentum_widget)
        bottom.addWidget(momentum_frame)
        
        # Quick stats
        stats_frame = QFrame()
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        
        self.interruptions_label = QLabel("0")
        self.questions_label = QLabel("0")
        self.silences_label = QLabel("0")
        
        stats_layout.addWidget(QLabel("Interruptions"), 0, 0)
        stats_layout.addWidget(self.interruptions_label, 0, 1)
        stats_layout.addWidget(QLabel("Questions"), 1, 0)
        stats_layout.addWidget(self.questions_label, 1, 1)
        stats_layout.addWidget(QLabel("Awkward Silences"), 2, 0)
        stats_layout.addWidget(self.silences_label, 2, 1)
        
        bottom.addWidget(stats_frame, 1)
        layout.addLayout(bottom)
        
        # Participant cards cache
        self._participant_cards: dict = {}
    
    def update_data(self, data: dict):
        """Update panel with meeting analytics data."""
        # Duration
        duration = data.get("meeting_duration", 0)
        mins = int(duration // 60)
        secs = int(duration % 60)
        self.duration_label.setText(f"{mins:02d}:{secs:02d}")
        
        # Balance
        balance = data.get("balance_score", 0)
        self.balance_label.setText(f"Balance: {balance:.0%}")
        if balance < 0.4:
            self.balance_label.setStyleSheet("color: #da3633;")
        elif balance < 0.6:
            self.balance_label.setStyleSheet("color: #d29922;")
        else:
            self.balance_label.setStyleSheet("color: #3fb950;")
        
        # Pie chart
        pie_data = []
        colors = ["#58a6ff", "#3fb950", "#f778ba", "#d29922", "#a371f7"]
        participants = data.get("participants", {})
        
        for i, (pid, pdata) in enumerate(participants.items()):
            pie_data.append({
                "name": pdata.get("name", pid),
                "value": pdata.get("talk_percentage", 0),
                "color": colors[i % len(colors)]
            })
        
        self.pie_chart.set_data(pie_data)
        
        # Update participant cards
        for pid, pdata in participants.items():
            if pid not in self._participant_cards:
                card = ParticipantCard()
                self._participant_cards[pid] = card
                self.participants_layout.insertWidget(
                    self.participants_layout.count() - 1, card
                )
            
            card = self._participant_cards[pid]
            card.update_data(
                name=pdata.get("name", pid),
                talk_time=pdata.get("talk_time", 0),
                talk_percentage=pdata.get("talk_percentage", 0),
                turns=pdata.get("turns", 0),
                is_speaking=pdata.get("is_speaking", False)
            )
    
    def update_momentum(self, score: float, label: str):
        """Update momentum display."""
        self.momentum_widget.set_momentum(score, label)
    
    def update_stats(self, interruptions: int, questions: int, silences: int):
        """Update quick stats."""
        self.interruptions_label.setText(str(interruptions))
        self.questions_label.setText(str(questions))
        self.silences_label.setText(str(silences))
    
    def _update_display(self):
        """Periodic display update."""
        pass  # Data comes from external updates
