"""
Relationship Graph - Visualize interaction patterns between participants.
"""

import math
import time
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath
import numpy as np
from loguru import logger


class RelationshipGraphWidget(QWidget):
    """
    Interactive graph showing relationships between participants.
    
    Nodes = participants, edges = interactions.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        
        # Graph data
        self._nodes: dict = {}  # id -> {name, x, y, size, color, active}
        self._edges: dict = {}  # (from, to) -> {weight, type, color}
        
        # Layout parameters
        self._center = QPointF(150, 150)
        self._radius = 100
        
        # Animation
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._animate)
        self._animation_timer.start(50)
        
        # Interaction
        self._hovered_node = None
        self.setMouseTracking(True)
        
        logger.info("RelationshipGraphWidget initialized")
    
    def add_node(self, node_id: str, name: str = "", 
                 color: str = "#58a6ff"):
        """Add a node to the graph."""
        if node_id not in self._nodes:
            # Position in a circle
            angle = len(self._nodes) * (2 * math.pi / max(8, len(self._nodes) + 1))
            x = self._center.x() + self._radius * math.cos(angle)
            y = self._center.y() + self._radius * math.sin(angle)
            
            self._nodes[node_id] = {
                "name": name or node_id,
                "x": x,
                "y": y,
                "vx": 0,
                "vy": 0,
                "size": 20,
                "color": color,
                "active": False,
                "talk_time": 0
            }
            self._reposition_nodes()
    
    def update_node(self, node_id: str, 
                    active: bool = False,
                    talk_time: float = 0,
                    size: float = None):
        """Update node properties."""
        if node_id in self._nodes:
            self._nodes[node_id]["active"] = active
            self._nodes[node_id]["talk_time"] = talk_time
            if size:
                self._nodes[node_id]["size"] = size
    
    def add_edge(self, from_id: str, to_id: str, 
                 weight: float = 1.0,
                 edge_type: str = "interaction"):
        """Add or update an edge."""
        key = (from_id, to_id) if from_id < to_id else (to_id, from_id)
        
        if key not in self._edges:
            self._edges[key] = {
                "weight": 0,
                "type": edge_type,
                "color": "#30363d"
            }
        
        # Accumulate weight
        self._edges[key]["weight"] += weight
        
        # Update color based on type
        if edge_type == "interruption":
            self._edges[key]["color"] = "#da3633"
        elif edge_type == "question":
            self._edges[key]["color"] = "#58a6ff"
        elif edge_type == "response":
            self._edges[key]["color"] = "#3fb950"
    
    def record_interaction(self, from_id: str, to_id: str,
                           interaction_type: str = "speech"):
        """Record an interaction between two participants."""
        self.add_node(from_id)
        self.add_node(to_id)
        self.add_edge(from_id, to_id, weight=0.1, edge_type=interaction_type)
        self.update()
    
    def _reposition_nodes(self):
        """Reposition nodes in a circle."""
        n = len(self._nodes)
        if n == 0:
            return
        
        self._center = QPointF(self.width() / 2, self.height() / 2)
        self._radius = min(self.width(), self.height()) / 2 - 40
        
        for i, (node_id, node) in enumerate(self._nodes.items()):
            angle = i * (2 * math.pi / n) - math.pi / 2
            node["x"] = self._center.x() + self._radius * math.cos(angle)
            node["y"] = self._center.y() + self._radius * math.sin(angle)
    
    def _animate(self):
        """Animation step - subtle movement for active nodes."""
        for node in self._nodes.values():
            if node["active"]:
                # Pulse effect
                node["size"] = 20 + 5 * math.sin(time.time() * 3)
        self.update()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_nodes()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw edges
        for (from_id, to_id), edge in self._edges.items():
            if from_id in self._nodes and to_id in self._nodes:
                from_node = self._nodes[from_id]
                to_node = self._nodes[to_id]
                
                # Edge thickness based on weight
                thickness = min(1 + edge["weight"] * 2, 8)
                
                pen = QPen(QColor(edge["color"]))
                pen.setWidth(int(thickness))
                painter.setPen(pen)
                
                painter.drawLine(
                    int(from_node["x"]), int(from_node["y"]),
                    int(to_node["x"]), int(to_node["y"])
                )
        
        # Draw nodes
        for node_id, node in self._nodes.items():
            x, y = node["x"], node["y"]
            size = node["size"]
            
            # Node circle
            color = QColor(node["color"])
            if node["active"]:
                color = color.lighter(130)
            
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("#e6edf3"), 2))
            painter.drawEllipse(int(x - size/2), int(y - size/2), 
                              int(size), int(size))
            
            # Label
            painter.setPen(QColor("#e6edf3"))
            painter.setFont(QFont("Segoe UI", 9))
            
            label = node["name"][:10]
            text_rect = painter.fontMetrics().boundingRect(label)
            painter.drawText(
                int(x - text_rect.width()/2),
                int(y + size/2 + 15),
                label
            )
    
    def mouseMoveEvent(self, event):
        # Check for hover
        pos = event.pos()
        self._hovered_node = None
        
        for node_id, node in self._nodes.items():
            dist = math.sqrt((pos.x() - node["x"])**2 + (pos.y() - node["y"])**2)
            if dist < node["size"]:
                self._hovered_node = node_id
                break
        
        self.update()
    
    def clear(self):
        """Clear all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()
        self.update()


class RelationshipPanel(QFrame):
    """Panel containing the relationship graph."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        self.setStyleSheet("""
            QFrame {
                background: #161b22;
                border-radius: 8px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header = QLabel("🔗 Interaction Graph")
        header.setStyleSheet("font-size: 14px; font-weight: 600; color: #e6edf3;")
        layout.addWidget(header)
        
        # Graph widget
        self.graph = RelationshipGraphWidget()
        layout.addWidget(self.graph, 1)
        
        # Legend
        legend = QLabel("● Speaking  ━ Interaction  ━ Interruption")
        legend.setStyleSheet("color: #8b949e; font-size: 10px;")
        layout.addWidget(legend)
    
    def add_participant(self, participant_id: str, name: str = ""):
        """Add a participant to the graph."""
        self.graph.add_node(participant_id, name)
    
    def update_participant(self, participant_id: str, 
                           is_speaking: bool = False,
                           talk_time: float = 0):
        """Update participant state."""
        self.graph.update_node(participant_id, active=is_speaking, 
                               talk_time=talk_time)
    
    def record_interaction(self, from_id: str, to_id: str,
                           interaction_type: str = "speech"):
        """Record an interaction."""
        self.graph.record_interaction(from_id, to_id, interaction_type)
