"""
Person Database - Face recognition and person tracking across sessions.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class Person:
    """Represents a recognized person."""
    person_id: str
    name: str
    created_at: float
    last_seen: float
    face_encoding: Optional[list] = None
    thumbnail_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    sightings: int = 0
    total_speaking_time: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Person':
        return cls(**data)


class PersonDatabase:
    """
    Database for storing and recognizing persons.
    Uses face embeddings for recognition.
    """
    
    def __init__(self, db_path: str = "data/persons"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.persons: dict[str, Person] = {}
        self.face_encodings: dict[str, np.ndarray] = {}
        
        # Face detection
        self.face_cascade = None
        if CV2_AVAILABLE:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load existing database
        self._load_database()
        
        logger.info(f"PersonDatabase initialized: {len(self.persons)} persons")
    
    def _load_database(self):
        """Load persons from disk."""
        db_file = self.db_path / "persons.json"
        if db_file.exists():
            try:
                with open(db_file) as f:
                    data = json.load(f)
                for person_data in data.get("persons", []):
                    person = Person.from_dict(person_data)
                    self.persons[person.person_id] = person
                    
                    # Load face encoding if exists
                    enc_path = self.db_path / f"{person.person_id}_encoding.npy"
                    if enc_path.exists():
                        self.face_encodings[person.person_id] = np.load(enc_path)
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
    
    def _save_database(self):
        """Save persons to disk."""
        db_file = self.db_path / "persons.json"
        data = {
            "persons": [p.to_dict() for p in self.persons.values()],
            "updated_at": time.time()
        }
        with open(db_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_person(self, name: str, face_image: np.ndarray = None,
                   metadata: dict = None) -> Person:
        """Add a new person to the database."""
        person_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]
        
        person = Person(
            person_id=person_id,
            name=name,
            created_at=time.time(),
            last_seen=time.time(),
            metadata=metadata or {}
        )
        
        # Save face encoding if provided
        if face_image is not None and CV2_AVAILABLE:
            encoding = self._compute_face_encoding(face_image)
            if encoding is not None:
                person.face_encoding = encoding.tolist()
                self.face_encodings[person_id] = encoding
                np.save(self.db_path / f"{person_id}_encoding.npy", encoding)
                
                # Save thumbnail
                thumbnail_path = self.db_path / f"{person_id}_thumb.jpg"
                cv2.imwrite(str(thumbnail_path), face_image)
                person.thumbnail_path = str(thumbnail_path)
        
        self.persons[person_id] = person
        self._save_database()
        
        logger.info(f"Added person: {name} ({person_id})")
        return person
    
    def update_person(self, person_id: str, **kwargs) -> Optional[Person]:
        """Update person attributes."""
        if person_id not in self.persons:
            return None
        
        person = self.persons[person_id]
        for key, value in kwargs.items():
            if hasattr(person, key):
                setattr(person, key, value)
        
        self._save_database()
        return person
    
    def delete_person(self, person_id: str) -> bool:
        """Delete a person from the database."""
        if person_id not in self.persons:
            return False
        
        person = self.persons.pop(person_id)
        self.face_encodings.pop(person_id, None)
        
        # Remove files
        for suffix in ["_encoding.npy", "_thumb.jpg"]:
            path = self.db_path / f"{person_id}{suffix}"
            if path.exists():
                path.unlink()
        
        self._save_database()
        logger.info(f"Deleted person: {person.name} ({person_id})")
        return True
    
    def get_person(self, person_id: str) -> Optional[Person]:
        """Get person by ID."""
        return self.persons.get(person_id)
    
    def get_all_persons(self) -> list[Person]:
        """Get all persons."""
        return list(self.persons.values())
    
    def recognize_face(self, face_image: np.ndarray, 
                       threshold: float = 0.6) -> Optional[tuple[Person, float]]:
        """
        Recognize a face from the database.
        
        Returns: (Person, confidence) or None if not recognized
        """
        if not CV2_AVAILABLE or face_image is None:
            return None
        
        encoding = self._compute_face_encoding(face_image)
        if encoding is None:
            return None
        
        best_match = None
        best_score = 0.0
        
        for person_id, stored_encoding in self.face_encodings.items():
            # Compute cosine similarity
            score = self._cosine_similarity(encoding, stored_encoding)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = self.persons.get(person_id)
        
        if best_match:
            # Update last seen
            best_match.last_seen = time.time()
            best_match.sightings += 1
            self._save_database()
            
            return (best_match, best_score)
        
        return None
    
    def _compute_face_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Compute a simple face encoding using image histograms."""
        if not CV2_AVAILABLE:
            return None
        
        try:
            # Resize to standard size
            face = cv2.resize(image, (128, 128))
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Compute histogram as simple encoding
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            
            # Add LBP-like features
            lbp_features = self._compute_lbp_histogram(gray)
            
            encoding = np.concatenate([hist, lbp_features])
            return encoding
            
        except Exception as e:
            logger.error(f"Face encoding failed: {e}")
            return None
    
    def _compute_lbp_histogram(self, gray: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern histogram."""
        # Simple LBP implementation
        h, w = gray.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 7
                code |= (gray[i-1, j] > center) << 6
                code |= (gray[i-1, j+1] > center) << 5
                code |= (gray[i, j+1] > center) << 4
                code |= (gray[i+1, j+1] > center) << 3
                code |= (gray[i+1, j] > center) << 2
                code |= (gray[i+1, j-1] > center) << 1
                code |= (gray[i, j-1] > center) << 0
                lbp[i-1, j-1] = code
        
        hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-7)
        return hist
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return dot / (norm + 1e-7)
    
    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces in a frame."""
        if not CV2_AVAILABLE or self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def record_speaking_time(self, person_id: str, duration: float):
        """Record speaking time for a person."""
        if person_id in self.persons:
            self.persons[person_id].total_speaking_time += duration
            self._save_database()
