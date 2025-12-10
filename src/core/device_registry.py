"""
Device Registry for the Acoustic-Visual Fusion System.

Provides persistent device identification and configuration storage.
Ensures device IDs remain consistent across sessions.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict, field
import yaml
from loguru import logger


@dataclass
class DeviceInfo:
    """Persistent device information."""
    device_id: str
    device_type: str  # "camera" or "microphone"
    hardware_id: str  # System-level identifier
    friendly_name: str
    manufacturer: str = ""
    serial_number: str = ""
    
    # Physical calibration data
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[list[float]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    
    # Camera-specific calibration
    intrinsic_matrix: Optional[list[list[float]]] = None
    distortion_coeffs: Optional[list[float]] = None
    
    # Status tracking
    first_seen: str = ""
    last_seen: str = ""
    total_usage_hours: float = 0.0
    
    # User notes
    notes: str = ""
    
    def __post_init__(self):
        if not self.first_seen:
            self.first_seen = datetime.now().isoformat()
        self.last_seen = datetime.now().isoformat()


class DeviceRegistry:
    """
    Manages persistent device identification and configuration.
    
    Features:
    - Consistent device IDs across sessions
    - Calibration data storage
    - Device usage tracking
    - Configuration persistence
    """
    
    def __init__(self, registry_path: Optional[str | Path] = None):
        """
        Initialize the device registry.
        
        Args:
            registry_path: Path to the registry file. Defaults to data/device_registry.yaml
        """
        if registry_path is None:
            registry_path = Path(__file__).parent.parent.parent / "data" / "device_registry.yaml"
        
        self.registry_path = Path(registry_path)
        self.devices: dict[str, DeviceInfo] = {}
        
        self._load_registry()
        logger.info(f"Device registry loaded from {self.registry_path}")
    
    def _load_registry(self) -> None:
        """Load registry from file."""
        if not self.registry_path.exists():
            logger.info("No existing registry found, starting fresh")
            return
        
        try:
            with open(self.registry_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            
            for device_id, device_data in data.get("devices", {}).items():
                self.devices[device_id] = DeviceInfo(**device_data)
            
            logger.info(f"Loaded {len(self.devices)} devices from registry")
            
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "devices": {
                    device_id: asdict(device)
                    for device_id, device in self.devices.items()
                }
            }
            
            with open(self.registry_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def generate_device_id(
        self,
        device_type: str,
        hardware_id: str,
        name: str
    ) -> str:
        """
        Generate a consistent device ID from hardware identifiers.
        
        Args:
            device_type: "camera" or "microphone"
            hardware_id: System-level device identifier
            name: Device name
            
        Returns:
            Unique device ID that persists across sessions.
        """
        # Create hash from stable identifiers
        id_string = f"{device_type}:{hardware_id}:{name}"
        hash_value = hashlib.md5(id_string.encode()).hexdigest()[:8]
        
        return f"{device_type}_{hash_value}"
    
    def register_device(
        self,
        device_type: str,
        hardware_id: str,
        name: str,
        **kwargs
    ) -> DeviceInfo:
        """
        Register a device or update existing registration.
        
        Args:
            device_type: "camera" or "microphone"
            hardware_id: System-level device identifier
            name: Device name
            **kwargs: Additional device properties
            
        Returns:
            Device information object.
        """
        device_id = self.generate_device_id(device_type, hardware_id, name)
        
        if device_id in self.devices:
            # Update existing device
            device = self.devices[device_id]
            device.last_seen = datetime.now().isoformat()
            
            # Update any provided properties
            for key, value in kwargs.items():
                if hasattr(device, key):
                    setattr(device, key, value)
        else:
            # Create new device
            device = DeviceInfo(
                device_id=device_id,
                device_type=device_type,
                hardware_id=hardware_id,
                friendly_name=name,
                **kwargs
            )
            self.devices[device_id] = device
            logger.info(f"Registered new device: {device_id} ({name})")
        
        self._save_registry()
        return device
    
    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """Get device information by ID."""
        return self.devices.get(device_id)
    
    def get_devices_by_type(self, device_type: str) -> list[DeviceInfo]:
        """Get all devices of a specific type."""
        return [
            device for device in self.devices.values()
            if device.device_type == device_type
        ]
    
    def update_calibration(
        self,
        device_id: str,
        position: Optional[list[float]] = None,
        orientation: Optional[list[list[float]]] = None,
        intrinsic_matrix: Optional[list[list[float]]] = None,
        distortion_coeffs: Optional[list[float]] = None
    ) -> bool:
        """
        Update device calibration data.
        
        Args:
            device_id: Device ID
            position: 3D position [x, y, z]
            orientation: 3x3 rotation matrix
            intrinsic_matrix: 3x3 camera intrinsic matrix
            distortion_coeffs: Distortion coefficients
            
        Returns:
            True if update successful.
        """
        if device_id not in self.devices:
            logger.error(f"Device {device_id} not found")
            return False
        
        device = self.devices[device_id]
        
        if position is not None:
            device.position = position
        if orientation is not None:
            device.orientation = orientation
        if intrinsic_matrix is not None:
            device.intrinsic_matrix = intrinsic_matrix
        if distortion_coeffs is not None:
            device.distortion_coeffs = distortion_coeffs
        
        device.last_seen = datetime.now().isoformat()
        self._save_registry()
        
        logger.info(f"Updated calibration for device {device_id}")
        return True
    
    def update_usage(self, device_id: str, hours: float) -> None:
        """Update device usage hours."""
        if device_id in self.devices:
            self.devices[device_id].total_usage_hours += hours
            self.devices[device_id].last_seen = datetime.now().isoformat()
            self._save_registry()
    
    def set_notes(self, device_id: str, notes: str) -> None:
        """Set user notes for a device."""
        if device_id in self.devices:
            self.devices[device_id].notes = notes
            self._save_registry()
    
    def remove_device(self, device_id: str) -> bool:
        """Remove a device from the registry."""
        if device_id in self.devices:
            del self.devices[device_id]
            self._save_registry()
            logger.info(f"Removed device {device_id} from registry")
            return True
        return False
    
    def export_calibration(self, output_path: str | Path) -> None:
        """
        Export all calibration data to a file.
        
        Args:
            output_path: Path for the export file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        calibration_data = {
            "exported_at": datetime.now().isoformat(),
            "cameras": {},
            "microphones": {}
        }
        
        for device in self.devices.values():
            device_data = {
                "id": device.device_id,
                "name": device.friendly_name,
                "position": device.position,
                "orientation": device.orientation
            }
            
            if device.device_type == "camera":
                device_data["intrinsic_matrix"] = device.intrinsic_matrix
                device_data["distortion_coeffs"] = device.distortion_coeffs
                calibration_data["cameras"][device.device_id] = device_data
            else:
                calibration_data["microphones"][device.device_id] = device_data
        
        with open(output_path, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False)
        
        logger.info(f"Exported calibration to {output_path}")
    
    def import_calibration(self, input_path: str | Path) -> bool:
        """
        Import calibration data from a file.
        
        Args:
            input_path: Path to the calibration file.
            
        Returns:
            True if import successful.
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"Calibration file not found: {input_path}")
            return False
        
        try:
            with open(input_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Update cameras
            for device_id, cal_data in data.get("cameras", {}).items():
                if device_id in self.devices:
                    self.update_calibration(
                        device_id,
                        position=cal_data.get("position"),
                        orientation=cal_data.get("orientation"),
                        intrinsic_matrix=cal_data.get("intrinsic_matrix"),
                        distortion_coeffs=cal_data.get("distortion_coeffs")
                    )
            
            # Update microphones
            for device_id, cal_data in data.get("microphones", {}).items():
                if device_id in self.devices:
                    self.update_calibration(
                        device_id,
                        position=cal_data.get("position"),
                        orientation=cal_data.get("orientation")
                    )
            
            logger.info(f"Imported calibration from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing calibration: {e}")
            return False
    
    def get_summary(self) -> dict:
        """Get summary of registered devices."""
        cameras = self.get_devices_by_type("camera")
        microphones = self.get_devices_by_type("microphone")
        
        return {
            "total_devices": len(self.devices),
            "cameras": {
                "count": len(cameras),
                "calibrated": sum(
                    1 for c in cameras
                    if c.intrinsic_matrix is not None
                )
            },
            "microphones": {
                "count": len(microphones),
                "positioned": sum(
                    1 for m in microphones
                    if any(p != 0 for p in m.position)
                )
            }
        }


if __name__ == "__main__":
    # Test device registry
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    console.print("\n[bold blue]Device Registry Test[/bold blue]\n")
    
    registry = DeviceRegistry()
    
    # Register some test devices
    registry.register_device(
        "camera",
        "usb_vid_1234_pid_5678",
        "USB Webcam",
        manufacturer="Logitech"
    )
    
    registry.register_device(
        "microphone",
        "usb_audio_device_1",
        "USB Microphone",
        manufacturer="Blue"
    )
    
    # Display devices
    table = Table(title="Registered Devices")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Name")
    table.add_column("First Seen")
    table.add_column("Position")
    
    for device in registry.devices.values():
        table.add_row(
            device.device_id,
            device.device_type,
            device.friendly_name,
            device.first_seen[:10],
            str(device.position)
        )
    
    console.print(table)
    
    summary = registry.get_summary()
    console.print(f"\n[green]Summary: {summary}[/green]")
