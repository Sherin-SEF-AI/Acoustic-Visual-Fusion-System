"""
Sound Source Localization Module for the Acoustic-Visual Fusion System.

Implements GCC-PHAT for TDOA estimation and 3D multilateration for
sound source position estimation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy import signal
from scipy.optimize import least_squares
from loguru import logger


@dataclass
class LocalizationResult:
    """Result of sound source localization."""
    position: np.ndarray  # 3D position (x, y, z) in meters
    timestamp: float
    confidence: float  # 0.0 to 1.0
    uncertainty: np.ndarray  # 3x3 covariance matrix
    tdoa_values: np.ndarray  # TDOA between mic pairs
    tdoa_confidence: np.ndarray  # Confidence for each TDOA
    
    @property
    def uncertainty_ellipsoid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get uncertainty ellipsoid axes and radii.
        
        Returns:
            Tuple of (axes as 3x3 matrix, radii as 3-vector)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.uncertainty)
        radii = np.sqrt(np.maximum(eigenvalues, 0))  # Standard deviations
        return eigenvectors, radii
    
    def __repr__(self) -> str:
        return (
            f"LocalizationResult(pos=[{self.position[0]:.2f}, "
            f"{self.position[1]:.2f}, {self.position[2]:.2f}]m, "
            f"conf={self.confidence:.2f})"
        )


class GCCPHATLocalizer:
    """
    Generalized Cross-Correlation with Phase Transform (GCC-PHAT) localizer.
    
    Computes time-difference-of-arrival between microphone pairs and
    estimates 3D source position through multilateration.
    """
    
    def __init__(
        self,
        microphone_positions: np.ndarray,
        sample_rate: int = 48000,
        speed_of_sound: float = 343.0,
        fft_size: int = 4096,
        interpolate: bool = True,
        max_tdoa_error: float = 0.001  # seconds
    ):
        """
        Initialize GCC-PHAT localizer.
        
        Args:
            microphone_positions: Nx3 array of microphone positions in meters
            sample_rate: Audio sample rate in Hz
            speed_of_sound: Speed of sound in m/s
            fft_size: FFT size for cross-correlation
            interpolate: Use parabolic interpolation for sub-sample accuracy
            max_tdoa_error: Maximum expected TDOA error in seconds
        """
        self.mic_positions = np.asarray(microphone_positions)
        self.num_mics = len(self.mic_positions)
        self.sample_rate = sample_rate
        self.speed_of_sound = speed_of_sound
        self.fft_size = fft_size
        self.interpolate = interpolate
        self.max_tdoa_error = max_tdoa_error
        
        # Precompute mic pair information
        self._compute_mic_pairs()
        
        logger.info(
            f"GCCPHATLocalizer initialized: {self.num_mics} mics, "
            f"{len(self.mic_pairs)} pairs, c={speed_of_sound} m/s"
        )
    
    def _compute_mic_pairs(self) -> None:
        """Compute microphone pair indices and distances."""
        self.mic_pairs = []
        self.pair_distances = []
        self.max_tdoas = []
        
        for i in range(self.num_mics):
            for j in range(i + 1, self.num_mics):
                self.mic_pairs.append((i, j))
                dist = np.linalg.norm(
                    self.mic_positions[i] - self.mic_positions[j]
                )
                self.pair_distances.append(dist)
                self.max_tdoas.append(dist / self.speed_of_sound)
        
        self.mic_pairs = np.array(self.mic_pairs)
        self.pair_distances = np.array(self.pair_distances)
        self.max_tdoas = np.array(self.max_tdoas)
    
    def gcc_phat(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray,
        max_delay_samples: Optional[int] = None
    ) -> tuple[float, float]:
        """
        Compute GCC-PHAT between two signals.
        
        Args:
            sig1: First signal
            sig2: Second signal
            max_delay_samples: Maximum delay to search (samples)
            
        Returns:
            Tuple of (delay in samples, peak value/confidence)
        """
        n = max(len(sig1), len(sig2))
        n_fft = max(self.fft_size, 2 ** int(np.ceil(np.log2(2 * n - 1))))
        
        # Compute cross-power spectrum
        SIG1 = np.fft.rfft(sig1, n_fft)
        SIG2 = np.fft.rfft(sig2, n_fft)
        
        # Cross-power spectrum with PHAT weighting
        R = SIG1 * np.conj(SIG2)
        R_magnitude = np.abs(R)
        R_magnitude[R_magnitude < 1e-10] = 1e-10  # Avoid division by zero
        R_phat = R / R_magnitude
        
        # Inverse FFT
        cc = np.fft.irfft(R_phat, n_fft)
        
        # Shift to put zero lag in center
        cc = np.fft.fftshift(cc)
        center = len(cc) // 2
        
        # Search range
        if max_delay_samples is None:
            max_delay_samples = n // 2
        
        search_start = center - max_delay_samples
        search_end = center + max_delay_samples + 1
        search_start = max(0, search_start)
        search_end = min(len(cc), search_end)
        
        # Find peak
        search_region = cc[search_start:search_end]
        peak_idx_local = np.argmax(np.abs(search_region))
        peak_idx = search_start + peak_idx_local
        peak_value = np.abs(cc[peak_idx])
        
        # Convert to sample delay
        delay_samples = peak_idx - center
        
        # Parabolic interpolation for sub-sample accuracy
        if self.interpolate and 0 < peak_idx_local < len(search_region) - 1:
            alpha = np.abs(search_region[peak_idx_local - 1])
            beta = np.abs(search_region[peak_idx_local])
            gamma = np.abs(search_region[peak_idx_local + 1])
            
            if 2 * beta - alpha - gamma != 0:
                delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                delay_samples += delta
                peak_value = beta - 0.25 * (alpha - gamma) * delta
        
        return delay_samples, peak_value
    
    def compute_tdoa(
        self,
        audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute TDOA for all microphone pairs.
        
        Args:
            audio: Multi-channel audio (samples x channels)
            
        Returns:
            Tuple of (TDOA array in seconds, confidence array)
        """
        if audio.shape[1] != self.num_mics:
            raise ValueError(
                f"Expected {self.num_mics} channels, got {audio.shape[1]}"
            )
        
        tdoas = np.zeros(len(self.mic_pairs))
        confidences = np.zeros(len(self.mic_pairs))
        
        for k, (i, j) in enumerate(self.mic_pairs):
            # Max expected delay for this pair
            max_delay_samples = int(
                self.max_tdoas[k] * self.sample_rate * 1.5
            )
            
            delay_samples, confidence = self.gcc_phat(
                audio[:, i],
                audio[:, j],
                max_delay_samples=max_delay_samples
            )
            
            # Convert to seconds
            tdoas[k] = delay_samples / self.sample_rate
            confidences[k] = confidence
        
        return tdoas, confidences
    
    def multilaterate(
        self,
        tdoas: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Estimate 3D position from TDOA measurements.
        
        Uses weighted nonlinear least squares optimization.
        
        Args:
            tdoas: TDOA values in seconds for each mic pair
            confidences: Confidence weights for each TDOA
            initial_guess: Initial position estimate
            
        Returns:
            Tuple of (position, overall confidence, covariance matrix)
        """
        if confidences is None:
            confidences = np.ones(len(tdoas))
        
        # Normalize confidences to use as weights
        weights = confidences / (np.sum(confidences) + 1e-10)
        
        # Initial guess: centroid of microphones
        if initial_guess is None:
            initial_guess = np.mean(self.mic_positions, axis=0)
        
        def residuals(pos):
            """Compute residuals between observed and predicted TDOA."""
            residuals = []
            
            for k, (i, j) in enumerate(self.mic_pairs):
                # Distances from source to each mic
                d_i = np.linalg.norm(pos - self.mic_positions[i])
                d_j = np.linalg.norm(pos - self.mic_positions[j])
                
                # Predicted TDOA
                predicted_tdoa = (d_i - d_j) / self.speed_of_sound
                
                # Weighted residual
                residuals.append(weights[k] * (tdoas[k] - predicted_tdoa))
            
            return np.array(residuals)
        
        # Solve using Levenberg-Marquardt
        result = least_squares(
            residuals,
            initial_guess,
            method='lm',
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=100
        )
        
        position = result.x
        
        # Estimate uncertainty from Jacobian
        try:
            # Jacobian at solution
            jac = result.jac
            # Covariance approximation
            jtj = jac.T @ jac
            jtj += np.eye(3) * 1e-10  # Regularization
            cov = np.linalg.inv(jtj)
            
            # Scale by residual variance
            residual_var = np.sum(result.fun ** 2) / max(1, len(tdoas) - 3)
            cov *= residual_var
        except:
            # Fallback: identity covariance
            cov = np.eye(3) * 0.1
        
        # Overall confidence based on residual
        residual_norm = np.linalg.norm(result.fun)
        confidence = np.exp(-residual_norm * 10)  # Heuristic mapping
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return position, confidence, cov
    
    def localize(
        self,
        audio: np.ndarray,
        timestamp: float,
        previous_position: Optional[np.ndarray] = None
    ) -> LocalizationResult:
        """
        Localize sound source from multi-channel audio.
        
        Args:
            audio: Multi-channel audio (samples x channels)
            timestamp: Timestamp of audio
            previous_position: Previous position estimate for initialization
            
        Returns:
            LocalizationResult with position and uncertainty
        """
        # Compute TDOAs
        tdoas, tdoa_confidences = self.compute_tdoa(audio)
        
        # Filter out low-confidence TDOAs
        valid_mask = tdoa_confidences > 0.1
        if np.sum(valid_mask) < 3:
            # Not enough confident measurements
            logger.warning("Insufficient confident TDOA measurements")
            return LocalizationResult(
                position=np.zeros(3),
                timestamp=timestamp,
                confidence=0.0,
                uncertainty=np.eye(3) * 10.0,
                tdoa_values=tdoas,
                tdoa_confidence=tdoa_confidences
            )
        
        # Multilaterate
        position, confidence, uncertainty = self.multilaterate(
            tdoas,
            tdoa_confidences,
            initial_guess=previous_position
        )
        
        return LocalizationResult(
            position=position,
            timestamp=timestamp,
            confidence=confidence,
            uncertainty=uncertainty,
            tdoa_values=tdoas,
            tdoa_confidence=tdoa_confidences
        )
    
    def update_speed_of_sound(self, temperature_celsius: float) -> None:
        """
        Update speed of sound based on temperature.
        
        Args:
            temperature_celsius: Air temperature in Celsius
        """
        # c = 331.3 * sqrt(1 + T/273.15) m/s
        self.speed_of_sound = 331.3 * np.sqrt(1 + temperature_celsius / 273.15)
        self._compute_mic_pairs()  # Recompute max TDOAs
        logger.info(f"Speed of sound updated to {self.speed_of_sound:.1f} m/s")


class SoundLocalizer:
    """
    High-level sound source localizer with tracking.
    
    Wraps GCC-PHAT with Kalman filtering for smooth tracking.
    """
    
    def __init__(
        self,
        microphone_positions: np.ndarray,
        sample_rate: int = 48000,
        speed_of_sound: float = 343.0,
        process_noise: float = 0.1,
        measurement_noise: float = 0.2
    ):
        """
        Initialize sound localizer with tracking.
        
        Args:
            microphone_positions: Nx3 array of mic positions
            sample_rate: Audio sample rate
            speed_of_sound: Speed of sound in m/s
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
        """
        self.gcc_phat = GCCPHATLocalizer(
            microphone_positions=microphone_positions,
            sample_rate=sample_rate,
            speed_of_sound=speed_of_sound
        )
        
        # Kalman filter state: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 10.0
        
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        self.last_timestamp = 0.0
        self.initialized = False
        
        logger.info("SoundLocalizer initialized with Kalman tracking")
    
    def _kalman_predict(self, dt: float) -> None:
        """Kalman filter prediction step."""
        # State transition matrix (constant velocity model)
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # Process noise
        Q = np.zeros((6, 6))
        Q[0, 0] = Q[1, 1] = Q[2, 2] = (self.process_noise * dt) ** 2
        Q[3, 3] = Q[4, 4] = Q[5, 5] = self.process_noise ** 2
        
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
    
    def _kalman_update(
        self,
        measurement: np.ndarray,
        measurement_noise: float
    ) -> None:
        """Kalman filter update step."""
        # Observation matrix (we observe position only)
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        
        # Measurement noise
        R = np.eye(3) * measurement_noise ** 2
        
        # Innovation
        y = measurement - H @ self.state
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update
        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ H) @ self.covariance
    
    def localize(
        self,
        audio: np.ndarray,
        timestamp: float
    ) -> LocalizationResult:
        """
        Localize sound source with tracking.
        
        Args:
            audio: Multi-channel audio
            timestamp: Timestamp
            
        Returns:
            Smoothed localization result
        """
        # Get raw localization
        raw_result = self.gcc_phat.localize(
            audio,
            timestamp,
            previous_position=self.state[:3] if self.initialized else None
        )
        
        if raw_result.confidence < 0.1:
            # Low confidence: return prediction only
            if self.initialized:
                dt = timestamp - self.last_timestamp
                self._kalman_predict(dt)
                self.last_timestamp = timestamp
            
            return LocalizationResult(
                position=self.state[:3].copy(),
                timestamp=timestamp,
                confidence=raw_result.confidence * 0.5,
                uncertainty=self.covariance[:3, :3].copy(),
                tdoa_values=raw_result.tdoa_values,
                tdoa_confidence=raw_result.tdoa_confidence
            )
        
        if not self.initialized:
            # Initialize with first measurement
            self.state[:3] = raw_result.position
            self.covariance[:3, :3] = raw_result.uncertainty
            self.initialized = True
            self.last_timestamp = timestamp
            return raw_result
        
        # Predict
        dt = timestamp - self.last_timestamp
        if dt > 0:
            self._kalman_predict(dt)
        
        # Update
        measurement_noise = self.measurement_noise * (2 - raw_result.confidence)
        self._kalman_update(raw_result.position, measurement_noise)
        
        self.last_timestamp = timestamp
        
        return LocalizationResult(
            position=self.state[:3].copy(),
            timestamp=timestamp,
            confidence=raw_result.confidence,
            uncertainty=self.covariance[:3, :3].copy(),
            tdoa_values=raw_result.tdoa_values,
            tdoa_confidence=raw_result.tdoa_confidence
        )
    
    def reset(self) -> None:
        """Reset tracking state."""
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 10.0
        self.initialized = False
        self.last_timestamp = 0.0
    
    @property
    def current_position(self) -> np.ndarray:
        """Get current estimated position."""
        return self.state[:3].copy()
    
    @property
    def current_velocity(self) -> np.ndarray:
        """Get current estimated velocity."""
        return self.state[3:].copy()


class SRPPHATLocalizer:
    """
    Steered Response Power with Phase Transform (SRP-PHAT) localizer.
    
    An alternative to GCC-PHAT that directly searches over candidate
    positions for maximum steered response.
    """
    
    def __init__(
        self,
        microphone_positions: np.ndarray,
        sample_rate: int = 48000,
        speed_of_sound: float = 343.0,
        grid_resolution: float = 0.1,
        search_bounds: tuple[np.ndarray, np.ndarray] = None
    ):
        """
        Initialize SRP-PHAT localizer.
        
        Args:
            microphone_positions: Nx3 array of microphone positions
            sample_rate: Audio sample rate
            speed_of_sound: Speed of sound in m/s
            grid_resolution: Spatial grid resolution in meters
            search_bounds: (min_corner, max_corner) of search region
        """
        self.mic_positions = np.asarray(microphone_positions)
        self.num_mics = len(self.mic_positions)
        self.sample_rate = sample_rate
        self.speed_of_sound = speed_of_sound
        self.grid_resolution = grid_resolution
        
        # Default search bounds: cube around microphone array
        if search_bounds is None:
            center = np.mean(self.mic_positions, axis=0)
            extent = np.max(np.abs(self.mic_positions - center)) + 3.0
            self.min_bound = center - extent
            self.max_bound = center + extent
        else:
            self.min_bound, self.max_bound = search_bounds
        
        # Create search grid
        self._create_search_grid()
        
        logger.info(
            f"SRPPHATLocalizer initialized: {len(self.grid_points)} search points"
        )
    
    def _create_search_grid(self) -> None:
        """Create the 3D search grid."""
        x = np.arange(self.min_bound[0], self.max_bound[0], self.grid_resolution)
        y = np.arange(self.min_bound[1], self.max_bound[1], self.grid_resolution)
        z = np.arange(self.min_bound[2], self.max_bound[2], self.grid_resolution)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        self.grid_points = np.stack([
            xx.flatten(),
            yy.flatten(),
            zz.flatten()
        ], axis=1)
    
    def compute_srp(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute SRP-PHAT for all grid points.
        
        Args:
            audio: Multi-channel audio (samples x channels)
            
        Returns:
            SRP values for each grid point
        """
        n_samples = len(audio)
        n_fft = 2 ** int(np.ceil(np.log2(n_samples)))
        
        # Compute FFT for each channel
        spectra = []
        for i in range(self.num_mics):
            S = np.fft.rfft(audio[:, i], n_fft)
            spectra.append(S)
        
        # Compute SRP for each grid point
        srp = np.zeros(len(self.grid_points))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        
        for idx, point in enumerate(self.grid_points):
            # Compute delays from source to each microphone
            delays = np.array([
                np.linalg.norm(point - self.mic_positions[i])
                for i in range(self.num_mics)
            ]) / self.speed_of_sound
            
            # Sum steered responses
            steered_sum = np.zeros(len(freqs), dtype=complex)
            for i in range(self.num_mics):
                phase_shift = np.exp(-2j * np.pi * freqs * delays[i])
                steered = spectra[i] * phase_shift
                magnitude = np.abs(steered)
                magnitude[magnitude < 1e-10] = 1e-10
                steered_sum += steered / magnitude  # PHAT weighting
            
            srp[idx] = np.sum(np.abs(steered_sum) ** 2)
        
        return srp
    
    def localize(
        self,
        audio: np.ndarray,
        timestamp: float
    ) -> LocalizationResult:
        """
        Localize sound source using SRP-PHAT.
        
        Args:
            audio: Multi-channel audio
            timestamp: Timestamp
            
        Returns:
            Localization result
        """
        srp = self.compute_srp(audio)
        
        # Find maximum
        max_idx = np.argmax(srp)
        position = self.grid_points[max_idx]
        
        # Confidence from SRP peak sharpness
        srp_normalized = srp / (np.max(srp) + 1e-10)
        confidence = 1.0 - np.mean(srp_normalized)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Estimate uncertainty from SRP distribution
        uncertainty = np.eye(3) * self.grid_resolution ** 2
        
        return LocalizationResult(
            position=position,
            timestamp=timestamp,
            confidence=confidence,
            uncertainty=uncertainty,
            tdoa_values=np.array([]),
            tdoa_confidence=np.array([])
        )


if __name__ == "__main__":
    # Test localization
    import time
    
    print("Testing Sound Localizer...")
    
    # Create test microphone array (square, 0.5m sides)
    mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0]
    ])
    
    # Create localizer
    localizer = SoundLocalizer(
        microphone_positions=mic_positions,
        sample_rate=48000,
        speed_of_sound=343.0
    )
    
    # Generate test signal (simulated source at known location)
    source_pos = np.array([1.0, 0.25, 0.5])
    sample_rate = 48000
    duration = 0.1  # seconds
    t = np.arange(int(duration * sample_rate)) / sample_rate
    
    # Generate signals with appropriate delays
    audio = np.zeros((len(t), 4))
    for i in range(4):
        dist = np.linalg.norm(source_pos - mic_positions[i])
        delay = dist / 343.0
        delay_samples = int(delay * sample_rate)
        
        # White noise burst
        if delay_samples < len(t):
            audio[delay_samples:, i] = np.random.randn(len(t) - delay_samples)
    
    # Localize
    result = localizer.localize(audio, time.time())
    
    print(f"\nTrue position: {source_pos}")
    print(f"Estimated: {result}")
    print(f"Error: {np.linalg.norm(source_pos - result.position):.3f} m")
