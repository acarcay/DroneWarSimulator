import numpy as np

class Sensor:
    """Base sensor model applying bias, Gaussian noise and random dropout."""
    def __init__(self, bias=0.0, noise=0.0, dropout=0.0):
        self.bias = np.array(bias, dtype=float)
        self.noise = float(noise)
        self.dropout = float(dropout)

    def measure(self, true_val):
        if np.random.rand() < self.dropout:
            return None
        true_val = np.asarray(true_val, dtype=float)
        noise = np.random.randn(*true_val.shape) * self.noise
        return true_val + self.bias + noise


class GPSSensor(Sensor):
    """GPS sensor measuring position."""
    pass


class IMUSensor(Sensor):
    """IMU sensor measuring velocity or acceleration."""
    pass


class LiDARSensor(Sensor):
    """LiDAR sensor measuring obstacle influence vectors."""
    pass
