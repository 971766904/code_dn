from jddb.processor import Signal, BaseProcessor
import numpy as np
from copy import deepcopy


class NormalizedDensity(BaseProcessor):

    def __init__(self, a):
        super().__init__()
        self.a = a

    def transform(self, dens_signal: Signal, ip_signal: Signal) -> Signal:
        """input signals within the same samplerate and lenth to calculate the normalized density value

        Args:
            a: small radius. unit: m
            dens_signal: The signal to be normalized. magnitude: 1e19, unit: cm^-3
            ip_signal: Plasma current. unit: kA

        Returns: Signal: The normalized density signal.

        """
        resampled_attributes = deepcopy(ip_signal.attributes)
        new_data = (dens_signal.data * np.pi * self.a) / (ip_signal.data * 2)

        return Signal(data=new_data, attributes=resampled_attributes)


class LimiterSecurityFactor(BaseProcessor):

    def __init__(self, a, R):
        """
        Args:
            a: short radius. unit: m
            R: large radius. unit: m
        """
        super().__init__()
        self.a = a
        self.R = R

    def transform(self, bt_signal: Signal, ip_signal: Signal) -> Signal:
        """input signals within the same samplerate and lenth to calculate the normalized density value

        Args:
            bt_signal: toroidal magnetic induction. unit: T
            ip_signal: plasma current. unit: kA

        Returns: Signal: The boundary security factor signal.

        """
        resampled_attributes = deepcopy(ip_signal.attributes)
        new_data = (bt_signal.data * 2 * np.pi * (self.a ** 2)) / (ip_signal.data * self.R)

        return Signal(data=new_data, attributes=resampled_attributes)


class FiltterSecurityFactor(BaseProcessor):

    def __init__(self, a, b, R):
        """
        Args:
            a: short radius. unit: m
            b: long radius. unit: m
            R: large radius. unit: m

        """
        super().__init__()
        self.a = a
        self.b = b
        self.R = R

    def transform(self, bt_signal: Signal, ip_signal: Signal) -> Signal:
        """input signals within the same samplerate and lenth to calculate the normalized density value

        Args:
            bt_signal: toroidal magnetic induction. unit: T
            ip_signal: plasma current. unit: kA

        Returns: Signal: The boundary security factor signal.

        """
        resampled_attributes = deepcopy(ip_signal.attributes)
        c = (1 + (self.a / self.b) ** 2) * (1 + 2/3 * (self.a / self.R) ** 2)
        new_data = c * (bt_signal.data * 5 * (self.a ** 2)) / (ip_signal.data * 2 * self.R)

        return Signal(data=new_data, attributes=resampled_attributes)
