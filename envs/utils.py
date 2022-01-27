from collections.abc import Iterable
import numpy as np


def to_pixel(meas_cm, shift=0):

    if isinstance(meas_cm, Iterable):
        return 1.5 * 37.795 * meas_cm + np.array(shift)

    return 1.5 * 37.795 * meas_cm + shift