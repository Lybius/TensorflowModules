import numpy as np


def window_stack(a, stepsize=1, window_length=3):
    H=tuple(a[i:1+i-window_length or None:stepsize] for i in range(0,window_length))
    return np.vstack(H)