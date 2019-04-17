from scipy import special
import matplotlib.pyplot as plt
import random
import numpy as np

def err_gauss(position, err = 0.1, len_err = 50):
    err_array = np.linspace(1 - err, 1 + err, len_err)
    value_err = position * random.choice(err_array)
    return value_err
