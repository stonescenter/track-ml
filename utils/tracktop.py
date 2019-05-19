from scipy import special
import matplotlib.pyplot as plt
import random
import numpy as np

default_err = 0.03
default_err_fad = 0.174
default_len_err = 500
default_center = 1


def scale_range(input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


def err_lin(position, err=default_err, len_err=default_len_err):
    err_array = np.linspace(default_center - err, default_center + err, len_err)
    value_err = position * random.choice(err_array)
    return value_err


def err_normal(position, err=default_err, len_err=default_len_err):
    err_array = np.random.normal(loc=default_center, scale=err, size=len_err)
    value_err = position * random.choice(err_array)
    return value_err


def err_erf(position, err=default_err, len_err=default_len_err):
    int_array = np.linspace(-err * 100, err * 100, len_err)
    err_array = special.erfc(x)
    scaled_err_array = scale_range(
        err_array, default_center - err, default_center + err
    )
    value_err = position * random.choice(scaled_err_array)
    return value_err


def err_faddeeva(position, err=default_err_fad, len_err=default_len_err):
    err_array = np.linspace(-err, err, len_err)
    faddeev_array = special.wofz(err_array)
    factor = [1, -1]
    value_err = round(
        position * (random.choice(faddeev_array.real) ** random.choice(factor)), 8
    )
    return value_err


# Flatten hits, pad the line to make up the original length,
# add back the index, vertex, momentum in the front
# and the 0 (for "False") in the back
def make_random_track(selected_hits, total_hits):
    flat_selected_hits = selected_hits.flatten()
    padded_selected_hits = np.pad(
        flat_selected_hits,
        (0, total_hits * 6 - flat_selected_hits.size),
        "constant",
        constant_values=0,
    )

    candidate_track = np.concatenate(
        [
            np.array([-1]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            padded_selected_hits,
            np.array([0]),
        ]
    )
    return candidate_track
