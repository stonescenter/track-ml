import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.histograms import make_phi_range
import utils.transformation as utf


def delta_phi(phi1, phi2):
    result = phi1 - phi2
    while result > np.pi:
        result -= 2 * np.pi
    while result <= -np.pi:
        result += 2 * np.pi
    return result


def vdelta_phi(x1, x2):
    return np.vectorize(delta_phi)(x1, x2)


# PARAMETERS
input_path = "./real.csv"
output_path = "/data/track-ml/output/"
mean_nhits = 11.0
plt.ioff()

# Open the input file
tracks = np.genfromtxt(input_path, delimiter=",")
print(tracks.shape)

# Slice the input file in its constituent parts
indexes = tracks[:, 0]  # Particle indexes (0,1,2,...)
vertices = tracks[:, 1:4]  # Particle vertex (tx,ty,tz)
momenta = tracks[:, 4:7]  # Particle momentum (px,py,pz)
hits = tracks[:, 7:-1]  # N hits with the following information: 3D point + 3 tech. info
print(hits.shape)

global_X = np.empty(0)
global_Y = np.empty(0)
global_rho = np.empty(0)
global_eta = np.empty(0)
global_phi = np.empty(0)
global_hits = np.empty(0)
global_tech1 = np.empty(0)
global_tech2 = np.empty(0)
global_tech3 = np.empty(0)

# Loop over particles
for i in range(hits.shape[0]):
    # for i in range(100):
    the_hits = hits[i]
    reshaped_hits = the_hits.reshape(int(114 / 6), 6)
    X = reshaped_hits[:, 0]
    Y = reshaped_hits[:, 1]
    Z = reshaped_hits[:, 2]
    tech1 = reshaped_hits[:, 3]
    tech2 = reshaped_hits[:, 4]
    tech3 = reshaped_hits[:, 5]
    X = np.trim_zeros(X)
    Y = np.trim_zeros(Y)
    Z = np.trim_zeros(Z)
    tech1 = tech1[: X.size]
    tech2 = tech3[: X.size]
    tech3 = tech3[: X.size]

    assert (
        X.size == Y.size
        and X.size == Z.size
        and Y.size == Z.size
        and tech1.size == X.size
        and tech2.size == X.size
        and tech3.size == X.size
    )

    if X.size < 1:
        continue
    #  conversion for eta, phi
    rho = utf.rho_from_xy(X, Y)
    eta = utf.eta_from_xyz(X, Y, Z)
    phi = utf.phi_from_xy(X, Y)
    global_rho = np.append(global_rho, rho)
    global_eta = np.append(global_eta, eta)
    global_phi = np.append(global_phi, phi)
    global_tech1 = np.append(global_tech1, tech1)
    global_tech2 = np.append(global_tech2, tech2)
    global_tech3 = np.append(global_tech3, tech3)

    if i % 1000 == 0:
        print(i)
        print(rho.shape)
        print(eta.shape)
        print(phi.shape)

global_hits = np.column_stack(
    (global_rho, global_eta, global_phi, global_tech1, global_tech2, global_tech3)
)
print(global_hits.shape)

# Select only in a small square of (0.4 X 0.5232)
cell_hits = global_hits[
    np.where(
        (global_hits[:, 1] > -0.2)
        * (global_hits[:, 1] < 0.2)
        * (global_hits[:, 2] > -0.2616)
        * (global_hits[:, 2] < 0.2616)
    )
]

print(cell_hits.shape)

generated_nhits = np.random.poisson(mean_nhits, 1)[0]
selected_hits = cell_hits[
    np.random.choice(cell_hits.shape[0], generated_nhits, replace=False)
]
print(selected_hits.shape)
# print(selected_hits)
flat_selected_hits = selected_hits.flatten()
padded_selected_hits = np.pad(
    flat_selected_hits,
    (0, 114 - flat_selected_hits.size),
    "constant",
    constant_values=0,
)
print(padded_selected_hits.shape)
# print(padded_selected_hits)
candidate_track = np.concatenate(
    [
        np.array([-1]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        padded_selected_hits,
        np.array([0]),
    ]
)
print(candidate_track.shape)
print(candidate_track)
