import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.histograms import make_eta_range, make_phi_range
import utils.transformation as utf
import utils.tracktop as tracktop

# PARAMETERS
input_path = "./tracksReais_04_06_2019"
output_path = "."  # "/data/track-ml/output/"
output_name = output_path + "foo.csv"
extra_track_info = 7
max_num_hits = 28
hit_size = 6
mean_num_hits = 12.0
num_tracks_to_generate = 10000

# Open the input file
tracks = np.genfromtxt(input_path, delimiter=",")
print("Original file", tracks.shape)

# Separate the input tracks in different parts
indexes, vertices, momenta, hits = utf.parts_from_tracks(tracks)
print("Hits", hits.shape)

# Reshape the hits to make a collection of global hits
global_hits = utf.make_hits_rhoetaphi_collection(hits, max_num_hits, hit_size)
print("Global hits", global_hits.shape)

# Select hits only in a small square of (0.4 X 0.5232)
# First make the cells
eta_range = make_eta_range(21, -4.2, 4.2)
phi_range = make_phi_range(13)
print("Eta range", eta_range)
print("Phi range", phi_range)

cell_indexes = list()
cell_edges = dict()
cell_number = 0
for low_eta_index in range(len(eta_range) - 1):
    for low_phi_index in range(len(phi_range) - 1):
        cell_edges[cell_number] = (
            eta_range[low_eta_index],
            eta_range[low_eta_index + 1],
            phi_range[low_phi_index],
            phi_range[low_phi_index + 1],
        )
        cell_indexes.append(cell_number)
        cell_number = cell_number + 1

# print(cell_edges)
output_tracks = np.empty((0, extra_track_info + hit_size * max_num_hits))
for track_number in range(num_tracks_to_generate):
    if track_number % 100 == 0:
        print("Track number", track_number)

    # First, choose where in the tracker we are going to generate a track
    chosen_cell = np.random.choice(cell_indexes)
    low_eta = cell_edges[chosen_cell][0]
    high_eta = cell_edges[chosen_cell][1]
    low_phi = cell_edges[chosen_cell][2]
    high_phi = cell_edges[chosen_cell][3]

    # Second, select hits in that cell only
    cell_hits = utf.select_hits_eta_phi(
        global_hits, low_eta, high_eta, low_phi, high_phi
    )
    # print("Cell hits", cell_hits.shape)
    # We can now shift back to (X,Y,Z)
    rho = cell_hits[:, 0]
    eta = cell_hits[:, 1]
    phi = cell_hits[:, 2]
    tech1 = cell_hits[:, 3]
    tech2 = cell_hits[:, 4]
    tech3 = cell_hits[:, 5]
    X, Y, Z = utf.convert_rhoetaphi_to_xyz(rho, eta, phi)
    # Here we need to transpose, for some arcane reason
    cell_hits = (np.vstack((X, Y, Z, tech1, tech2, tech3))).transpose()

    # Generate nhits, select randomly nhits to make up the track candidate
    # We select a poisson number around nhits, truncated in between [1, 19]
    generated_nhits = 0
    while generated_nhits not in range(1, max_num_hits):
        generated_nhits = np.random.poisson(mean_num_hits, 1)[0]

    # print("Generated nhits", generated_nhits)
    selected_hits = cell_hits[
        np.random.choice(cell_hits.shape[0], generated_nhits, replace=False)
    ]
    # print("Selected hits", selected_hits.shape)
    # print("Selected hits", selected_hits)

    # This track is already sorted
    candidate_track = tracktop.make_random_track(selected_hits, max_num_hits)

    # print("Candidate track", candidate_track.shape)
    # print("Full candidate track", candidate_track)
    output_tracks = np.vstack((output_tracks, candidate_track))

print("Output tracks", output_tracks.shape)
np.savetxt("foo.csv", output_tracks, delimiter=",", fmt="%.9e")
