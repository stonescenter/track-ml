import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.histograms import make_phi_range
import utils.transformation as utf
import utils.tracktop as tracktop

# PARAMETERS
input_path = "./real.csv"
output_path = "/data/track-ml/output/"
mean_nhits = 11.0

# Open the input file
tracks = np.genfromtxt(input_path, delimiter=",")
print("Original file", tracks.shape)

# Separate the input tracks in different parts
indexes, vertices, momenta, hits = utf.parts_from_tracks(tracks)
print("Hits", hits.shape)

# Reshape the hits to make a collection of global hits
global_hits = utf.make_hits_rhoetaphi_collection(hits)
print("Global hits", global_hits.shape)

# Select hits only in a small square of (0.4 X 0.5232)
cell_hits = utf.select_hits_eta_phi(global_hits, -0.2, 0.2, -0.2616, 0.2616)
print("Cell hits", cell_hits.shape)

# Generate nhits, select randomly nhits to make up the track candidate
generated_nhits = np.random.poisson(mean_nhits, 1)[0]
selected_hits = cell_hits[
    np.random.choice(cell_hits.shape[0], generated_nhits, replace=False)
]
print("Selected hits", selected_hits.shape)

candidate_track = tracktop.make_random_track(selected_hits, 19)
print("Candidate track", candidate_track.shape)
print("Full candidate track", candidate_track)
