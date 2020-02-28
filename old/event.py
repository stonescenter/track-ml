import sys
import os

import pandas as pd
import numpy as np
import random
import glob

from trackml.dataset import load_event
from trackml.dataset import load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event

def dicover_highest_amount_of_hits():
    maxV=0

    with open('/tmp/Events_Train_1') as f:
        for line in f:
            event_prefix = line.strip()
            print(event_prefix)
            hits, cells, particles, truth = load_event(os.path.join('/data/trackMLDB/train_1', event_prefix))

            for index, row in particles.iterrows():
                truth_0 = truth[truth.particle_id == row['particle_id']]
                AUX=truth_0['hit_id'].count()
                if (AUX > maxV):
                    maxV=AUX
                    print(maxV)

dicover_highest_amount_of_hits()
