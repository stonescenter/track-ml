#!/bin/bash

python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta-05_05.csv" --cylindrical True
python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta-05_05.csv" --cylindrical False
python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta1_2.csv" --cylindrical True
python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta1_2.csv" --cylindrical False
python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta2_3.csv" --cylindrical True
python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta2_3.csv" --cylindrical False