#!/bin/bash

#python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta-05_05.csv" --cylindrical True
#python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta-05_05.csv" --cylindrical False
#python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta1_2.csv" --cylindrical True
#python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta1_2.csv" --cylindrical False
#python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta2_3.csv" --cylindrical True
#python main_mlp.py --config config-mlp.json --dataset "/data/track-ml/eramia/results/eta2_3.csv" --cylindrical False

#python main_test.py --config config_test_lstm.json --dataset "/data/track-ml/eramia/results/eta-05_05.csv" --cylindrical True
#python main_test.py --config config_test_lstm.json --dataset "/data/track-ml/eramia/results/eta-05_05.csv" --cylindrical False
#python main_test.py --config config_test_lstm.json --dataset "/data/track-ml/eramia/results/eta1_2.csv" --cylindrical True
#python main_test.py --config config_test_lstm.json --dataset "/data/track-ml/eramia/results/eta1_2.csv" --cylindrical False
#python main_test.py --config config_test_lstm.json --dataset "/data/track-ml/eramia/results/eta2_3.csv" --cylindrical True
#python main_test.py --config config_test_lstm.json --dataset "/data/track-ml/eramia/results/eta2_3.csv" --cylindrical False

#python main_train.py --config config_cnn.json --dataset "/data/track-ml/eramia/results/eta-05_05.csv" --cylindrical False
#python main_train.py --config config_cnn.json --dataset "/data/track-ml/eramia/results/eta1_2.csv" --cylindrical False
#python main_train.py --config config_cnn.json --dataset "/data/track-ml/eramia/results/eta2_3.csv" --cylindrical False

# ---- novas compilaçoes ----
#python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/dataset/eta-05_05.csv" --cylindrical False
#python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/dataset/eta-05_05.csv" --cylindrical True
#python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/dataset/eta1_2.csv" --cylindrical False
#python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/dataset/eta1_2.csv" --cylindrical True
#python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/dataset/eta2_3.csv" --cylindrical False
#python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/dataset/eta2_3.csv" --cylindrical True


#python main_train.py --config config_cnn.json --dataset "/data/track-ml/eramia/dataset/eta-05_05.csv" --cylindrical False
#python main_train.py --config config_cnn.json --dataset "/data/track-ml/eramia/dataset/eta-05_05.csv" --cylindrical True
#python main_train.py --config config_cnn.json --dataset "/data/track-ml/eramia/dataset/eta1_2.csv" --cylindrical False
#python main_train.py --config config_cnn.json --dataset "/data/track-ml/eramia/dataset/eta1_2.csv" --cylindrical True
#python main_train.py --config config_cnn.json --dataset "/data/track-ml/eramia/dataset/eta2_3.csv" --cylindrical False
#python main_train.py --config config_cnn.json --dataset "/data/track-ml/eramia/dataset/eta2_3.csv" --cylindrical True

#python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/results/eta_n0.5-0.5_phi_n0.5-0.5.csv" --cylindrical False
python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/results/eta_n0.5-0.5_phi_n0.5-0.5.csv" --cylindrical True
#python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/results/eta_0.0-1.0_phi_0.0-1.0.csv " --cylindrical False
#python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/results/eta_0.0-1.0_phi_0.0-1.0.csv " --cylindrical True
#python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/results/eta_n0.5-0.5_phi_ninf-pinf.csv" --cylindrical False
#python main_train.py --config config_lstm.json --dataset "/data/track-ml/eramia/results/eta_n0.5-0.5_phi_ninf-pinf.csv" --cylindrical True
