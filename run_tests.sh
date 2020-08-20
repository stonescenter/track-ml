## new datasets for testing

# region internal
#python main_inference.py --config config_lstm.json --dataset "/data/track-ml/eramia/dataset/eta_n0.5-0.5_phi_n0.5-0.5.csv" --cylindrical False --load True
#python main_inference.py --config config_lstm.json --dataset "/data/track-ml/eramia/dataset/eta_n0.5-0.5_phi_n0.5-0.5.csv" --cylindrical True --load True

# intermediary region
#python main_inference.py --config config_lstm.json --dataset "/data/track-ml/eramia/results/eta_0.0-1.0_phi_0.0-1.0.csv " --cylindrical False
#python main_inference.py --config config_lstm.json --dataset "/data/track-ml/eramia/results/eta_0.0-1.0_phi_0.0-1.0.csv " --cylindrical True

# external region
#python main_inference.py --config config_lstm.json --dataset "/data/track-ml/eramia/dataset/eta_n0.5-0.5_phi_ninf-pinf.csv" --cylindrical True --load True

# external region . first we generate de data distribution
python script_generate_dist.py --dataset "dataset/eta_n0.5-0.5_phi_n0.5-0.5_short.csv" --cylindrical False --split 0.8 --normalise True
python main_inference.py --config config_lstm.json --dataset "dataset/eta_n0.5-0.5_phi_n0.5-0.5_short.csv" --cylindrical False --load True

python script_generate_dist.py --dataset "dataset/eta_n0.5-0.5_phi_n0.5-0.5_short.csv" --cylindrical True --split 0.8 --normalise True
python main_inference.py --config config_lstm.json --dataset "dataset/eta_n0.5-0.5_phi_n0.5-0.5_short.csv" --cylindrical True --load True



