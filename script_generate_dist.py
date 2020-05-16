import os
import argparse
import json

from core.data.data_loader import *
from core.utils.utils import *

# python generate_distribution.py --dataset dataset/2020_100_sorted.csv --cylindrical true --split 0.7 --normalise true
def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser()

    # Dataset setting
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--cylindrical', type=str, help='Type of Coordenates system')
    parser.add_argument('--split', type=float, help='this param load model')
    parser.add_argument('--normalise', type=str, help='this param load model')
    parser.add_argument('--output', type=str, default="results", help='this param load model')
    
    # parse the arguments
    args = parser.parse_args()

    return args

def main():

	args = parse_args()

	if args.dataset is not None:
		data_file = args.dataset

	output_path = args.output	
	#create a encryp name for dataset
	path_to, filename = os.path.split(data_file)

	orig_ds_name = filename

	encryp_ds_name = get_unique_name(orig_ds_name)
	decryp_ds_name = get_decryp_name(encryp_ds_name)

	output_encry = os.path.join(output_path, encryp_ds_name)

	if os.path.isdir(output_encry) == False: 
		os.mkdir(output_encry)

	cylindrical = True if args.cylindrical == "True" else False
	normalise = True if args.normalise == "True" else False

	data = Dataset(data_file, float(args.split), cylindrical, 10, KindNormalization.Zscore)

	X_train, y_train = data.get_training_data(n_hit_in=4, n_hit_out=1,
	                             n_features=3, normalise=normalise)

	print('path:', data_file)
	print('encry:', encryp_ds_name)
	print(cylindrical)
	print(normalise)

	if normalise:
	    data.save_scale_param(output_encry)
	    print('scaled data saved!')
	else:
		print('no scaled saved')

if __name__=='__main__':
    main()

