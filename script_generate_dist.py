import os
import argparse
import json

from core.data.data_loader import *
from core.utils.utils import *

# python script_generate_dist.py --dataset "/path_to_.csv" --cylindrical False --split 0.8 --normalise True
def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser()

    # Dataset setting
    parser.add_argument('--dataset', type=str, help='path to dataset')
    parser.add_argument('--cylindrical', type=str, help='type of coordenates system')
    parser.add_argument('--split', type=float, help='split rate')
    parser.add_argument('--normalise', type=str, help='normalise the data')
    parser.add_argument('--type_norm', type=str, default='zscore', help='type of normalization data')
    parser.add_argument('--num_obs', type=int, default=4, help='this param is the number of hits of input')   
    parser.add_argument('--num_features', type=int, default=3, help='this param is the of features of a hit')
    parser.add_argument('--output', type=str, default="results", help='outpu of results')
    
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

	if args.type_norm == "zscore":
		kind_norm = KindNormalization.Zscore
	elif args.type_norm == "maxmin":
		kind_norm = KindNormalization.Scaling

	data = Dataset(data_file, float(args.split), cylindrical, 10, kind_norm)

	X_train, y_train = data.get_training_data(n_hit_in=args.num_obs, n_hit_out=1,
	                             n_features=args.num_features, normalise=normalise)

	print('path:', data_file)
	print('encry:', encryp_ds_name)

	if cylindrical:
		coord = 'cylin'
	else:
		coord = 'xyz'

	if normalise:
		data.save_scale_param(output_encry)
		print('Data distribution saved with using %s', coord)
	else:
		print('No data distribution saved')

if __name__=='__main__':
    main()

