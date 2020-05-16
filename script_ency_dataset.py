import os
import argparse
import json
from core.utils.utils import *

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser()

    # Dataset setting
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--cylindrical', type=str, help='Type of Coordenates system')
    parser.add_argument('--load', type=str, help='this param load model')
    
    # parse the arguments
    args = parser.parse_args()

    return args


def main():

	args = parse_args()

	# load configurations of model and others
	print(args.config)
	if args.config is not None:
		configs = json.load(open(args.config, 'r'))
		data_file = configs['data']['filename']

	if args.dataset is not None:
		data_file = args.dataset
		
	#create a encryp name for dataset
	path_to, filename = os.path.split(data_file)

	orig_ds_name = filename

	encryp_ds_name = get_unique_name(orig_ds_name)
	decryp_ds_name = get_decryp_name(encryp_ds_name)

	#output_encry = os.path.join(output_path, encryp_ds_name)

	print('path:', data_file)
	print('encry:', encryp_ds_name)

if __name__=='__main__':
    main()