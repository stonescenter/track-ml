import sys
import os
import warnings
import argparse
import json
import pandas as pd


def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="Join events")

    # Dataset setting
    parser.add_argument('--config', type=str, default="config.json", help='Configuration file')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():

    args = parse_args()       

    # load configurations of datasets
    configs = json.load(open(args.config, 'r'))

    # Get values fron config file
    datasets_dir = configs['join']['datasets_dir']
    output_path = configs['join']['output_path']
    n_split = configs['join']['n_split']

    # Get the list of dataset in the input dir 
    event_files = pd.DataFrame(os.listdir(datasets_dir))

    # Initializing the final dataset
    tracks_final = pd.DataFrame()

    # Joining the datasets (concatenate)
    for index, row in event_files.iterrows():
        path = datasets_dir + '/' + str(row[0])
        print(str(row[0]))
        #path = str(row[0])
        tracks = pd.read_csv(path)
        tracks_final = pd.concat([tracks_final, tracks], ignore_index=True)

    # splitting the final dataset
    if type(n_split) is int:
        if n_split >  tracks_final.shape[0]:
            n_split = tracks_final.shape[0]
            wrn_msg = ('The number of tracks to split is greater than the number of tracks in '
                       'the file.\nn_plit will be: ' +  str(n_split) +
                       ' (the number of tracks in the file)')
            warnings.warn(wrn_msg, RuntimeWarning, stacklevel=2)

        df_split = tracks_final.iloc[:n_split,:]
        df_split.to_csv(output_path, index = False)
    else:
        tracks_final.to_csv(output_path, index = False)

if __name__=='__main__':
    main()
