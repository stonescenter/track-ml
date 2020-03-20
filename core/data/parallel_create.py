import json
import os
import sys
import time
import subprocess
import getpass
import psutil
import warnings
import argparse


def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="create parallel datasets")

    # Dataset setting
    parser.add_argument('--config', type=str, default="config.json", help='Configuration file')

    # parse the arguments
    args = parser.parse_args()

    return args

def send_proc(cmd):
    process = subprocess.Popen(cmd)
    return process.pid


def main():

    args = parse_args()

    # load configurations of datasets
    configs = json.load(open(args.config, 'r'))

    lib_path = configs['create']['lib_path']
    
    print(lib_path)

    #sys.path.append(lib_path)
    #from tracktop import *

    begin_id = int(configs['create']['begin_id'])
    end_id = int(configs['create']['end_id'])
    n_cores = int(configs['create']['n_cores'])
    count = 0

    while begin_id <= end_id:
        for i in range(0,n_cores):
            if ((begin_id + i) <= end_id):
                process = send_proc(['python','generate.py', str(begin_id + i), '&'])
        count +=1 
        print('Remaining datasets: ',end_id - begin_id)
        begin_id += n_cores
        time.sleep(configs['create']['round_time'])


if __name__=='__main__':
    main()
        
