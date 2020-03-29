import os
import sys
import json

#opening json config file
with open('config.json') as config_file:
        data = json.load(config_file)


sys.path.append(data['create']['lib_path'])
from tracktop import *

input_dir = data['create']['input_dir']
output_dir = data['create']['output_dir']

# get id of event
id = sys.argv[1]

# concatenate prefix with id
full_id = ("event00000" + str(id))

# concatenate path + prefix + id
full_output = (str(output_dir) + "/" + str(full_id)) 

#create_input(input_dir, full_id, output = full_output)
create_input_im(input_dir, full_id, 
                output = full_output,
                n_hits_range = [int(data['create']['n_hits_range_min']), int(data['create']['n_hits_range_max'])],
                phi_range=[data['create']['phi_range_min'], data['create']['phi_range_max']], 
                eta_range=[data['create']['eta_range_min'], data['create']['eta_range_max']],
                pt_range=[data['create']['pt_range_min'], data['create']['pt_range_max']],
                silent = data['create']['silent'])

