import random
import sys

sys.path.append('/home/silvio/github/track-ml-1/utils/')
from explorer_fake_tracks import *

err=0.3

for i in range(3):
    err=err*0.1
    err_string = "%.10f" % err
    print(err_string)

    with open('/data/ds') as f:
        lines = random.sample(f.readlines(),1500)

    with open('/data/dsRandom', 'w') as f:
        for item in lines:
            f.write("%s" % item)

    filenames = ['/data/dsHeader', '/data/dsRandom']
    with open('/data/dsEval', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())
    call_create_fake("/data/dsEval",err)
