#!/bin/bash

for f in ` ls /data/trackMLDB/train_1/ | head -n 40 | cut -d "-" -f 1 | uniq ` ; do
  echo $f
  python dataPreparation.py $f &
done

wait -n

echo "finish all"
