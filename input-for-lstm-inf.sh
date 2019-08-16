#!/bin/bash
#1 -> file to transform
#ex: ~/pt1p0_train_1_realv3_10

x=27
y=31
echo "$x -- $y"
cut $1 -d "," -f $x-$y | tail -n+2 > res/res0

#for i in `seq 1 25` ; do
for i in `seq 1 24` ; do
  x=$(( $x + 6 ))
  y=$(( $y + 6 ))
  echo "$x -- $y"
  cut $1 -d "," -f $x-$y | tail -n+2 > res/res$i
done

#cut $1 -d "," -f 1-31 | head -n 1 > res/headdd
cut $1 -d "," -f 27-31 | head -n 1 > res/head
cut $1 -d "," -f 1-26 | head -n 1 > res/headP

cut $1 -d "," -f 1-26 | tail -n+2 > res/aux

cat res/head > res/total
cat res/headP > res/totalP
for i in `seq 0 24` ; do
  cp res/aux res/aux$i
  cat res/res$i >> res/total
  cat res/aux$i >> res/totalP
done

paste -d, res/totalP res/total > res/resfinalinf
#paste res/totalP res/total > res/resfinal
