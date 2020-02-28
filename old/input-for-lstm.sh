#!/bin/bash
#1 -> file to transform
#2 -> outputfile
#ex: ~/pt1p0_train_1_realv3_10

user_dir=`echo ~`
dir="$user_dir"/res

if [ -d $dir ] ; then

x=8
y=31
#echo "$x -- $y"
cut $1 -d "," -f $x-$y | tail -n+2 > ~/res/res0

#for i in `seq 1 25` ; do
for i in `seq 1 24` ; do
  x=$(( $x + 6 ))
  y=$(( $y + 6 ))
  #echo "$x -- $y"
  cut $1 -d "," -f $x-$y | tail -n+2 > ~/res/res$i
done

cut $1 -d "," -f 1-31 | head -n 1 > ~/res/headdd
cut $1 -d "," -f 8-31 | head -n 1 > ~/res/head
cut $1 -d "," -f 1-7 | head -n 1 > ~/res/headP

cut $1 -d "," -f 1-7 | tail -n+2 > ~/res/aux

cat ~/res/head > ~/res/total
cat ~/res/headP > ~/res/totalP
for i in `seq 0 24` ; do
  cp ~/res/aux ~/res/aux$i
  cat ~/res/aux$i >> ~/res/totalP
  cat ~/res/res$i >> ~/res/total
done

paste -d, ~/res/totalP ~/res/total > $2

#paste res/totalP res/total > res/resfinal
else
    echo "Error: Directory ~/res does not exists."
fi
