#!/bin/sh
# position 6 71 201 113
for i in "$*"
do
	p=$i
done

i=0
t=1340
while [ $i -lt $t ]
do
	echo $i
	python test.py --index=$i --position=$p --val=0
	i=`expr $i + 1`
done