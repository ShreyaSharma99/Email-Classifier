#!/bin/sh
# if [$1 == train]
# then
# 	python3 a2.py $2 $3
# elif [$2 == test ]
# then
# 	python3 a2_eval.py $2 $2
# else
# 	echo "Wrong command"
# fi

python $1.py $2 $3
