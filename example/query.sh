#!/bin/bash
cd "$(dirname $0)"

# check for liteqa datasets
[ ! -e data1.lqa/ ] && echo "no data1.lqa/ directory, run compress.sh" && exit 1
[ ! -e data2.lqa/ ] && echo "no data2.lqa/ directory, run compress.sh" && exit 1

# export location of liteqa_c shared object if not installed in system
[ ! -e ../liteqa_c/liteqa_c.so ] && ( cd ../liteqa_c/ ; make )
export LITEQA_C=../liteqa_c/liteqa_c.so

# run queries using different operators: < > <= >= == !=
# intersect query results using AND (OR is not supported)
# compute function on index look-up tables: QUANTILE, MEAN, MEDIAN, MIN, MAX

../lqaquery.py data1.lqa "vel0 < 0 AND vmag > QUANTILE(0.05)"
../lqaquery.py data1.lqa "vmag > MEAN()"
../lqaquery.py data1.lqa "vmag > MEDIAN()"
../lqaquery.py data1.lqa "vmag > QUANTILE(0.99)"
../lqaquery.py data1.lqa "vmag == MIN()"
../lqaquery.py data1.lqa "vmag == MAX()"
../lqaquery.py data1.lqa "qcrit > QUANTILE(0.99)"
../lqaquery.py data1.lqa "qcrit > QUANTILE(0.99) AND dist >= 2"
../lqaquery.py data1.lqa "vort > QUANTILE(0.99)"
../lqaquery.py data1.lqa "vort > QUANTILE(0.99) AND dist >= 2"
../lqaquery.py data1.lqa "vel0 < 0 AND vmag > QUANTILE(0.05)"

# functions can be computed on other datasets,
# use first parameter for choosing the other dataset

../lqaquery.py data1.lqa "vel0 < 0 AND vmag > QUANTILE(data2.lqa/vmag, 0.01)"
