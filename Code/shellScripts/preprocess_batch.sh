#!/bin/bash
data_dir=$1
out_dir=$2
cat ./img_type_list | awk '{print "python preprocess.py '${data_dir}'"$1" '$out_dir' &"}' > workflow.sh
