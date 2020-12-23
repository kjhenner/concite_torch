#!/bin/bash
for data_path in $1/*; do
	echo $data_path
	python ./pubmed2elastic.py $data_path
done
