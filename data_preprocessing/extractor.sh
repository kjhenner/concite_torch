#!/bin/bash
for data_path in $1/*; do
	echo $data_path
	python ./pubmed_extractor.py $data_path $2 20
done
