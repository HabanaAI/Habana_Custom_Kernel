#!/bin/bash

rm -rf ~/habana_logs/*

for h in {10..20..2}; 
do
    python3 custom_outer_test.py $h
    sleep 5s
done

echo "Exps Finish !!! "
