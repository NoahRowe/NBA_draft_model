#!/usr/bin/env bash

# teamFeatures_1 is manhattan with label stuff
# teamFeatures_2 is euclidean with label stuff

# Set up the loop
scaler_types=( 'standard' ) # best one
n_clusters=( 3 ) # best one
distance_metrics=( 'manhattan' ) # best one 
OS_sizes=( 0 ) # best one
OS_vals=( 0 ) # best one
#losses=( 'mse' 'msle' 'mae' ) 
#OS_sizes=( 50 100 150 200 250)
#OS_vals=( 2.0 2.5 3.0 3.5 4.0 )

counts=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 ) 

for distance_metric in "${distance_metrics[@]}" 
do
for n_cluster in "${n_clusters[@]}"
do
for scaler_type in "${scaler_types[@]}"
do
for count in "${counts[@]}"
do
    nice -n 19 python3 NN_model.py $scaler_type $n_cluster $distance_metric $count &
    sleep 1
done
done
done
done