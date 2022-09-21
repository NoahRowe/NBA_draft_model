#!/usr/bin/env bash

# Set up the loop
scaler_type='standard'
n_clusters=3
distance_metric='euclidean'
n_players=10
nba_scaler_type='standard'
OS_sizes=( 0 )
US_sizes=( 0 )
OS_vals=( 0 )
US_vals=( 0 )
counts=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 )

for OS_val in "${OS_vals[@]}" 
do
for US_val in "${US_vals[@]}" 
do
for OS_size in "${OS_sizes[@]}" 
do
for US_size in "${US_sizes[@]}" 
do
for count in "${counts[@]}" 
do
    nice -n 19 python3 RF_model.py $scaler_type $n_clusters $distance_metric $n_players $nba_scaler_type $OS_size $US_size $OS_val $US_val $count &
    sleep 1
done
done
done
done
done