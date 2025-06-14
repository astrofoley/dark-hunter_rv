#!/bin/bash

for file in ./data/Gaia_DR3_*_epoch_?.txt; do
  xxx=$(echo "$file" | awk -F'/' '{print $NF}')
  if [[ $* == *"--args_list"* ]]; then
    target="--args_list"
    file_list=$(awk -v target="$target" '{split($0, arr, " "); for(i=1; i<=length(arr); i++) if(arr[i] == target) print arr[i+1];}' <<< "$*")
    xxx=$(awk -v var="$xxx" '$0~var {var=$0; exit} END {print var}' $file_list)
  fi
  xxx="./data/$xxx"
  read xxx args <<< "$xxx"
  python3 dh_cc_mask_workingcopy.py $xxx $args $*
done

# LAMOST medium resolution data
for file in ./data/med-*.fits; do
  xxx=$(echo "$file" | awk -F'/' '{print $NF}')
  if [[ $* == *"--args_list"* ]]; then
    target="--args_list"
    file_list=$(awk -v target="$target" '{split($0, arr, " "); for(i=1; i<=length(arr); i++) if(arr[i] == target) print arr[i+1];}' <<< "$*")
    xxx=$(awk -v var="$xxx" '$0~var {var=$0; exit} END {print var}' $file_list)
  fi
  xxx="./data/$xxx"
  read xxx args <<< "$xxx"
  python3 dh_cc_mask_workingcopy.py $xxx $args $*
done

# LAMOST Low resolution data
for file in ./data/spec-*.fits; do
  xxx=$(echo "$file" | awk -F'/' '{print $NF}')
  if [[ $* == *"--args_list"* ]]; then
    target="--args_list"
    file_list=$(awk -v target="$target" '{split($0, arr, " "); for(i=1; i<=length(arr); i++) if(arr[i] == target) print arr[i+1];}' <<< "$*")
    xxx=$(awk -v var="$xxx" '$0~var {var=$0; exit} END {print var}' $file_list)
  fi
  xxx="./data/$xxx"
  read xxx args <<< "$xxx"
  python3 dh_cc_mask_workingcopy.py $xxx $args $*
done
