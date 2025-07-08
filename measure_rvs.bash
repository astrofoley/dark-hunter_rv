#!/bin/bash

if [[ $* == *"--path"* ]]; then
    target="--path"
    path_par=$(awk -v target="$target" '{split($0, arr, " "); for(i=1; i<=length(arr); i++) if(arr[i] == target) print arr[i+1];}' <<< "$*")
    pathname=$(awk -v var="$path_par" '$0~var {var=$0; exit} END {print var}' "$path_par")
else
  pathname="./data"
  fi

for file in "$pathname"/*; do
  filename=$(echo "$file" | awk -F'/' '{print $NF}')
  if [[ $* == *"--args_list"* ]]; then
    target="--args_list"
    file_list=$(awk -v target="$target" '{split($0, arr, " "); for(i=1; i<=length(arr); i++) if(arr[i] == target) print arr[i+1];}' <<< "$*")
    filename=$(awk -v var="$filename" '$0~var {var=$0; exit} END {print var}' "$file_list")
  fi
  filename="./data/$filename"
  read filename args <<< "$filename"
  python3 dh_cc_mask.py $filename $args $*
done
