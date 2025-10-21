#!/bin/bash

if [[ $* == *"--path"* ]]; then
    target="--path"
    path_par=$(awk -v target="$target" '{split($0, arr, " "); for(i=1; i<=length(arr); i++) if(arr[i] == target) print arr[i+1];}' <<< "$*")
    pathname=$(awk -v var="$path_par" '$0~var {var=$0; exit} END {print var}' "$path_par")
else
  pathname="./data"  # Default directory where RV summary files are stored
  fi

for file in "$pathname"/*_summary.txt; do
  if [[ $* == *"--args_list"* ]]; then
    target="--args_list"
    file_list=$(awk -v target="$target" '{split($0, arr, " "); for(i=1; i<=length(arr); i++) if(arr[i] == target) print arr[i+1];}' <<< "$*")
    file=$(awk -v var="$file" '$0~var {var=$0; exit} END {print var}' "$file_list")
  fi
  python3 rv_fit.py $file $args $*
done