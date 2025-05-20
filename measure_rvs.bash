#!/bin/bash

for file in Gaia_DR3_*_epoch_?.txt; do
    xxx=$(echo "$file" | awk -F'[_]' '{print $0}')
    python3 dh_cc_mask.py $xxx
done
