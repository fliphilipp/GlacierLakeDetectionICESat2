#!/bin/bash

# move input files
mkdir misc
mkdir geojsons
mv test1 misc/
mv test2 misc/
mv *.geojson geojsons/
mkdir output

# Run the Python script 
python3 test1.py --granule $1 --polygon $2
