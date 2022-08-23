#!/bin/bash

# move input files
mkdir misc
mkdir shapefiles
mv test1 misc/
mv test2 misc/
mv jakobshavn_small.* shapefiles/

# Run the Python script 
python3 test1.py
