#!/bin/bash

# move input files
mkdir icelakes
mkdir misc
mkdir geojsons
mkdir detection_out_data
mkdir detection_out_plot
mkdir detection_out_stat
mv __init__.py icelakes/
mv utilities.py icelakes/
mv nsidc.py icelakes/
mv detection.py icelakes/
mv test1 misc/
mv test2 misc/
mv *.geojson geojsons/

# Run the Python script 
python3 detect_lakes.py --granule $1 --polygon $2

if (ls detection_out_stat | wc -l) > 0
then
    echo "Success!!!"
    exit 0
else
    echo "$1" >>$_CONDOR_WRAPPER_ERROR_FILE
    echo "Looks like something went wrong....."
    exit 4
fi