#!/bin/bash

grep -oP '(?<=This job is running at the following lat/lon location: ).*' outs/$1 --no-filename | sort > osg_job_stats/compute_locations.csv

grep -oP '(?<=Memory \(MB\)          :).*' logs/$1 | awk -F'[ ,]+' '{printf "%s, %s, %s, %s\n",$1,$2,$3,$4;}' > osg_job_stats/memory_usage_mb.csv

grep -oP '(?<=Disk \(KB\)            :).*' logs/$1 | awk -F'[ ,]+' '{printf "%s, %s, %s, %s\n",$1,$2,$3,$4;}' > osg_job_stats/disk_usage_kb.csv
