#!/bin/bash

while getopts c:n: flag
do
    case "${flag}" in
        c) clusterid=${OPTARG};;
        n) clustername=${OPTARG};;
    esac
done

# condor_q overview, and check for unique hold resons
condor_q $clusterid
condor_q $clusterid -held -af HoldReason | wc -l | xargs printf "%s total jobs on hold\n"
echo "Hold reasons:"
condor_q $clusterid -held -af HoldReason | cut -c-11 | sort | uniq -c
printf "\n\n"

filename_out=held_jobs/$clustername-$clusterid.csv
>| $filename_out

condor_q -held -af Args UserLog HoldReason > $filename_out

cat $filename_out | wc -l | xargs printf "%s lines in file\n\n"
