# GlacierLakeDetectionICESat2

NOTE: This repo is private for now. (early testing stage)

**A repository for automatic supraglacial lake detection on the ice sheets in ICESat-2 data**

## Useful commands for OSG:

Login to access node with SSH Keys set up
([Generate SSH Keys and Activate Your OSG Login](https://support.opensciencegrid.org/support/solutions/articles/12000027675)):
```
ssh <username>@<osg-login-node>
```
Example:
```
ssh fliphilipp@login05.osgconnect.net
```

Submit a file to HTCondor:
```
condor_submit <my_submit-file.submit>
```

Watch the progress of the queue after submitting jobs:
```
watch condor_q
```

See which jobs are on hold and why:
```
condor_q -hold
```

Release and re-queue jobs on hold:
```
condor_release <cluster ID>/<job ID>/<username>
```

Remove jobs on hold:
```
condor_rm <cluster ID>/<job ID>/<username>
```

Example to grep log files for memory/disk usage:
```
grep -A 3 'Partitionable Resources' <log_directory>/*.log
```

Put a container in /public stash:
```
scp <mycontainer>.sif fliphilipp@login05.osgconnect.net:/public/fliphilipp/containers/
```

Explore a container on access node:
```
singularity shell --home $PWD:/srv --pwd /srv --scratch /var/tmp --scratch /tmp --contain --ipc --pid /public/fliphilipp/containers/<mycontainer>.sif
```

## To get Singularity working with root privileges:

Get some required packages: 
```
$ sudo apt-get update && sudo apt-get install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup
```
    
Remove any previous intallation of Go, if needed: 
```
$ rm -rf /usr/local/go
```

Download Go and untar: 
```
$ wget https://go.dev/dl/go1.19.linux-amd64.tar.gz
$ sudo tar -C /usr/local -xzf go1.19.linux-amd64.tar.gz
```

Add to path and check installation of Go: 
```
$ echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
$ go version
```

Need glibc for Singularity install:
```
$ sudo apt-get install -y libglib2.0-dev
```

Download Singularity and untar:
```
$ wget https://github.com/sylabs/singularity/releases/download/v3.10.2/singularity-ce-3.10.2.tar.gz
$ tar -xzf singularity-ce-3.10.2.tar.gz
```

Move to the directory and run installation commands:
```
$ ./mconfig
$ make -C builddir
$ sudo make -C builddir install
```

