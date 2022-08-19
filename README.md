# GlacierLakeDetectionICESat2

NOTE: This repo is private for now. (early testing stage)

**A repository for automatic supraglacial lake detection on the ice sheets in ICESat-2 data**

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

