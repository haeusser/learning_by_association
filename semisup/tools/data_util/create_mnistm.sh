#!/bin/bash

# fetch BSDS500 data
wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz

# call conversion script via python
python create_mnistm.py

# delete temporary files
rm -r MNIST_data/
rm BSR_bsds500.tgz


