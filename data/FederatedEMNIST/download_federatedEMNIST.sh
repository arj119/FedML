#!/bin/bash

mkdir -p datasets
cd datasets
wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/fed_emnist.tar.bz2
tar -xvf fed_emnist.tar.bz2
