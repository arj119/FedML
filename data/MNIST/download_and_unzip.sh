#!/bin/bash

rm -rf mnist
rm -rf train
rm -rf test

ARG1=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cU_LcBAUZvfZWveOMhG4G5Fg9uFXhVdf' -O- \
\| sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$ARG1&id=1cU_LcBAUZvfZWveOMhG4G5Fg9uFXhVdf" -O MNIST.zip && rm -rf /tmp/cookies.txt

unzip MNIST.zip

mv mnist/train train
mv mnist/test test
rm -rf mnist
rm -rf MNIST.zip