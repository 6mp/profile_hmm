#!/usr/bin/env bash

#/autograder/source is home directory in gradescope

apt-get install -y python3 python3-pip python3-dev gcc meson

pip3 install -r /autograder/source/requirements.txt

# Build the C++ code
cd /autograder/source
meson setup builddir && cd builddir
meson compile
cp /autograder/source/builddir/profile_hmm /autograder/source/profile_hmm