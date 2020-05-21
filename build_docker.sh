#!/usr/bin/env bash


set -e


# Cori has a dedicated `git-lfs module`
if [[ $NERSC_HOST = "cori" ]]; then
    module load git-lfs
fi

git lfs install
git lfs pull
git submodule update --init


# use absolute paths:
project_root=$(readlink -f $(dirname ${BASH_SOURCE[0]}))


docker build                                \
    -t lcls2:benchmarks                     \
    -f $project_root/docker/Dockerfile.base \
    $project_root 
