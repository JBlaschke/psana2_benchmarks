#!/usr/bin/env bash


set -e


git submodule update --init


# use absolute paths:
project_root=$(readlink -f $(dirname ${BASH_SOURCE[0]}))


docker build                                \
    -t lcls2:benchmarks                     \
    -f $project_root/docker/Dockerfile.base \
    $project_root 
