#!/usr/bin/env bash


# stop running if there's an error
set -e


# load site-specific variables:
source $(readlink -f $(dirname ${BASH_SOURCE[0]}))/vars.sh


# access "our conda"
if [[ -e $(readlink -f $(dirname ${BASH_SOURCE[0]}))/../../conda.local/env.sh ]]; then
    source $(readlink -f $(dirname ${BASH_SOURCE[0]}))/../../conda.local/env.sh
fi


# check if already installed
# this will only match packages in the current env root
env_grep=$(conda env list | tr '/' 'x' | grep -w $CONDA_ENV  || true)
# don't match paths        ^^^^^^^^^^^^       ^^
if [[ -n $env_grep ]]; then
    echo "$CONDA_ENV already exists... exiting"
    exit
fi


# update user
echo "BUILDING NEW CONDA ENV $CONDA_ENV"


# create new conda environment (things specific to the particular compute
# environment are kept in `base`)
conda create -y -n $CONDA_ENV --clone $CONDA_BASE
# activate the new environment (this way we do not need to do `conda init
# <shell name>`, as this is not compatible with all compute environments ...
# cough... nersc ... cough...)
source activate $CONDA_ENV


# install conda and pip packages
pkg_data_dir=$(readlink -f $(dirname ${BASH_SOURCE[0]}))/pkg_data

echo "ADDING PACKAGES TO CONDA ENV $CONDA_ENV"
# conda install -y ${CONDA_PKG[@]} $(_channel_list ${CONDA_CH[@]})
for (( i=0; i<${#CONDA_REQ[@]}; ++i )); do
    conda install -y --file $pkg_data_dir/${CONDA_REQ[$i]} -c ${CONDA_CH[$i]}
done

echo "ADDING PIP PACKAGES TO CONDA ENV $CONDA_ENV"
# pip install ${PIP_PKG[@]}
pip install -r $pkg_data_dir/$PIP



echo ""
