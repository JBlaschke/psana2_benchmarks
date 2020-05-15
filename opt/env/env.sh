#!/usr/bin/env bash


# don's stop if errors => this is supposed to be sourced by the main environment
# set -e


# load site-specific variables:
source $(readlink -f $(dirname ${BASH_SOURCE[0]}))/vars.sh


# prepend local conda install
if [[ -e $(readlink -f $(dirname ${BASH_SOURCE[0]}))/../../conda.local/env.sh ]]; then
    source $(readlink -f $(dirname ${BASH_SOURCE[0]}))/../../conda.local/env.sh
fi


# check if the local environment exists
# this will only match packages in the current env root
env_grep=$(conda env list | tr '/' 'x' | grep -w $CONDA_ENV  || true)
# don't match paths        ^^^^^^^^^^^^       ^^
if [[ -z $env_grep ]]; then
    echo "$CONDA_ENV does not exist... exiting"
    echo "To create a conda env, run:"
    echo "    ./conda/setup_env.sh"
    return
fi


# make sure that a conda env isn't already running
# redirect stderr because we might get the warning that conda hasn't modified
# the .bashrc (which we don't want to do anyway)
conda deactivate 2> /dev/null || true
# activate conda
source activate $CONDA_ENV


# Python version
export PYVER=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
