#!/bin/bash

# Local variables
PROJECT_NAME=tda
PYTHON=3.9

# Recover the project's directory from the position of the install.sh
# script and move there. Not doing so would install some dependencies in
# the wrong place
HERE=`dirname $0`
HERE=`realpath $HERE`
cd $HERE

# Installation of ML Module in a conda environment
echo "_____________________________________________"
echo
echo " 🧩 Persistent Homology Module Installer 🤖 "
echo "                                             "
echo "_____________________________________________"
echo
echo
echo "⭐ Searching for installed CUDA"
echo

# Recover the CUDA version using nvcc
CUDA_VERSION=`nvcc --version | grep release | sed 's/.* release //' | sed 's/, .*//'`
CUDA_MAJOR=`echo ${CUDA_VERSION} | sed 's/\..*//'`
CUDA_MINOR=`echo ${CUDA_VERSION} | sed 's/.*\.//'`

echo
echo
echo "⭐ Searching for installed conda"
echo
# Recover the path to conda on your machine
# First search the default '~/miniconda3' and '~/anaconda3' paths. If
# those do not exist, ask for user input
CONDA_DIR=`realpath ~/miniconda3`
if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
  CONDA_DIR=`realpath ~/anaconda3`
fi
if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
  CONDA_DIR=`realpath ~/miniforge3`
fi

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide your conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda conda found at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh

echo
echo
echo "⭐ Creating conda environment '${PROJECT_NAME}'"
echo
# Create deep_view_aggregation environment from yml
conda create --name ${PROJECT_NAME} python=${PYTHON} -y

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh  
conda activate ${PROJECT_NAME}

echo
echo
echo "⭐ Installing conda and pip dependencies"
echo
conda install pip nb_conda_kernels -y
pip install flooder
pip install persim
pip install laspy
pip install torch --index-url https://download.pytorch.org/whl/${CUDA_MAJOR}${CUDA_MINOR}

# let user know
echo
echo
echo "🚀 Successfully completed"