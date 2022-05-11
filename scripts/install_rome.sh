#!/bin/bash

set -Eeuo pipefail

SCRIPTS_DIR=$(realpath $(dirname $0))
REPO_DIR=$(dirname $SCRIPTS_DIR)
cd $REPO_DIR

echo "Cloning rome into ${REPO_DIR}..."
git clone -q https://github.com/kmeng01/rome.git

echo "installing rome dependencies..."
./rome/scripts/setup_conda.sh

eval "$(conda shell.bash hook)"
conda activate rome
pip install -e .

echo "all done!"