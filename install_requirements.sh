#!/bin/bash

set -e

conda install -y cudatoolkit=11.0.221
conda install --file -r requirements.txt