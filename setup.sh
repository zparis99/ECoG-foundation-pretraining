#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

pip install --upgrade pip

python3.11 -m venv ecog
source ecog/bin/activate

pip install -r requirements.txt

