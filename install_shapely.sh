#!/bin/bash

set -e

echo "** Install requirements"

sudo apt-get install libxml2-dev libxslt-dev
sudo apt-get install libgeos-dev
pip3 install Shapely