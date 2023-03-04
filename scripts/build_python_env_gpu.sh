#!/bin/bash

virtualenv ~/my/venv -p /usr/bin/python3.6
source venv/bin/activate
pip install tensorflow-gpu==1.14.0
cd ~/my/stable-baselines/
pip install -e .[tests]
pip install pandas
pip install sklearn
pip install empyrical
pip install ta==0.4.7
pip install openpyxl
pip install jupyter
deactivate