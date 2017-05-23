# reload modules when they change
%load_ext autoreload
%autoreload 2

# Alow plotting
%matplotlib inline

# Import modules higher in folder hierarchy
import os
import sys
import logging

from imp import reload

reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)