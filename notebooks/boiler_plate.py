# reload modules when they change
%load_ext autoreload
%autoreload 2

# Alow plotting
%matplotlib inline

# Import modules higher in folder hierarchy
import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)
