#!/usr/bin/env python3
# PYTHON_PREAMBLE_START_STANDARD:{{{

# Christopher David Cotton (c)
# http://www.cdcotton.com

import os
import sys

# Get full real filename
if os.path.islink(__file__):
    __fullrealfile__ = os.readlink(__file__)
else:
    __fullrealfile__ = os.path.abspath(__file__)

# Function to get git directory containing this file
def getprojectdir(filename):
    curlevel = filename
    while curlevel is not '/':
        curlevel = os.path.dirname(curlevel)
        if os.path.exists(curlevel + '/.git/'):
            return(curlevel + '/')
    return(None)

# Directory of project
__projectdir__ = getprojectdir(__fullrealfile__)

# Setting up importattr (function to load attributes from files by absolute path)
if 'fullmodulesloaded' not in globals():
    fullmodulesloaded = {}
sys.path.append(__projectdir__ + 'submodules/python-preamble/')
from importattr2 import importattr2

def importattr(filename, function, fullmodulesloaded = fullmodulesloaded, curfilename = __fullrealfile__, bybasename = False):
    if os.path.abspath(filename) == curfilename:
        func = eval(function)
    else:
        func = importattr2(filename, function, fullmodulesloaded, bybasename = bybasename)
    return(func)

# PYTHON_PREAMBLE_END:}}}

old = importattr(__projectdir__ + 'sim.py', 'defaultsdictgen')()
new = importattr(__projectdir__ + 'sim.py', 'multipleofdefaults')({'T': 1.01})
importattr(__projectdir__ + 'sim.py', 'allmodels')(old, after = new, savedir = __projectdir__ + 'temp/showclass/')
