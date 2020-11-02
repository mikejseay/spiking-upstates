# spiking-upstates

 Simulating Up states with models of large networks of spiking neurons

# Installation

- Start from a Python 3 Anaconda environment.
- Install Brian2 according to the installation instructions (https://brian2.readthedocs.io/en/stable/introduction/install.html).
- Install dill by running 'conda install dill' at an Anaconda-enable command line.
- Install tqdm by running 'pip install tqdm' at an Anaconda-enable command line.
- Get the latest version of PyCharm Community Edition.

# Usage

- Open simulate*.py in PyCharm.
- Set the save folder to a local folder location of choice.
- Run simulate*.py
- Once it's done, open analyze.py in PyCharm.
- Set the load folder to the same folder as the save folder above.
- Set the targetSim to analyze by copying the prefix before '_params" and '_results'.
- Run analyze.py.
- Go to "Run --> Edit Configurations" and for analyze.py, check "Run with Python console." (Only needs to be done once.)
- Run analyze.py again.
