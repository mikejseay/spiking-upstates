# Creation of neuronal ensembles and cell-specific homeostatic plasticity through chronic sparse optogenetic stimulation

Ben Liu, Michael Seay, Dean V. Buonomano (2022)

This github repository is designed to reproduce the figures containing the spiking model results.

## Installation

- Start from a Python 3 Anaconda environment (the crucial libraries are `numpy`, `matplotlib`,`scipy`, `seaborn`, and `pandas`).
- Install `brian2` according to the installation instructions (https://brian2.readthedocs.io/en/stable/introduction/install.html).
- Install `dill` by running `conda install dill` at an Anaconda-enable command line.
- Clone this repository.

## Usage

- At an Anaconda-enabled command line, make this repository your current directory.
- Run `python fig5a.py`, `python fig5bc.py`, `python fig5de.py`, and `python fig6.py`. Generate figures will appear as PDF files in the root directory. 

## Authors

* **Michael Seay** - [mikejseay](https://github.com/mikejseay)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

The spiking model implemented here was heavily based on  the spiking model from:

* Jercog, D. et al. UP-DOWN cortical dynamics reflect state transitions in a bistable network. Elife 6, 1–33 (2017). https://doi.org/10.7554/eLife.22425 

## Citation

This code is the product of work carried out in the group of [Dean Buonomano at the University of California Los Angeles](http://www.buonomanolab.com/). If you find our code helpful to your work, consider citing us in your publications:

* Ben Liu, Michael Seay, Dean V. Buonomano. Creation of neuronal ensembles and cell-specific homeostatic plasticity through chronic sparse optogenetic stimulation. 2022.