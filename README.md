# Paradoxical Self-Sustained Dynamics Emerge from Orchestrated Excitatory and Inhibitory Homeostatic Rules

Saray Soldado-Magraner, Michael Seay, Rodrigo Laje, Dean V. Buonomano (2022)

This github repository is designed to reproduce the figure containing the spiking model results.

## Installation

- Start from a Python 3 Anaconda environment.
- Install Brian2 according to the installation instructions (https://brian2.readthedocs.io/en/stable/introduction/install.html).
- Install dill by running 'conda install dill' at an Anaconda-enable command line.
- Clone this repository.

## Usage

- At an Anaconda-enabled command line, make this repository your current directory.
- Run 'python figureTraining.py'. This will use the pre-computed results file to generate panels B, C, D, E, F, G, H, J, and K. They will appear as PDF files in the root directory.
- Run 'python figureParadoxical.py'. This will use the pre-computed results file to generate panel I. It will appear as a PDF file in the root directory.
- If interested, you can run the training session and/or paradoxical effect experiments yourself by running 'python runTrainingSession.py' and 'python runParadoxicalSession.py' respectively. These will generate new results files in the results subdirectory, which can then be plotted by editing the target files of the figure-generating scripts. 

## Authors

* **Michael Seay** - [mikejseay](https://github.com/mikejseay)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

The spiking model implemented here was heavily based on  the spiking model from:

* Jercog, D. et al. UP-DOWN cortical dynamics reflect state transitions in a bistable network. Elife 6, 1–33 (2017). https://doi.org/10.7554/eLife.22425 

## Citation

This code is the product of work carried out in the group of [Dean Buonomano at the University of California Los Angeles](http://www.buonomanolab.com/). If you find our code helpful to your work, consider citing us in your publications:

* Saray Soldado-Magraner, Michael Seay, Rodrigo Laje, Dean V. Buonomano. Paradoxical Self-Sustained Dynamics Emerge from Orchestrated Excitatory and Inhibitory Homeostatic Rules. 2022.
