"""
specialized script for analyzing the results of simulateDestexheEphys or simulateJercogEphys
"""

from results import ResultsEphys
import matplotlib.pyplot as plt

loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/'
targetSim = 'classicDestexheEphysOrig_2020-11-03-11-21'

R = ResultsEphys()
R.init_from_file(targetSim, loadFolder)

fig1, ax1 = plt.subplots(2, 2, num=1)

R.calculate_and_plot(fig1, ax1)
