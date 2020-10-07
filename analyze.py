from results import Results
import matplotlib.pyplot as plt

loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/'
targetSim = 'classicJercogUpCrit_2020-10-07-16-50'

R = Results(targetSim, loadFolder)

R.calculate_spike_rate()
R.calculate_voltage_histogram()

fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]})
R.plot_spike_raster(ax1[0])
R.plot_firing_rate(ax1[1])

fig2, ax2 = plt.subplots(3, 1, num=2, figsize=(10, 9))
R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
R.plot_voltage_detail(ax2[1], unitType='Inh', useStateInd=0)
R.plot_voltage_histogram(ax2[2])

