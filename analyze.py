from results import Results
import matplotlib.pyplot as plt

loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/'
# loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/perfect_files/'
# loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/long_files/'

# targetSim = 'classicJercog_2020-10-12-16-02'
# targetSim = 'classicJercog0p05Conn_2020-10-13-11-45'
# targetSim = 'classicDestexhe0p05Conn_2020-10-13-12-09'
# targetSim = 'classicDestexheFullConn_2020-10-13-11-21'

# targetSim = 'jercog0p05Conn_2020-10-27-15-24'
# targetSim = 'destexheUpCrit0p05Conn5e4units_2020-11-01-23-17'
targetSim = 'destexhe0p05Conn5e4units_2020-11-02-11-06'

R = Results(targetSim, loadFolder)
R.calculate_spike_rate()
R.calculate_voltage_histogram(removeMode=True)
R.calculate_upstates()
if len(R.ups) > 0:
    R.reshape_upstates()
    R.calculate_FR_in_upstates()
    print('average FR in upstate for Exc: {:.2f}, Inh: {:.2f} '.format(R.upstateFRExc.mean(), R.upstateFRInh.mean()))


# quit()

fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                         sharex=True)
R.plot_spike_raster(ax1[0])
R.plot_firing_rate(ax1[1])


fig2, ax2 = plt.subplots(3, 1, num=2, figsize=(10, 9))
R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
R.plot_updur_lines(ax2[0])
R.plot_voltage_detail(ax2[1], unitType='Inh', useStateInd=0)
R.plot_updur_lines(ax2[1])
R.plot_voltage_histogram(ax2[2], yScaleLog=True)


fig3, ax3 = plt.subplots(num=3, figsize=(10, 9))
R.plot_state_duration_histogram(ax3)


fig4, ax4 = plt.subplots(1, 2, num=4, figsize=(10, 9))
R.plot_consecutive_state_correlation(ax4)


fig5, ax5 = plt.subplots(2, 1, num=5, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                         sharex=True)
R.plot_upstate_voltage_image(ax5[0])
R.plot_upstate_voltages(ax5[1])

fig6, ax6 = plt.subplots(2, 1, num=6, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                         sharex=True)
R.plot_upstate_raster(ax6[0])
R.plot_upstate_FR(ax6[1])
