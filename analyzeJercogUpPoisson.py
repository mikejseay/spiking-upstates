from results import Results
import matplotlib.pyplot as plt

saveName = 'buonoEphysBen1_2000_0p25_upPoisson_resumePrior__2021-09-05-19-22_results'
saveFolder = 'C:/Users/mikejseay/Documents/BrianResults/'

R = Results()
R.init_from_file(saveName, saveFolder)

R.calculate_PSTH()
R.calculate_voltage_histogram(removeMode=True, removeReset=True, useAllRecordedUnits=True)
R.calculate_upstates()
if len(R.ups) > 0:
    R.reshape_upstates()
    R.calculate_FR_in_upstates()
    print('average FR in upstate for Exc: {:.2f}, Inh: {:.2f}, average upDur: {:.2f}'.format(R.upstateFRExcHist.mean(), R.upstateFRInhHist.mean(), R.upDurs.mean()))


R.calculate_voltage_histogram(useExcUnits=(2, 4,), useInhUnits=(0,), removeMode=True, removeReset=True)
R.reshape_upstates()

fig1, ax1 = plt.subplots(5, 1, num=1, figsize=(16, 9),
                         gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]},
                         sharex=True)
R.plot_spike_raster(ax1[0])  # uses RNG but with a separate random seed
R.plot_firing_rate(ax1[1])
ax1[1].set_ylim(0, 30)
R.plot_voltage_detail(ax1[2], unitType='Exc', useStateInd=5)
R.plot_voltage_detail(ax1[3], unitType='Inh', useStateInd=0)
R.plot_voltage_detail(ax1[4], unitType='Exc', useStateInd=2)
R.plot_updur_lines(ax1[2])
R.plot_updur_lines(ax1[3])
R.plot_updur_lines(ax1[4])
ax1[4].set(xlabel='Time (s)')
R.plot_voltage_histogram_sideways(ax1[2], 'Exc')
R.plot_voltage_histogram_sideways(ax1[3], 'Inh')
R.plot_voltage_histogram_sideways(ax1[4], 'Exc')
fig1.suptitle(R.p['simName'])