from results import Results
import matplotlib.pyplot as plt

loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/'
targetSim = 'classicDestexheNMDATest_2020-11-11-13-48'

R = Results(targetSim, loadFolder)
R.calculate_spike_rate()
R.calculate_voltage_histogram(removeMode=True)
R.calculate_upstates()
if len(R.ups) > 0:
    R.reshape_upstates()
    R.calculate_FR_in_upstates()


fig2, ax2 = plt.subplots(3, 1, num=2, figsize=(10, 9))
R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
R.plot_voltage_detail(ax2[1], unitType='Exc', useStateInd=1)
R.plot_voltage_detail(ax2[2], unitType='Inh', useStateInd=0)
