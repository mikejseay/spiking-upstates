from results import Results
import matplotlib.pyplot as plt

loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/'
targetSim = 'jercogFanFullConn250Units_2020-10-27-10-05'

R = Results(targetSim, loadFolder)

fig2, ax2 = plt.subplots(2, 1, num=2, figsize=(10, 6))
R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
R.plot_voltage_detail(ax2[1], unitType='Inh', useStateInd=0)
