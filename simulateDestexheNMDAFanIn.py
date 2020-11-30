from brian2 import defaultclock, pA, ms, nA, mV
from params import paramsDestexhe as p
from network import DestexheNetwork
from results import Results
import matplotlib.pyplot as plt
import numpy as np

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

# determine 'fan in' empirically
# here, should show that if a unit is DEPOLARIZED, the NMDA enabled version has a smaller fan-in

p['simName'] = 'destexheFanInNMDA'
p['recordStateVariables'] = p['recordStateVariables'] = ['v', 'ge', 's_NMDA_tot']

# we set the number of units that will be used to set the weight
# ONLY FOR THE FAN IN EXPERIMENT
p['nInhFan'] = int(p['propInh'] * p['nUnits'])
p['nExcFan'] = int(p['nUnits'] - p['nInhFan'])

nRecurrentExcitatorySynapsesPerUnit = int(p['nExcFan'] * p['propConnect'])
nRecurrentInhibitorySynapsesPerUnit = int(p['nInhFan'] * p['propConnect'])

useQExc = p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
useQInh = p['qInh'] / nRecurrentInhibitorySynapsesPerUnit
useQNMDA = 0.5 * useQExc

# we make one Exc/Inh unit each and make them not adapt
p['nUnits'] = 2
p['propInh'] = 0.5
p['bExc'] = 0 * pA
p['bInh'] = 0 * pA

MIN_UNITS = 1
MAX_UNITS = 15
UNIT_SPACING = 1
TIME_SPACING = 250 * ms
USE_I_EXT = 0.09 * nA

# NMDA version

DN = DestexheNetwork(p)
DN.initialize_network()
DN.initialize_units_NMDA()
DN.units.iAmp = USE_I_EXT

DN.determine_fan_in_NMDA(minUnits=MIN_UNITS, maxUnits=MAX_UNITS, unitSpacing=UNIT_SPACING, timeSpacing=TIME_SPACING)
DN.save_results()
DN.save_params()

R = Results(DN.saveName, DN.p['saveFolder'])

fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(10, 6), sharex=True)

R.plot_voltage_detail(ax1[0], unitType='Exc', useStateInd=0, ls="-")
R.plot_voltage_detail(ax1[1], unitType='Inh', useStateInd=0, ls="-")

stateMonExc = DN.stateMonExc

useUnit = 0
v = stateMonExc.v[useUnit, :]
s_NMDA_tot = stateMonExc.s_NMDA_tot[useUnit, :]
ge = stateMonExc.ge[useUnit, :]

AMPACurrent = ge * (p['eExcSyn'] - v)

hardGatePart = np.round(v > -60 * mV)
NMDACurrent = useQNMDA * (p['eExcSyn'] - v) / (1 + np.exp(-0.4 * (v + 50 * mV) / mV)) * s_NMDA_tot
ActualNMDACurrent = hardGatePart * NMDACurrent

fig2, ax2 = plt.subplots(2, 1, num=2, figsize=(10, 6), sharex=True)

ax2[0].plot(stateMonExc.s_NMDA_tot.T)
ax2[1].plot(AMPACurrent)
ax2[1].plot(ActualNMDACurrent)

# no NMDA version

# p['simName'] = 'destexheFanIn'
#
# DN = DestexheNetwork(p)
# DN.initialize_network()
# DN.initialize_units_iExt()
# DN.units.iAmp = USE_I_EXT
#
# DN.determine_fan_in(minUnits=MIN_UNITS, maxUnits=MAX_UNITS, unitSpacing=UNIT_SPACING, timeSpacing=TIME_SPACING)
# DN.save_results()
# DN.save_params()
#
# R = Results(DN.saveName, DN.p['saveFolder'])
#
# R.plot_voltage_detail(ax1[0], unitType='Exc', useStateInd=0, ls=":")
# R.plot_voltage_detail(ax1[1], unitType='Inh', useStateInd=0, ls=":")
