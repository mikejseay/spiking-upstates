from brian2 import set_device, defaultclock, ms, second, pA
from params import paramsJercog as p
from params import paramsJercogEphysBuono
from network import JercogNetwork
from results import Results
import matplotlib.pyplot as plt

# for using Brian2GENN
USE_BRIAN2GENN = False
if USE_BRIAN2GENN:
    import brian2genn
    set_device('genn', debug=False)

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

# determine 'up crit' empirically

USE_NEW_EPHYS_PARAMS = False

# remove protected keys from the dict whose params are being imported
ephysParams = paramsJercogEphysBuono.copy()
protectedKeys = ('nUnits', 'propInh', 'duration')
for pK in protectedKeys:
    del ephysParams[pK]

if USE_NEW_EPHYS_PARAMS:
    p.update(ephysParams)

# p['jEE'] = p['jEE'] * 452 / 560
# p['jEI'] = p['jEI'] * 452 / 560
# p['jIE'] = p['jIE'] * 802 / 541
# p['jII'] = p['jII'] * 802 / 541
# critExc = 0.784 * volt * 452 / 560  # this fraction is to adjust for the new Buono params
# critInh = 0.67625 * volt * 802 / 541

# p['simName'] = p['simName'] + 'UpCrit'

# p['simName'] = 'jercogUpCrit'
# critExc = 0.784 * volt  # this fraction is to adjust for the new Buono params
# critInh = 0.67625 * volt

p['propConnect'] = 1
p['simName'] = 'jercogUpCritFullConn500Units'
p['nUnits'] = 500

# p['propConnect'] = 1
# p['simName'] = 'jercogUpCritFullConn250Units'
# p['nUnits'] = 250

p['recordStateVariables'] = ['v', 'sE', 'sI', 'iAdapt']

indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)

# p['propConnect'] = 0.05
# p['simName'] = 'jercogUpCrit0p05'
# p['jEE'] = p['jEE'] * 0.75
# p['jIE'] = p['jIE'] * 0.75
# p['noiseSigma'] = 0.5 * mV

p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])

JN = JercogNetwork(p)
JN.initialize_network()
JN.initialize_units()
JN.initialize_recurrent_synapses2()

# JN.prepare_upCrit_experiment(minUnits=200, maxUnits=1000, unitSpacing=200, timeSpacing=3000 * ms)
# JN.prepare_upCrit_experiment(minUnits=200, maxUnits=300, unitSpacing=20, timeSpacing=3000 * ms)
JN.prepare_upCrit_experiment(minUnits=30, maxUnits=30, unitSpacing=40, timeSpacing=1000 * ms,
                             startTime=100 * ms, critExc=p['critExc'], critInh=p['critInh'])

JN.create_monitors()
JN.run()
JN.save_results()
JN.save_params()

R = Results(JN.saveName, JN.p['saveFolder'])

R.calculate_spike_rate()
R.calculate_voltage_histogram(removeMode=True)
R.calculate_upstates()
if len(R.ups) > 0:
    R.reshape_upstates()
    R.calculate_FR_in_upstates()
    print('average FR in upstate for Exc: {:.2f}, Inh: {:.2f} '.format(R.upstateFRExc.mean(), R.upstateFRInh.mean()))


fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                         sharex=True)
R.plot_spike_raster(ax1[0])
R.plot_firing_rate(ax1[1])

fig2, ax2 = plt.subplots(3, 1, num=2, figsize=(10, 9), sharex=True, sharey=True)
R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
R.plot_updur_lines(ax2[0])
R.plot_voltage_detail(ax2[1], unitType='Exc', useStateInd=1)
R.plot_updur_lines(ax2[1])
R.plot_voltage_detail(ax2[2], unitType='Inh', useStateInd=0)

fig2c, ax2c = plt.subplots(3, 1, num=3, figsize=(10, 9), sharex=True)

AMPA_E0 = JN.unitsExc.jE[0] * JN.stateMonExc.sE[0, :].T
AMPA_E1 = JN.unitsExc.jE[0] * JN.stateMonExc.sE[1, :].T
AMPA_I0 = JN.unitsInh.jE[0] * JN.stateMonInh.sE[0, :].T

ax2c[0].plot(R.timeArray, AMPA_E0, label='AMPA')
ax2c[1].plot(R.timeArray, AMPA_E1, label='AMPA')
ax2c[2].plot(R.timeArray, AMPA_I0, label='AMPA')

GABA_E0 = JN.unitsExc.jI[0] * JN.stateMonExc.sI[0, :].T
GABA_E1 = JN.unitsExc.jI[0] * JN.stateMonExc.sI[1, :].T
GABA_I0 = JN.unitsInh.jI[0] * JN.stateMonInh.sI[0, :].T

ax2c[0].plot(R.timeArray, GABA_E0, label='GABA')
ax2c[1].plot(R.timeArray, GABA_E1, label='GABA')
ax2c[2].plot(R.timeArray, GABA_I0, label='GABA')

ax2c[0].plot(R.timeArray, JN.stateMonExc.iAdapt[0, :].T, label='iAdapt')
ax2c[1].plot(R.timeArray, JN.stateMonExc.iAdapt[1, :].T, label='iAdapt')
ax2c[2].plot(R.timeArray, JN.stateMonInh.iAdapt[0, :].T, label='iAdapt')
ax2c[2].legend()

# find the average AMPA / GABA current during the first Up state duration

startUpInd = int(R.ups[0] * second / R.p['dt'])
endUpInd = int(R.downs[0] * second / R.p['dt'])

AMPA_E1_avg = AMPA_E1[startUpInd:endUpInd].mean()
GABA_E0_avg = GABA_E0[startUpInd:endUpInd].mean()

print('average AMPA/GABA current during Up state: {:.1f}/{:.1f} pA'.format(AMPA_E1_avg / pA, GABA_E0_avg / pA))
