from brian2 import defaultclock, ms, second, pA, mV
from params import paramsJercog as p
from network import JercogNetwork
from results import Results, convert_sNMDA_to_current_exc_Jercog_CellNotSyn, \
    convert_sNMDA_to_current_inh_Jercog_CellNotSyn
import matplotlib.pyplot as plt

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True
p['dt'] = 0.1 * ms

defaultclock.dt = p['dt']

# NMDA parameters
p['tauRiseNMDA'] = 2. * ms
p['tauFallNMDA'] = 50. * ms
p['alpha'] = 0.5 / ms
p['Mg2'] = 1.

# NMDA gating parameters
p['kSigmoid'] = 0.5
p['vMidSigmoid'] = 14 * mV
p['vStepSigmoid'] = -100 * mV  # having this be very low effectively removes any hard gating effect

# alter AMPA excitation
p['jEE'] = p['jEE'] * 0.6  # 0.6
p['jIE'] = p['jIE'] * 0.6  # 0.6

# alter GABA inhibition
p['jEI'] = p['jEI'] * 1
p['jII'] = p['jII'] * 1

# set the NMDA weight
NMDA_FACTOR = 0.06  # should be double what it was in syn version because it
p['jEE_NMDA'] = NMDA_FACTOR * p['jEE']
p['jIE_NMDA'] = NMDA_FACTOR * p['jIE']

p['propConnect'] = 1
p['simName'] = 'jercogNMDAUpCritFullConn500Units'
p['nUnits'] = 500

p['recordStateVariables'] = ['v', 'sE', 'sI', 'iAdapt', 'sE_NMDA']

indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)

p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])

JN = JercogNetwork(p)
JN.initialize_network()
JN.initialize_units_NMDA2()
JN.initialize_recurrent_excitation_NMDA2()
JN.initialize_recurrent_inhibition()

JN.prepare_upCrit_experiment(minUnits=15, maxUnits=50, unitSpacing=5, timeSpacing=3000 * ms,
                             startTime=100 * ms)

JN.create_monitors()
JN.run_NMDA2()
JN.save_results_to_file()
JN.save_params_to_file()

R = Results(JN.saveName, JN.p['saveFolder'])

R.calculate_PSTH()
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

fig2, ax2 = plt.subplots(3, 1, num=2, figsize=(10, 9), sharex=True)
R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
R.plot_updur_lines(ax2[0])
R.plot_voltage_detail(ax2[1], unitType='Exc', useStateInd=1)
R.plot_updur_lines(ax2[1])
R.plot_voltage_detail(ax2[2], unitType='Inh', useStateInd=0)

AMPA_E0 = JN.unitsExc.jE[0] * JN.stateMonExc.sE[0, :].T / pA
AMPA_E1 = JN.unitsExc.jE[0] * JN.stateMonExc.sE[1, :].T / pA
AMPA_I0 = JN.unitsInh.jE[0] * JN.stateMonInh.sE[0, :].T / pA
GABA_E0 = JN.unitsExc.jI[0] * JN.stateMonExc.sI[0, :].T / pA
GABA_E1 = JN.unitsExc.jI[0] * JN.stateMonExc.sI[1, :].T / pA
GABA_I0 = JN.unitsInh.jI[0] * JN.stateMonInh.sI[0, :].T / pA
NMDA_E0 = convert_sNMDA_to_current_exc_Jercog_CellNotSyn(JN, 0) / pA
NMDA_E1 = convert_sNMDA_to_current_exc_Jercog_CellNotSyn(JN, 1) / pA
NMDA_I0 = convert_sNMDA_to_current_inh_Jercog_CellNotSyn(JN, 0) / pA

fig2c, ax2c = plt.subplots(3, 1, num=3, figsize=(10, 9), sharex=True, sharey=True)

ax2c[0].plot(R.timeArray, AMPA_E0, label='AMPA', lw=.5)
ax2c[1].plot(R.timeArray, AMPA_E1, label='AMPA', lw=.5)
ax2c[2].plot(R.timeArray, AMPA_I0, label='AMPA', lw=.5)
ax2c[0].plot(R.timeArray, NMDA_E0, label='NMDA', lw=.5)
ax2c[1].plot(R.timeArray, NMDA_E1, label='NMDA', lw=.5)
ax2c[2].plot(R.timeArray, NMDA_I0, label='NMDA', lw=.5)
ax2c[0].plot(R.timeArray, AMPA_E0 + NMDA_E0, label='AMPA+NMDA', lw=.5)
ax2c[1].plot(R.timeArray, AMPA_E1 + NMDA_E1, label='AMPA+NMDA', lw=.5)
ax2c[2].plot(R.timeArray, AMPA_I0 + NMDA_I0, label='AMPA+NMDA', lw=.5)
ax2c[0].plot(R.timeArray, GABA_E0, label='GABA', lw=.5)
ax2c[1].plot(R.timeArray, GABA_E1, label='GABA', lw=.5)
ax2c[2].plot(R.timeArray, GABA_I0, label='GABA', lw=.5)
ax2c[0].plot(R.timeArray, JN.stateMonExc.iAdapt[0, :].T / pA, label='iAdapt', lw=.5)
ax2c[1].plot(R.timeArray, JN.stateMonExc.iAdapt[1, :].T / pA, label='iAdapt', lw=.5)
ax2c[2].plot(R.timeArray, JN.stateMonInh.iAdapt[0, :].T / pA, label='iAdapt', lw=.5)

ax2c[2].legend(loc='best')
ax2c[2].set(xlabel='Time (s)', ylabel='Current (pA)', xlim=(0., JN.p['duration'] / second))

# find the average AMPA / GABA current during the first Up state duration

startUpInd = int(R.ups[0] * second / R.p['dt'])
endUpInd = int(R.downs[0] * second / R.p['dt'])

AMPA_E1_avg = AMPA_E1[startUpInd:endUpInd].mean()
NMDA_E1_avg = NMDA_E1[startUpInd:endUpInd].mean()
GABA_E0_avg = GABA_E0[startUpInd:endUpInd].mean()

print('average AMPA/NMDA/GABA current during Up state: {:.1f}/{:.1f}/{:.1f} pA'.format(AMPA_E1_avg, NMDA_E1_avg,
                                                                                       GABA_E0_avg))

# plot voltage histogram

fig2b, ax2b = plt.subplots(1, 1, num=21, figsize=(4, 3))
R.plot_voltage_histogram(ax2b, yScaleLog=True)

# plot the raw seNMDA

sE_NMDA_E0 = JN.stateMonExc.sE_NMDA[0, :].T
sE_NMDA_E1 = JN.stateMonExc.sE_NMDA[1, :].T
sE_NMDA_I0 = JN.stateMonInh.sE_NMDA[0, :].T

fig3, ax3 = plt.subplots(1, 1, num=4, figsize=(10, 9))
ax3.plot(sE_NMDA_E0)
ax3.plot(sE_NMDA_E1)
ax3.plot(sE_NMDA_I0)


