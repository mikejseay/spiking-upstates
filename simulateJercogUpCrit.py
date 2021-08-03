from brian2 import set_device, defaultclock, ms, second, pA, nA
from params import paramsJercog as p
from params import paramsJercogEphysBuono
from network import JercogNetwork
from results import Results
import matplotlib.pyplot as plt
from generate import normal_positive_weights

# # for using Brian2GENN
# USE_BRIAN2GENN = False
# if USE_BRIAN2GENN:
#     import brian2genn
#     set_device('genn', debug=False)

# for loading from an old results file...

LOAD_PARAMS_FROM_RESULTS = True
targetPath = 'C:/Users/mikejseay/Documents/BrianResults/'
# targetFile = 'classicJercog_5000_0p05_balance_defaultEqual_bimodTest_2021-05-04-15-30_results'
# targetFile = 'jercogDefault_1e3_P05_eqwts_jercog_scale2_smallwts_repro_2021-04-30-10-17_balance_results'
# targetFile = 'classicJercog_1000_0p5_balance_defaultNormal__2021-05-04-22-36_results'  # classic stable low alpha
targetFiles = (
    'classicJercog_5000_0p8_balance_defaultNormal_showBimode_2021-05-10-14-44_results',
    'classicJercog_5000_0p4_balance_defaultNormal_showBimode_2021-05-10-14-44_results',
    'classicJercog_5000_0p2_balance_defaultNormal_showBimode_2021-05-10-14-44_results',
    'classicJercog_5000_0p1_balance_defaultNormal_showBimode_2021-05-10-14-44_results',
    'classicJercog_5000_0p05_balance_defaultNormal_showBimode_2021-05-10-14-44_results',
)

for targetFile in targetFiles:

    if LOAD_PARAMS_FROM_RESULTS:
        OR = Results()
        OR.init_from_file(targetFile, targetPath)
        OR.p['useOldWeightMagnitude'] = False
        p = OR.p.copy()
    else:
        p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
        p['saveWithDate'] = True
        p['propConnect'] = 0.05
        p['nUnits'] = 5e3
        p['useOldWeightMagnitude'] = False
        p['recordStateVariables'] = ['v', 'sE', 'sI', 'iAdapt']

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

    # set which unit indices to record state variables from (e.g. voltage)
    # perhaps 0, 1, 2, 3, 4 and indUnkickedExc, indUnkickedExc - 1, etc
    indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
    indUnkickedInh = int(p['nUnits'] - 1)
    excList = list(range(0, 5, 1))
    unKickedExcList = list(range(indUnkickedExc, indUnkickedExc - 5, -1))
    inhList = list(range(0, 5, 1))
    unKickedInhList = list(range(indUnkickedInh, indUnkickedInh - 5, -1))

    p['indsRecordStateExc'] = []
    p['indsRecordStateInh'] = []

    p['indsRecordStateExc'].extend(excList)
    p['indsRecordStateExc'].extend(unKickedExcList)
    p['indsRecordStateInh'].extend(inhList)
    p['indsRecordStateInh'].extend(unKickedInhList)

    p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
    p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])

    JN = JercogNetwork(p)
    JN.initialize_network()
    JN.initialize_units_twice_kickable2()

    if LOAD_PARAMS_FROM_RESULTS:
        JN.initialize_recurrent_synapses_4bundles_results(OR)
    else:
        JN.initialize_recurrent_synapses_4bundles_modifiable()

    # JN.synapsesEE.jEE = JN.synapsesEE.jEE * 0.96
    # JN.synapsesIE.jIE = JN.synapsesIE.jIE * 0.90
    # JN.synapsesEI.jEI = JN.synapsesEI.jEI * 1.21
    # JN.synapsesII.jII = JN.synapsesII.jII * 1.22

    # JN.synapsesEE.jEE = JN.synapsesEE.jEE[:] * normal_positive_weights(JN.synapsesEE.jEE[:].size, 1, 0.2) * 0.96
    # JN.synapsesIE.jIE = JN.synapsesIE.jIE[:] * normal_positive_weights(JN.synapsesIE.jIE[:].size, 1, 0.2) * 0.90
    # JN.synapsesEI.jEI = JN.synapsesEI.jEI[:] * normal_positive_weights(JN.synapsesEI.jEI[:].size, 1, 0.2) * 1.21
    # JN.synapsesII.jII = JN.synapsesII.jII[:] * normal_positive_weights(JN.synapsesII.jII[:].size, 1, 0.2) * 1.22

    # JN.synapsesEE.jEE = OR.wEE_final.mean() * pA
    # JN.synapsesIE.jIE = OR.wIE_final.mean() * pA
    # JN.synapsesEI.jEI = OR.wEI_final.mean() * pA
    # JN.synapsesII.jII = OR.wII_final.mean() * pA

    # JN.synapsesEE.jEE = OR.wEE_final.mean() * normal_positive_weights(JN.synapsesEE.jEE[:].size, 1, 0.2) * pA
    # JN.synapsesIE.jIE = OR.wIE_final.mean() * normal_positive_weights(JN.synapsesIE.jIE[:].size, 1, 0.2) * pA
    # JN.synapsesEI.jEI = OR.wEI_final.mean() * normal_positive_weights(JN.synapsesEI.jEI[:].size, 1, 0.2) * pA
    # JN.synapsesII.jII = OR.wII_final.mean() * normal_positive_weights(JN.synapsesII.jII[:].size, 1, 0.2) * pA

    # JN.prepare_upCrit_experiment(minUnits=200, maxUnits=1000, unitSpacing=200, timeSpacing=3000 * ms)
    # JN.prepare_upCrit_experiment(minUnits=200, maxUnits=300, unitSpacing=20, timeSpacing=3000 * ms)
    # JN.prepare_upCrit_experiment2(minUnits=50, maxUnits=24, unitSpacing=-25, timeSpacing=4000 * ms,
    #                               startTime=100 * ms, currentAmp=0.98 * nA)
    if LOAD_PARAMS_FROM_RESULTS:
        JN.prepare_upCrit_experiment2(minUnits=p['nUnitsToSpike'], maxUnits=p['nUnitsToSpike'],
                                      unitSpacing=5,  # unitSpacing is a useless input in this context
                                      timeSpacing=p['timeAfterSpiked'],
                                      startTime=p['timeAfterSpiked'],  # p['timeToSpike']
                                      currentAmp=p['spikeInputAmplitude'])
    else:
        JN.prepare_upCrit_experiment2(minUnits=250, maxUnits=250, unitSpacing=5, timeSpacing=1500 * ms,
                                  startTime=100 * ms, currentAmp=0.98 * nA)

    JN.create_monitors()
    JN.run()
    JN.save_results_to_file()
    JN.save_params_to_file()

    R = Results()
    R.init_from_file(JN.saveName, JN.p['saveFolder'])

    R.calculate_PSTH()
    R.calculate_voltage_histogram(removeMode=True, useAllRecordedUnits=True)
    R.calculate_upstates()
    if len(R.ups) > 0:
        R.reshape_upstates()
        R.calculate_FR_in_upstates()
        print('average FR in upstate for Exc: {:.2f}, Inh: {:.2f} '.format(R.upstateFRExcHist.mean(), R.upstateFRInhHist.mean()))

    plt.close('all')
    # fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(5, 5), sharex=True)
    R.plot_spike_raster(ax1[0], downSampleUnits=True)
    R.plot_firing_rate(ax1[1])
    fig1.savefig(targetPath + targetFile + '_spikes.tif')

    test = 0
    # fig2, ax2 = plt.subplots(3, 1, num=2, figsize=(10, 9), sharex=True, sharey=True)
    # R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
    # R.plot_updur_lines(ax2[0])
    # R.plot_voltage_detail(ax2[1], unitType='Exc', useStateInd=1)
    # R.plot_updur_lines(ax2[1])
    # R.plot_voltage_detail(ax2[2], unitType='Inh', useStateInd=0)
    # # R.plot_voltage_histogram_sideways(ax2[0], 'Exc', yScaleLog=True)
    # # R.plot_voltage_histogram_sideways(ax2[2], 'Inh', yScaleLog=True)
    # fig2.savefig(targetPath + targetFile + '.tif')
    #
    # fig2b, ax2b = plt.subplots(1, 1, num=21, figsize=(4, 3))
    # R.plot_voltage_histogram(ax2b, yScaleLog=True)
    # fig2b.savefig(targetPath + targetFile + '_histLog.tif')
    #
    # fig2c, ax2c = plt.subplots(1, 1, num=22, figsize=(4, 3))
    # R.plot_voltage_histogram(ax2c, yScaleLog=False)
    # fig2c.savefig(targetPath + targetFile + '_hist.tif')

    # fig2c, ax2c = plt.subplots(3, 1, num=3, figsize=(10, 9), sharex=True)
    #
    # AMPA_E0 = JN.unitsExc.jE[0] * JN.stateMonExc.sE[0, :].T
    # AMPA_E1 = JN.unitsExc.jE[0] * JN.stateMonExc.sE[1, :].T
    # AMPA_I0 = JN.unitsInh.jE[0] * JN.stateMonInh.sE[0, :].T
    #
    # ax2c[0].plot(R.timeArray, AMPA_E0, label='AMPA')
    # ax2c[1].plot(R.timeArray, AMPA_E1, label='AMPA')
    # ax2c[2].plot(R.timeArray, AMPA_I0, label='AMPA')
    #
    # GABA_E0 = JN.unitsExc.jI[0] * JN.stateMonExc.sI[0, :].T
    # GABA_E1 = JN.unitsExc.jI[0] * JN.stateMonExc.sI[1, :].T
    # GABA_I0 = JN.unitsInh.jI[0] * JN.stateMonInh.sI[0, :].T
    #
    # ax2c[0].plot(R.timeArray, GABA_E0, label='GABA')
    # ax2c[1].plot(R.timeArray, GABA_E1, label='GABA')
    # ax2c[2].plot(R.timeArray, GABA_I0, label='GABA')
    #
    # ax2c[0].plot(R.timeArray, JN.stateMonExc.iAdapt[0, :].T, label='iAdapt')
    # ax2c[1].plot(R.timeArray, JN.stateMonExc.iAdapt[1, :].T, label='iAdapt')
    # ax2c[2].plot(R.timeArray, JN.stateMonInh.iAdapt[0, :].T, label='iAdapt')
    # ax2c[2].legend()
    #
    # # find the average AMPA / GABA current during the first Up state duration
    #
    # startUpInd = int(R.ups[0] * second / R.p['dt'])
    # endUpInd = int(R.downs[0] * second / R.p['dt'])
    #
    # AMPA_E1_avg = AMPA_E1[startUpInd:endUpInd].mean()
    # GABA_E0_avg = GABA_E0[startUpInd:endUpInd].mean()
    #
    # print('average AMPA/GABA current during Up state: {:.1f}/{:.1f} pA'.format(AMPA_E1_avg / pA, GABA_E0_avg / pA))

    del OR
    del JN
    del R

