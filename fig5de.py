from brian2 import defaultclock, ms, nA, Hz, seed, second
from params import paramsJercogEphysBuono
from generate import weight_matrix_from_flat_inds_weights
from runner import JercogRunner
from results import Results

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# START OF PARAMS

p = paramsJercogEphysBuono.copy()
rngSeed = 43
defaultclock.dt = p['dt']

# naming / save params
p['nameSuffix'] = 'actualPoisson'
p['saveFolder'] = os.path.join(os.getcwd(), 'results')

# params for modifying cell-intrinsic params (fig 5)
p['propPop2Units'] = 0.1  # percentage of Ex units designated as "Ex+"
p['useSecondPopExc'] = True  # designates a distinct Ex population with different cell-intrinsic parameters

# params for modifying connectivity (fig 6)
p['manipEEConn'] = False  # designates a distinct Ex population and modifies connectivity with it
p['removePropConn'] = 0.1  # proportion of connections between ex- and ex+ to remove
p['compensateEEManipWithInhib'] = False  # whether to compensate inhibition based on connections removed (not used in paper)

# net params
p['nUnits'] = 2e3
p['propConnect'] = 0.25
p['propInh'] = 0.2
p['allowAutapses'] = False

# params for importing the weight matrix (can also generate randomly but not used here)
p['initWeightMethod'] = 'resumePrior'
p['initWeightPrior'] = 'liuEtAlInitialWeights'

# kick params
p['propKickedSpike'] = 0.05  # proportion of units to kick by causing a single spike in them
p['poissonLambda'] = 0.5 * Hz  # 0.2
p['duration'] = 20 * second  # 60

# dt params
p['dtHistPSTH'] = 10 * ms
p['recordAllVoltage'] = True
p['stateVariableDT'] = 1 * ms

if not os.path.exists(p['saveFolder']):
    os.mkdir(p['saveFolder'])

# other params that generally should not be modified
weightMult = 0.82  # multiply init weights by this (makes basin of attraction around upper fixed point more shallow)
overrideBetaAdaptExc = 18 * nA * ms  # override default adaptation strength (10 in params file but makes little diff)
p['kickType'] = 'spike'
p['spikeInputAmplitude'] = 0.98
p['nUnitsToSpike'] = int(np.round(p['propKickedSpike'] * p['nUnits']))
p['nUnitsSecondPopExc'] = int(np.round(p['propPop2Units'] * p['nUnits']))
p['startIndSecondPopExc'] = p['nUnitsToSpike']
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
p['nExc'] = int(p['nUnits'] * (1 - p['propInh']))
p['nInh'] = int(p['nUnits'] * p['propInh'])
p['indsRecordStateExc'].append(int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1))
p['indsRecordStateExc'].append(p['startIndSecondPopExc'])
if p['recordAllVoltage']:
    p['indsRecordStateExc'] = list(range(p['nExc']))
    p['indsRecordStateInh'] = list(range(p['nInh']))
elif 'stateVariableDT' not in p:
    p['stateVariableDT'] = p['dt'].copy()

# END OF PARAMS

# set RNG seeds...
p['rngSeed'] = rngSeed
rng = np.random.default_rng(rngSeed)  # for numpy
seed(rngSeed)  # for Brian... will insert code to set the random number generator seed into the generated code
p['rng'] = rng

JT = JercogRunner(p)  # it's called trainer but it just helps set up the simulation

if p['initWeightMethod'] == 'resumePrior':
    PR = Results()
    PR.init_from_file(p['initWeightPrior'], p['saveFolder'])
    p = dict(list(p.items()) + list(PR.p.items()))
    # p = PR.p.copy()  # note this completely overwrites all settings above
    p['nameSuffix'] = p['initWeightMethod'] + p['nameSuffix']  # a way to remember what it was...
    if 'seed' in p['nameSuffix']:  # this will only work for single digit seeds...
        rngSeed = int(p['nameSuffix'][p['nameSuffix'].find('seed') + 4])
    p['initWeightMethod'] = 'resumePrior'  # and then we put this back...
else:
    PR = None

if p['manipEEConn']:

    startIndExc2 = p['startIndSecondPopExc']
    endIndExc2 = p['startIndSecondPopExc'] + p['nUnitsSecondPopExc']

    E2E1Inds = \
        np.where(
            np.logical_and(PR.preEE >= endIndExc2, np.logical_and(PR.posEE >= startIndExc2, PR.posEE < endIndExc2)))[0]
    E1E2Inds = \
        np.where(
            np.logical_and(PR.posEE >= endIndExc2, np.logical_and(PR.preEE >= startIndExc2, PR.preEE < endIndExc2)))[0]

    removeE2E1Inds = p['rng'].choice(E2E1Inds, int(np.round(E2E1Inds.size * p['removePropConn'])), replace=False)
    removeE1E2Inds = p['rng'].choice(E1E2Inds, int(np.round(E1E2Inds.size * p['removePropConn'])), replace=False)

    removeInds = np.concatenate((removeE2E1Inds, removeE1E2Inds))

    # save some info...
    wEEInit = weight_matrix_from_flat_inds_weights(PR.p['nExc'], PR.p['nExc'], PR.preEE, PR.posEE, PR.wEE_final)
    weightsSaved = PR.wEE_final[removeInds]  # save the weights to be used below

    PR.preEE = np.delete(PR.preEE, removeInds, None)
    PR.posEE = np.delete(PR.posEE, removeInds, None)
    PR.wEE_final = np.delete(PR.wEE_final, removeInds, None)
    wEEFinal = weight_matrix_from_flat_inds_weights(PR.p['nExc'], PR.p['nExc'], PR.preEE, PR.posEE, PR.wEE_final)

    if p['compensateEEManipWithInhib']:
        # idea here is that by deleting excitatory connections only, we are upsetting the E/I balance
        # we can calculate the initial E/I balance (basically, summing the rows of wEE and also summing the rows of wEI)
        # then we can compare it to the E/I balance afterward
        wEIInit = weight_matrix_from_flat_inds_weights(PR.p['nInh'], PR.p['nExc'], PR.preEI, PR.posEI, PR.wEI_final)

        # turn the final into a matrix also...
        sumExcOntoExcInit = np.nansum(wEEInit, 0)
        sumInhOntoExcInit = np.nansum(wEIInit, 0)

        sumExcOntoExcFinal = np.nansum(wEEFinal, 0)

        # calculate a ratio of the change
        changeInExcAmount = sumExcOntoExcFinal / sumExcOntoExcInit
        changeInEx2 = changeInExcAmount[startIndExc2:endIndExc2].mean()
        changeInEx1 = changeInExcAmount[endIndExc2:].mean()

        # decrease inhibition appropriately
        wEICompensate = wEIInit.copy()
        wEICompensate[:, startIndExc2:endIndExc2] = wEICompensate[:, startIndExc2:endIndExc2] * changeInEx2
        wEICompensate[:, endIndExc2:] = wEICompensate[:, endIndExc2:] * changeInEx1
        PR.wEI_final = wEICompensate[PR.preEI, PR.posEI]

JT.p['betaAdaptExc'] = overrideBetaAdaptExc  # override...
JT.p['betaAdaptExc2'] = overrideBetaAdaptExc  # override...
JT.set_up_network(priorResults=PR, recordAllVoltage=p['recordAllVoltage'])
JT.initialize_weight_matrices()

# manipulate the weights to make things less stable
# here i will do so by decreasing all the weights by a constant multiplier
JT.wEE_init = JT.wEE_init * weightMult
JT.wEI_init = JT.wEI_init * weightMult
JT.wIE_init = JT.wIE_init * weightMult
JT.wII_init = JT.wII_init * weightMult

JT.run()
JT.save_params()
JT.save_results()

R = Results()
R.init_from_file(JT.saveName, JT.p['saveFolder'])

R.calculate_PSTH()
R.calculate_voltage_histogram(useAllRecordedUnits=True)
R.calculate_upstates()
R.calculate_upFR_units()
R.calculate_upCorr_units()

fig1, ax1 = plt.subplots(3, 1, num=1, figsize=(16, 9), sharex=True)

if p['useSecondPopExc'] or p['manipEEConn']:
    useColor = 'royalblue'
else:
    useColor = 'cyan'

R.plot_voltage_detail(ax1[0], unitType='Exc', useStateInd=0)
R.plot_voltage_detail(ax1[1], unitType='Exc', useStateInd=1, overrideColor=useColor)

R.plot_firing_rate(ax1[2])
ax1[2].set(xlabel='Time (seconds)', ylabel='FR (Hz)', ylim=(0, 30))

fig2, ax2 = plt.subplots(figsize=(3.25, 4))

frEx2 = R.upstateFRExcUnits[:, R.p['nUnitsToSpike']:(R.p['nUnitsToSpike'] + R.p['nUnitsSecondPopExc'])].mean(0)  # secondary pop
frEx1 = R.upstateFRExcUnits[:, (R.p['nUnitsToSpike'] + R.p['nUnitsSecondPopExc']):].mean(0)  # normal pop

frEx2DF = pd.DataFrame(frEx2, columns=('FR',))
frEx1DF = pd.DataFrame(frEx1, columns=('FR',))

frEx2DF['unitType'] = 'Ex2'
frEx1DF['unitType'] = 'Ex1'

frDF = pd.concat((frEx2DF, frEx1DF, ))

colors = ["#00FFFF", "#4E69B2"]
customPalette = sns.set_palette(sns.color_palette(colors))

sns.violinplot(x='unitType', y='FR', data=frDF, width=.6, palette=customPalette, order=('Ex1', 'Ex2'), ax=ax2)

# up state correlation

# calculating 3 submatrix averages with start and stop indices
# pop order is input, Ex2, Ex1

rhoUpExc = R.rhoUpExc.copy()
rhoUpExc[np.diag_indices_from(rhoUpExc)] = np.nan

popLabels = ['Inp', 'Ex2', 'Ex1']

sI = [0,
      R.p['nUnitsToSpike'],
      (R.p['nUnitsToSpike'] + R.p['nUnitsSecondPopExc'])]
eI = [R.p['nUnitsToSpike'],
      (R.p['nUnitsToSpike'] + R.p['nUnitsSecondPopExc']),
      -1]

assert(len(sI) == len(eI))

nPops = len(sI)

sectionCorrMean = np.full((nPops, nPops), np.nan)
sectionCorrStd = np.full((nPops, nPops), np.nan)

meanCompareLabels = []
seqLst = []
for p1I in range(nPops):
    for p2I in range(nPops):
        if p2I > p1I:
            continue
        rhoSub = rhoUpExc[sI[p1I]:eI[p1I], sI[p2I]:eI[p2I]]
        rhoSubNN = rhoSub[~np.isnan(rhoSub)]
        sectionCorrMean[p1I, p2I] = np.nanmean(rhoSubNN)
        sectionCorrStd[p1I, p2I] = np.nanstd(rhoSubNN)
        meanCompareLabels.append(popLabels[p1I] + '-' + popLabels[p2I])
        dfTmp = pd.DataFrame(rhoSubNN, columns=('Correlation',))
        dfTmp['Comparison'] = popLabels[p1I] + '-' + popLabels[p2I]
        seqLst.append(dfTmp)

corrCompareDF = pd.concat(seqLst)

colors = ["#00FFFF", "#6A0DAD", "#4E69B2"]
customPalette = sns.set_palette(sns.color_palette(colors))

fig3, ax = plt.subplots()
sns.boxplot(x='Comparison', y='Correlation', data=corrCompareDF, order=('Ex1-Ex1', 'Ex1-Ex2', 'Ex2-Ex2'), palette=customPalette,)
ax.set_xticklabels(('(Ex-/Ex-)', '(Ex-/Ex+)', '(Ex+/Ex+)'))
ax.set_xlabel('')

SAVE_PLOT = True
if SAVE_PLOT:
    fig1.savefig('fig5d.pdf', transparent=True)
    fig2.savefig('fig5e.pdf', transparent=True)
    fig3.savefig('fig5f.pdf', transparent=True)
