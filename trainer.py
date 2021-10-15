"""
a class that consumes a network and some parameters,
then uses them to run many trials,
modifying weights in between each trial...
"""

import os
from datetime import datetime
from itertools import product
import pickle
import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from brian2 import ms, nA, pA, Hz, second, mV
from network import JercogEphysNetwork, JercogNetwork
from results import ResultsEphys, Results
from generate import lognorm_weights, norm_weights, adjacency_matrix_from_flat_inds, \
    weight_matrix_from_flat_inds_weights


class JercogTrainer(object):

    def __init__(self, p):
        self.p = p
        self.p['initTime'] = datetime.now().strftime('%Y-%m-%d-%H-%M')
        # construct name from parameterSet, nUnits, propConnect, useRule, initWeightMethod, nameSuffix, initTime
        self.saveName = '_'.join((p['paramSet'], str(int(p['nUnits'])), str(p['propConnect']).replace('.', 'p'),
                                  p['useRule'], p['initWeightMethod'], p['nameSuffix'], p['initTime']))

    def calculate_unit_thresh_and_gain(self):
        # first quickly run an EPhys experiment with the given params to calculate the thresh and gain
        pForEphys = self.p.copy()
        if 'iExtRange' not in pForEphys:
            pForEphys['propInh'] = 0.5
            pForEphys['duration'] = 250 * ms
            pForEphys['iExtRange'] = np.linspace(0, .3, 31) * nA
        JEN = JercogEphysNetwork(pForEphys)
        JEN.build_classic()
        JEN.run()
        RE = ResultsEphys()
        RE.init_from_network_object(JEN)
        RE.calculate_thresh_and_gain()

        # self.p['threshExc'] = RE.threshExc
        # self.p['threshInh'] = RE.threshInh
        # self.p['gainExc'] = RE.gainExc
        # self.p['gainInh'] = RE.gainInh

        # testing an experimental modification to these values...
        self.p['threshExc'] = RE.threshExc  # / 13
        self.p['threshInh'] = RE.threshInh  # / 13
        self.p['gainExc'] = RE.gainExc
        self.p['gainInh'] = RE.gainInh

    def set_up_network(self, priorResults=None):
        # set up network, experiment, and start recording
        JN = JercogNetwork(self.p)
        JN.initialize_network()
        JN.initialize_units_twice_kickable2()

        if self.p['kickType'] == 'kick':
            JN.set_kicked_units(onlyKickExc=self.p['onlyKickExc'])
        elif self.p['kickType'] == 'spike':
            JN.prepare_upCrit_experiment2(minUnits=self.p['nUnitsToSpike'], maxUnits=self.p['nUnitsToSpike'],
                                          unitSpacing=5,  # unitSpacing is a useless input in this context
                                          timeSpacing=self.p['timeAfterSpiked'], startTime=self.p['timeToSpike'],
                                          currentAmp=self.p['spikeInputAmplitude'])
        if priorResults is not None:
            JN.initialize_recurrent_synapses_4bundles_results(priorResults)
        else:
            JN.initialize_recurrent_synapses_4bundles_modifiable()

        JN.create_monitors()

        # by multiplying the proposed weight change by the "scale factor"
        # we take into account that
        # we are modifying a bundle of weights that might be "thicker" pre-synaptically
        # and so a small change results in larger total change in the sum of weights onto
        # each post-synaptic unit

        # let's turn off the scaling to see what happens...
        if self.p['disableWeightScaling']:
            JN.p['wEEScale'] = 1
            JN.p['wIEScale'] = 1
            JN.p['wEIScale'] = 1
            JN.p['wIIScale'] = 1

        self.JN = JN

    def set_up_network_upCrit(self, priorResults=None, recordAllVoltage=False):
        JN = JercogNetwork(self.p)
        JN.initialize_network()
        JN.initialize_units_twice_kickable2()

        if self.p['kickType'] == 'kick':
            JN.set_kicked_units(onlyKickExc=self.p['onlyKickExc'])
        elif self.p['kickType'] == 'spike':
            JN.prepare_upCrit_experiment2(minUnits=self.p['nUnitsToSpike'], maxUnits=self.p['nUnitsToSpike'],
                                          unitSpacing=5,  # unitSpacing is a useless input in this context
                                          timeSpacing=self.p['timeAfterSpiked'], startTime=self.p['timeToSpike'],
                                          currentAmp=self.p['spikeInputAmplitude'])

        if priorResults is not None:
            JN.initialize_recurrent_synapses_4bundles_results(priorResults)
        else:
            JN.initialize_recurrent_synapses_4bundles_modifiable()

        if recordAllVoltage:
            JN.create_monitors_allVoltage()
        else:
            JN.create_monitors()

        if self.p['disableWeightScaling']:
            JN.p['wEEScale'] = 1
            JN.p['wIEScale'] = 1
            JN.p['wEIScale'] = 1
            JN.p['wIIScale'] = 1

        self.JN = JN

    def set_up_network_Poisson(self, priorResults=None):
        # set up network, experiment, and start recording
        JN = JercogNetwork(self.p)
        JN.initialize_network()
        JN.initialize_units_twice_kickable2()
        JN.prepare_upPoisson_experiment(poissonLambda=self.p['poissonLambda'],
                                        duration=self.p['duration'],
                                        spikeUnits=self.p['nUnitsToSpike'],
                                        rng=self.p['rng'])
        if priorResults is not None:
            JN.initialize_recurrent_synapses_4bundles_results(priorResults)
        else:
            JN.initialize_recurrent_synapses_4bundles_modifiable()
        JN.create_monitors()

        self.JN = JN

    def initalize_history_variables(self):
        # initialize history variables
        self.trialUpFRExc = np.empty((self.p['nTrials'],))
        self.trialUpFRInh = np.empty((self.p['nTrials'],))
        self.trialUpDur = np.empty((self.p['nTrials'],))
        self.trialwEE = np.empty((self.p['nTrials'],))
        self.trialwIE = np.empty((self.p['nTrials'],))
        self.trialwEI = np.empty((self.p['nTrials'],))
        self.trialwII = np.empty((self.p['nTrials'],))
        self.trialUpFRExcUnits = np.empty((self.p['nTrials'], self.p['nExc']))
        self.trialUpFRInhUnits = np.empty((self.p['nTrials'], self.p['nInh']))

        # for cross-homeo with weight changes customized to the unit, the same weight change
        # is applied across all incoming presynaptic synapses for each post-synaptic unit
        # so the dW arrays are equal in size to the post-synaptic population
        self.trialdwEEUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
        self.trialdwEIUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
        self.trialdwIEUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')
        self.trialdwIIUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

        self.trialMAdwEE = np.empty((self.p['nTrials']), dtype='float32')
        self.trialMAdwIE = np.empty((self.p['nTrials']), dtype='float32')
        self.trialMAdwEI = np.empty((self.p['nTrials']), dtype='float32')
        self.trialMAdwII = np.empty((self.p['nTrials']), dtype='float32')

        # in this approach we actually save the trials DURING learning
        # i would say start with the spike raster + FR + voltage of two units
        if self.p['recordMovieVariables']:
            # voltage
            nSpecialTrials = len(self.p['saveTrials'])
            self.frameMult = int(self.p['downSampleVoltageTo'] / self.p['dt'])  # how much to downsample in time
            nTimePoints = int(self.p['duration'] / self.p['dt'] / self.frameMult)
            timeFull = np.arange(0, self.p['duration'] / second, self.p['dt'] / second)
            self.selectTrialT = timeFull[::self.frameMult]
            self.selectTrialVExc = np.empty((nSpecialTrials, nTimePoints), dtype='float32')
            self.selectTrialVInh = np.empty((nSpecialTrials, nTimePoints), dtype='float32')

            # FR
            dtHist = float(10 * ms)
            histCenters = np.arange(0 + dtHist / 2, float(self.p['duration']) - dtHist / 2, dtHist)
            self.selectTrialFRExc = np.empty((nSpecialTrials, histCenters.size), dtype='float32')
            self.selectTrialFRInh = np.empty((nSpecialTrials, histCenters.size), dtype='float32')

            # spiking
            self.selectTrialSpikeExcI = np.empty((nSpecialTrials,), dtype=object)
            self.selectTrialSpikeExcT = np.empty((nSpecialTrials,), dtype=object)
            self.selectTrialSpikeInhI = np.empty((nSpecialTrials,), dtype=object)
            self.selectTrialSpikeInhT = np.empty((nSpecialTrials,), dtype=object)

        # in another approach, we can simply reinstate the network from a certain point in time
        # arrays wil be about 15 x nUnits x nUnits x pConn.... which is huge...

    def initialize_weight_matrices(self):

        netSizeConnNorm = 500 / (self.p['nUnits'] * self.p['propConnect'])

        if self.p['initWeightMethod'] == 'resumePrior':
            self.wEE_init = self.JN.synapsesEE.jEE[:]
            self.wIE_init = self.JN.synapsesIE.jIE[:]
            self.wEI_init = self.JN.synapsesEI.jEI[:]
            self.wII_init = self.JN.synapsesII.jII[:]
        elif self.p['initWeightMethod'] == 'monolithic':
            self.wEE_init = self.JN.unitsExc.jE[0]
            self.wIE_init = self.JN.unitsInh.jE[0]
            self.wEI_init = self.JN.unitsExc.jI[0]
            self.wII_init = self.JN.unitsInh.jI[0]
        elif self.p['initWeightMethod'] == 'defaultEqual':
            self.wEE_init = self.JN.synapsesEE.jEE[:]
            self.wIE_init = self.JN.synapsesIE.jIE[:]
            self.wEI_init = self.JN.synapsesEI.jEI[:]
            self.wII_init = self.JN.synapsesII.jII[:]
        elif self.p['initWeightMethod'] == 'defaultNormal':
            self.wEE_init = self.JN.synapsesEE.jEE[:] * norm_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2,
                                                                     rng=self.p['rng'])
            self.wIE_init = self.JN.synapsesIE.jIE[:] * norm_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2,
                                                                     rng=self.p['rng'])
            self.wEI_init = self.JN.synapsesEI.jEI[:] * norm_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2,
                                                                     rng=self.p['rng'])
            self.wII_init = self.JN.synapsesII.jII[:] * norm_weights(self.JN.synapsesII.jII[:].size, 1, 0.2,
                                                                     rng=self.p['rng'])
        elif self.p['initWeightMethod'] == 'defaultNormalScaled':
            self.wEE_init = self.JN.synapsesEE.jEE[:] * norm_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2,
                                                                     rng=self.p['rng']) * self.p['jEEScaleRatio']
            self.wIE_init = self.JN.synapsesIE.jIE[:] * norm_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2,
                                                                     rng=self.p['rng']) * self.p['jIEScaleRatio']
            self.wEI_init = self.JN.synapsesEI.jEI[:] * norm_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2,
                                                                     rng=self.p['rng']) * self.p['jEIScaleRatio']
            self.wII_init = self.JN.synapsesII.jII[:] * norm_weights(self.JN.synapsesII.jII[:].size, 1, 0.2,
                                                                     rng=self.p['rng']) * self.p['jIIScaleRatio']
        elif self.p['initWeightMethod'] == 'defaultUniform':
            self.wEE_init = self.p['rng'].rand(self.JN.synapsesEE.jEE[:].size) * 2 * self.JN.synapsesEE.jEE[0]
            self.wIE_init = self.p['rng'].rand(self.JN.synapsesIE.jIE[:].size) * 2 * self.JN.synapsesIE.jIE[0]
            self.wEI_init = self.p['rng'].rand(self.JN.synapsesEI.jEI[:].size) * 2 * self.JN.synapsesEI.jEI[0]
            self.wII_init = self.p['rng'].rand(self.JN.synapsesII.jII[:].size) * 2 * self.JN.synapsesII.jII[0]
        elif self.p['initWeightMethod'] == 'randomUniform':
            self.wEE_init = (100 + 100 * self.p['rng'].rand(self.JN.synapsesEE.jEE[:].size)) * netSizeConnNorm * pA
            self.wIE_init = (100 + 100 * self.p['rng'].rand(self.JN.synapsesIE.jIE[:].size)) * netSizeConnNorm * pA
            self.wEI_init = (100 + 100 * self.p['rng'].rand(self.JN.synapsesEI.jEI[:].size)) * netSizeConnNorm * pA
            self.wII_init = (100 + 100 * self.p['rng'].rand(self.JN.synapsesII.jII[:].size)) * netSizeConnNorm * pA
        elif self.p['initWeightMethod'] == 'randomUniformMid':
            self.wEE_init = (80 + 80 * self.p['rng'].rand(self.JN.synapsesEE.jEE[:].size)) * netSizeConnNorm * pA
            self.wIE_init = (80 + 80 * self.p['rng'].rand(self.JN.synapsesIE.jIE[:].size)) * netSizeConnNorm * pA
            self.wEI_init = (80 + 80 * self.p['rng'].rand(self.JN.synapsesEI.jEI[:].size)) * netSizeConnNorm * pA
            self.wII_init = (80 + 80 * self.p['rng'].rand(self.JN.synapsesII.jII[:].size)) * netSizeConnNorm * pA
        elif self.p['initWeightMethod'] == 'randomUniformHighUnequal':
            wEE_mean = 90 + 20 * self.p['rng'].rand() * netSizeConnNorm
            wIE_mean = 90 + 20 * self.p['rng'].rand() * netSizeConnNorm
            wEI_mean = 90 + 20 * self.p['rng'].rand() * netSizeConnNorm
            wII_mean = 90 + 20 * self.p['rng'].rand() * netSizeConnNorm
            wEE_std = 35 + 20 * self.p['rng'].rand() * netSizeConnNorm
            wIE_std = 35 + 20 * self.p['rng'].rand() * netSizeConnNorm
            wEI_std = 35 + 20 * self.p['rng'].rand() * netSizeConnNorm
            wII_std = 35 + 20 * self.p['rng'].rand() * netSizeConnNorm
            self.wEE_init = (wEE_mean + wEE_std * self.p['rng'].rand(self.JN.synapsesEE.jEE[:].size)) * pA
            self.wIE_init = (wIE_mean + wIE_std * self.p['rng'].rand(self.JN.synapsesIE.jIE[:].size)) * pA
            self.wEI_init = (wEI_mean + wEI_std * self.p['rng'].rand(self.JN.synapsesEI.jEI[:].size)) * pA
            self.wII_init = (wII_mean + wII_std * self.p['rng'].rand(self.JN.synapsesII.jII[:].size)) * pA
        elif self.p['initWeightMethod'] == 'randomUniformMidUnequal':
            wEE_mean = 80 + 20 * self.p['rng'].rand() * netSizeConnNorm
            wIE_mean = 80 + 20 * self.p['rng'].rand() * netSizeConnNorm
            wEI_mean = 80 + 20 * self.p['rng'].rand() * netSizeConnNorm
            wII_mean = 80 + 20 * self.p['rng'].rand() * netSizeConnNorm
            wEE_std = 20 + 10 * self.p['rng'].rand() * netSizeConnNorm
            wIE_std = 20 + 10 * self.p['rng'].rand() * netSizeConnNorm
            wEI_std = 20 + 10 * self.p['rng'].rand() * netSizeConnNorm
            wII_std = 20 + 10 * self.p['rng'].rand() * netSizeConnNorm
            self.wEE_init = (wEE_mean + wEE_std * self.p['rng'].rand(self.JN.synapsesEE.jEE[:].size)) * pA
            self.wIE_init = (wIE_mean + wIE_std * self.p['rng'].rand(self.JN.synapsesIE.jIE[:].size)) * pA
            self.wEI_init = (wEI_mean + wEI_std * self.p['rng'].rand(self.JN.synapsesEI.jEI[:].size)) * pA
            self.wII_init = (wII_mean + wII_std * self.p['rng'].rand(self.JN.synapsesII.jII[:].size)) * pA
        elif self.p['initWeightMethod'] == 'randomUniformLow':
            self.wEE_init = (30 + 30 * self.p['rng'].rand(self.JN.synapsesEE.jEE[:].size)) * netSizeConnNorm * pA
            self.wIE_init = (30 + 30 * self.p['rng'].rand(self.JN.synapsesIE.jIE[:].size)) * netSizeConnNorm * pA
            self.wEI_init = (30 + 30 * self.p['rng'].rand(self.JN.synapsesEI.jEI[:].size)) * netSizeConnNorm * pA
            self.wII_init = (30 + 30 * self.p['rng'].rand(self.JN.synapsesII.jII[:].size)) * netSizeConnNorm * pA
        elif self.p['initWeightMethod'] == 'randomUniformSaray':
            self.wEE_init = (30 + 30 * self.p['rng'].rand(self.JN.synapsesEE.jEE[:].size)) * netSizeConnNorm * pA
            self.wIE_init = (60 + 60 * self.p['rng'].rand(self.JN.synapsesIE.jIE[:].size)) * netSizeConnNorm * pA
            self.wEI_init = (30 + 30 * self.p['rng'].rand(self.JN.synapsesEI.jEI[:].size)) * netSizeConnNorm * pA
            self.wII_init = (30 + 30 * self.p['rng'].rand(self.JN.synapsesII.jII[:].size)) * netSizeConnNorm * pA
        elif self.p['initWeightMethod'] == 'randomUniformSarayMid':
            self.wEE_init = (60 + 60 * self.p['rng'].rand(self.JN.synapsesEE.jEE[:].size)) * netSizeConnNorm * pA
            self.wIE_init = (90 + 90 * self.p['rng'].rand(self.JN.synapsesIE.jIE[:].size)) * netSizeConnNorm * pA
            self.wEI_init = (60 + 60 * self.p['rng'].rand(self.JN.synapsesEI.jEI[:].size)) * netSizeConnNorm * pA
            self.wII_init = (60 + 60 * self.p['rng'].rand(self.JN.synapsesII.jII[:].size)) * netSizeConnNorm * pA
        elif self.p['initWeightMethod'] == 'randomUniformSarayHigh':
            self.wEE_init = (80 + 60 * self.p['rng'].rand(self.JN.synapsesEE.jEE[:].size)) * netSizeConnNorm * pA
            self.wIE_init = (100 + 60 * self.p['rng'].rand(self.JN.synapsesIE.jIE[:].size)) * netSizeConnNorm * pA
            self.wEI_init = (80 + 60 * self.p['rng'].rand(self.JN.synapsesEI.jEI[:].size)) * netSizeConnNorm * pA
            self.wII_init = (80 + 60 * self.p['rng'].rand(self.JN.synapsesII.jII[:].size)) * netSizeConnNorm * pA
        elif self.p['initWeightMethod'] == 'randomUniformSarayHigh5e3p02Converge':
            wEE_mean = 60  # 62  # 64  # 66  # 67
            wIE_mean = 45  # 50  # 58  # 54  # 60
            wEI_mean = 52  # 53  # 48  # 61  # 62
            wII_mean = 27  # 33  # 32  # 40  # 50
            self.wEE_init = wEE_mean * norm_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * norm_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * norm_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * norm_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessGoodWeights2e3p025':
            wEE_mean = 120
            wIE_mean = 90
            wEI_mean = 104
            wII_mean = 54
            self.wEE_init = wEE_mean * np.ones(self.JN.synapsesEE.jEE[:].size) * pA
            self.wIE_init = wIE_mean * np.ones(self.JN.synapsesIE.jIE[:].size) * pA
            self.wEI_init = wEI_mean * np.ones(self.JN.synapsesEI.jEI[:].size) * pA
            self.wII_init = wII_mean * np.ones(self.JN.synapsesII.jII[:].size) * pA
        elif self.p['initWeightMethod'] == 'guessGoodWeights2e3p025Normal':
            wEE_mean = 120
            wIE_mean = 90
            wEI_mean = 104
            wII_mean = 54
            self.wEE_init = wEE_mean * norm_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * norm_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * norm_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * norm_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessGoodWeights2e3p025LogNormal':
            wEE_mean = 120
            wIE_mean = 90
            wEI_mean = 104
            wII_mean = 54
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessGoodWeights2e3p1LogNormal':
            wEE_mean = 30
            wIE_mean = 22.5
            wEI_mean = 26
            wII_mean = 13.5
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessGoodWeights5e3p02LogNormal':
            wEE_mean = 60
            wIE_mean = 45
            wEI_mean = 52
            wII_mean = 27
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessLowWeights2e3p025LogNormal':
            wEE_mean = 90
            wIE_mean = 67.5
            wEI_mean = 104
            wII_mean = 54
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size,
                                                       rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size,
                                                       rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size,
                                                       rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size,
                                                       rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessLowWeights2e3p025LogNormal2':
            wEE_mean = 80
            wIE_mean = 70
            wEI_mean = 130
            wII_mean = 50
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size,
                                                       rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size,
                                                       rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size,
                                                       rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size,
                                                       rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessUpperLeftWeights2e3p025LogNormal':
            wEE_mean = 125.9
            wIE_mean = 94.6
            wEI_mean = 82.4
            wII_mean = 35.5
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size,
                                                       rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size,
                                                       rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size,
                                                       rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size,
                                                       rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessLowerRightWeights2e3p025LogNormal':
            wEE_mean = 107
            wIE_mean = 74.8
            wEI_mean = 178.5
            wII_mean = 93.4
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size,
                                                       rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size,
                                                       rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size,
                                                       rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size,
                                                       rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessZeroActivityWeights2e3p025':
            wEE_mean = 62
            wIE_mean = 62
            wEI_mean = 250
            wII_mean = 250
            self.wEE_init = wEE_mean * norm_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * norm_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * norm_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * norm_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessZeroActivityWeights2e3p025LogNormal':
            wEE_mean = 62
            wIE_mean = 62
            wEI_mean = 250
            wII_mean = 250
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessHighActivityWeights2e3p025LogNormal':
            wEE_mean = 187
            wIE_mean = 187
            wEI_mean = 83
            wII_mean = 83
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessLowActivityWeights2e3p025':
            wEE_mean = 114
            wIE_mean = 82
            wEI_mean = 78
            wII_mean = 20
            self.wEE_init = wEE_mean * norm_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * norm_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * norm_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * norm_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono2Weights2e3p025LogNormal':
            wEE_mean = 110
            wIE_mean = 81
            wEI_mean = 144
            wII_mean = 89
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono2Weights2e3p025LogNormal2':
            wEE_mean = 100
            wIE_mean = 80
            wEI_mean = 160
            wII_mean = 85
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono2Weights2e3p025LogNormal3':
            wEE_mean = 94
            wIE_mean = 89
            wEI_mean = 184
            wII_mean = 75
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono5Weights2e3p025LogNormal':
            wEE_mean = 133
            wIE_mean = 99
            wEI_mean = 159
            wII_mean = 91
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono5Weights2e3p025LogNormal2':
            wEE_mean = 133 * 1.52
            wIE_mean = 99 * 1.52
            wEI_mean = 159 * 1.52
            wII_mean = 91 * 1.52
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size,
                                                       rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size,
                                                       rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size,
                                                       rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size,
                                                       rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono5Weights2e3p025LogNormal3':
            wEE_mean = 205.5  # 206
            wIE_mean = 163
            wEI_mean = 300
            wII_mean = 150
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono5Weights2e3p025Trained':
            wEE_mean = 311.4
            wIE_mean = 388.2
            wEI_mean = 222.7
            wII_mean = 195.8
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono5Weights2e3p025Beta10':
            wEE_mean = 298.35
            wIE_mean = 324
            wEI_mean = 429
            wII_mean = 440
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono5Weights2e3p025Fresh':
            wEE_mean = 276 * 1.13
            wIE_mean = 333
            wEI_mean = 361
            wII_mean = 402
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono6Weights2e3p025Beta10':
            wEE_mean = 288
            wIE_mean = 320
            wEI_mean = 461
            wII_mean = 457
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono7Weights2e3p025':
            wEE_mean = 289
            wIE_mean = 323
            wEI_mean = 459
            wII_mean = 444
            self.wEE_init = wEE_mean * norm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * norm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * norm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * norm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono7Weights2e3p025SlightLow':
            wEE_mean = 289 * 0.75
            wIE_mean = 323 * 0.75
            wEI_mean = 459 * 0.75
            wII_mean = 444 * 0.75
            self.wEE_init = wEE_mean * norm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * norm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * norm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * norm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono7Weights2e3p025SlightLowTuned':
            wEE_mean = 249.28 * .8  # * (1 + np.random.normal(0, 0.03, 1)[0])
            wIE_mean = 264.72 * .8  # * (1 + np.random.normal(0, 0.03, 1)[0])
            wEI_mean = 303.14 * .8  # * (1 + np.random.normal(0, 0.03, 1)[0])
            wII_mean = 282.64 * .8  # * (1 + np.random.normal(0, 0.03, 1)[0])
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono7Weights2e3p025Low':
            wEE_mean = 285 / 2
            wIE_mean = 321 / 2
            wEI_mean = 470 / 2
            wII_mean = 452 / 2
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono6Weights2e3p05Beta10Start':
            wEE_mean = 298 / 2
            wIE_mean = 325 / 2
            wEI_mean = 455 / 2
            wII_mean = 448 / 2
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono6Weights2e3p05Beta10':
            wEE_mean = 150
            wIE_mean = 169
            wEI_mean = 215
            wII_mean = 212
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono4Weights2e3p025Normal':
            wEE_mean = 237  # 179.73
            wIE_mean = 248  # 127.32
            wEI_mean = 331  # 250.38
            wII_mean = 327  # 119.28
            self.wEE_init = wEE_mean * norm_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * norm_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * norm_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * norm_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono4Weights2e3p025LogNormal':
            wEE_mean = 257  # 179.73
            wIE_mean = 180  # 127.32
            wEI_mean = 383  # 250.38
            wII_mean = 170  # 119.28
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono4Weights2e3p025LogNormal2':
            wEE_mean = 223  # 179.73
            wIE_mean = 315  # 127.32
            wEI_mean = 321  # 250.38
            wII_mean = 448  # 119.28
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono4Weights2e3p025LogNormal3':
            wEE_mean = 225  # 275  # 179.73
            wIE_mean = 185  # 185  # 127.32
            wEI_mean = 347  # 347  # 250.38
            wII_mean = 168  # 168  # 119.28
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono4Weights2e3p05LogNormal3':
            wEE_mean = 250 / 2  # 275  # 179.73
            wIE_mean = 185 / 2  # 185  # 127.32
            wEI_mean = 347 / 2  # 347  # 250.38
            wII_mean = 168 / 2  # 168  # 119.28
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono4Weights2e3p025LogNormalStart':
            wEE_mean = 217  # 217  # 179.73
            wIE_mean = 156  # 156  # 127.32
            wEI_mean = 344  # 344  # 250.38
            wII_mean = 150  # 150  # 119.28
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono4Weights2e3p025LogNormalStart2':
            wEE_mean = 186  # 217  # 179.73
            wIE_mean = 134  # 156  # 127.32
            wEI_mean = 288  # 344  # 250.38
            wII_mean = 131  # 150  # 119.28
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessBuono2Weights2e3p025Normal3':
            wEE_mean = 94
            wIE_mean = 89
            wEI_mean = 184
            wII_mean = 75
            self.wEE_init = wEE_mean * norm_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * norm_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * norm_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * norm_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng']) * pA

        elif self.p['initWeightMethod'][:4] == 'seed':
            excMeanWeightsPossible = (75, 112.5, 150)
            inhMeanWeightsPossible = (700, 450, 200)
            # mappingStringSeq = (('wIE', 'wEE'), ('wII', 'wEI'),)
            excWeightTupleList = list(product(excMeanWeightsPossible, excMeanWeightsPossible))
            inhWeightTupleList = list(product(inhMeanWeightsPossible, inhMeanWeightsPossible))
            useSeed = int(self.p['initWeightMethod'][-1])  # should be a value 0-8
            wIE_mean, wEE_mean = excWeightTupleList[useSeed]
            wII_mean, wEI_mean = inhWeightTupleList[useSeed]
            self.wEE_init = wEE_mean * lognorm_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognorm_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognorm_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognorm_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA

    def run(self):

        t0_overall = datetime.now()

        # for readability
        p = self.p
        JN = self.JN

        # initalize variables to represent the rolling average firing rates of the Exc and Inh units
        # we start at None because this is undefined, and we will initialize at the exact value of the first UpFR
        movAvgUpFRExc = None
        movAvgUpFRInh = None
        movAvgUpFRExcUnits = None
        movAvgUpFRInhUnits = None

        wEE = self.wEE_init.copy()
        wEI = self.wEI_init.copy()
        wIE = self.wIE_init.copy()
        wII = self.wII_init.copy()

        # store the network state
        JN.N.store()

        # get some adjacency matrix and nPre
        aEE = adjacency_matrix_from_flat_inds(p['nExc'], p['nExc'], JN.preEE, JN.posEE)
        aEI = adjacency_matrix_from_flat_inds(p['nInh'], p['nExc'], JN.preEI, JN.posEI)
        aIE = adjacency_matrix_from_flat_inds(p['nExc'], p['nInh'], JN.preIE, JN.posIE)
        aII = adjacency_matrix_from_flat_inds(p['nInh'], p['nInh'], JN.preII, JN.posII)

        nIncomingExcOntoEachExc = aEE.sum(0)
        nIncomingInhOntoEachExc = aEI.sum(0)
        nIncomingExcOntoEachInh = aIE.sum(0)
        nIncomingInhOntoEachInh = aII.sum(0)

        nOutgoingToExcFromEachExc = aEE.sum(1)
        nOutgoingToExcFromEachInh = aEI.sum(1)
        nOutgoingToInhFromEachExc = aIE.sum(1)
        nOutgoingToInhFromEachInh = aII.sum(1)

        # norm by incoming and outgoing
        # normMatEE = (p['nExc'] * p['nExc'] * p['propConnect'] ** 2) / np.outer(nOutgoingToExcFromEachExc,
        #                                                                        nIncomingExcOntoEachExc)
        # normMatEI = (p['nExc'] * p['nInh'] * p['propConnect'] ** 2) / np.outer(nOutgoingToExcFromEachInh,
        #                                                                        nIncomingInhOntoEachExc)
        # normMatIE = (p['nExc'] * p['nInh'] * p['propConnect'] ** 2) / np.outer(nOutgoingToInhFromEachExc,
        #                                                                        nIncomingExcOntoEachInh)
        # normMatII = (p['nInh'] * p['nInh'] * p['propConnect'] ** 2) / np.outer(nOutgoingToInhFromEachInh,
        #                                                                        nIncomingInhOntoEachInh)

        # norm by incoming
        normMatEE = ((p['nExc'] * p['propConnect']) / nIncomingExcOntoEachExc).reshape(1, -1)
        normMatEI = ((p['nInh'] * p['propConnect']) / nIncomingInhOntoEachExc).reshape(1, -1)
        normMatIE = ((p['nExc'] * p['propConnect']) / nIncomingExcOntoEachInh).reshape(1, -1)
        normMatII = ((p['nInh'] * p['propConnect']) / nIncomingInhOntoEachInh).reshape(1, -1)

        # initialize the pdf
        pdfObject = PdfPages(p['saveFolder'] + self.saveName + '_trials.pdf')

        # define message formatters
        meanWeightMsgFormatter = ('upstateFRExc: {:.2f} Hz, upstateFRInh: {:.2f}'
                                  ' Hz, wEE: {:.2f} pA, wIE: {:.2f} pA, wEI: {:.2f} pA, wII: {:.2f} pA')
        sumWeightMsgFormatter = ('movAvgUpFRExc: {:.2f} Hz, movAvgUpFRInh: {:.2f} Hz, '
                                 'dwEE: {:.2f} pA, dwIE: {:.2f} pA, dwEI: {:.2f} pA, dwII: {:.2f} pA')
        meanWeightChangeMsgFormatter = 'mean dwEE: {:.2f} pA, dwIE: {:.2f} pA, dwEI: {:.2f} pA, dwII: {:.2f} pA'

        saveTrialDummy = 0
        for trialInd in range(p['nTrials']):

            print('starting trial {}'.format(trialInd + 1))

            # restore the initial network state
            JN.N.restore()

            # set the weights (all weights are equivalent)
            # JN.unitsExc.jE = wEE
            # JN.unitsExc.jI = wEI
            # JN.unitsInh.jE = wIE
            # JN.unitsInh.jI = wII

            # set the weights (separately for each unit)
            JN.synapsesEE.jEE = wEE
            JN.synapsesEI.jEI = wEI
            JN.synapsesIE.jIE = wIE
            JN.synapsesII.jII = wII

            # run the simulation
            t0 = datetime.now()
            JN.run()
            t1 = datetime.now()

            saveThisTrial = trialInd in p['saveTrials']
            pickleThisFigure = False

            # calculate and record the average FR in the up state
            R = Results()
            R.init_from_network_object(JN)
            R.calculate_PSTH()
            R.calculate_upstates()

            # if there was not zero Up states
            if len(R.ups) == 1:
                R.calculate_upFR_units()  # (separately for each unit)
                print('there was exactly one Up of duration {:.2f} s'.format(R.upDurs[0]))
                self.trialUpFRExc[trialInd] = R.upstateFRExc[0]  # in Hz
                self.trialUpFRInh[trialInd] = R.upstateFRInh[0]  # in Hz
                self.trialUpFRExcUnits[trialInd, :] = R.upstateFRExcUnits[0, :]  # in Hz
                self.trialUpFRInhUnits[trialInd, :] = R.upstateFRInhUnits[0, :]  # in Hz
                self.trialUpDur[trialInd] = R.upDurs[0]
            elif len(R.ups) > 1:
                print('for some reason there were multiple up states!!!')
                R.calculate_upFR_units()  # (separately for each unit)
                print('there were multiple Ups of avg duration {:.2f} s'.format(R.upDurs.mean()))
                self.trialUpFRExc[trialInd] = R.upstateFRExc.mean()  # in Hz
                self.trialUpFRInh[trialInd] = R.upstateFRInh.mean()  # in Hz
                self.trialUpFRExcUnits[trialInd, :] = R.upstateFRExcUnits.mean(0)  # in Hz
                self.trialUpFRInhUnits[trialInd, :] = R.upstateFRInhUnits.mean(0)  # in Hz
                self.trialUpDur[trialInd] = R.upDurs.mean()
            else:
                # if there were no Up states, just take the avg FR (near 0 in this case)
                print('there were no up states')
                # R.calculate_avgFR_units()  # could be used...
                self.trialUpFRExc[trialInd] = R.FRExc.mean()
                self.trialUpFRInh[trialInd] = R.FRInh.mean()
                self.trialUpDur[trialInd] = 0
                # here we assign each unit FR to be the avg of all
                # otherwise we may introduce a bias against the kicked units
                self.trialUpFRExcUnits[trialInd, :] = R.FRExc.mean()  # in Hz
                self.trialUpFRInhUnits[trialInd, :] = R.FRInh.mean()  # in Hz

            # if the currently assessed upstateFR was higher than the saturated FRs of the two types, reduce it
            # this also helps if the system exploded (FRs are maximal)
            # we saturate the FR because the weight changes that ensue are catastrophic
            # if self.trialUpFRExc[trialInd] > p['maxAllowedFRExc']:
            #     self.trialUpFRExc[trialInd] = p['maxAllowedFRExc']
            # if self.trialUpFRInh[trialInd] > p['maxAllowedFRInh']:
            #     self.trialUpFRInh[trialInd] = p['maxAllowedFRInh']

            # separate by unit, record average weight?
            self.trialwEE[trialInd] = wEE.mean() / pA
            self.trialwEI[trialInd] = wEI.mean() / pA
            self.trialwIE[trialInd] = wIE.mean() / pA
            self.trialwII[trialInd] = wII.mean() / pA

            # save numerical results and/or plots!!!
            if saveThisTrial:
                R.calculate_voltage_histogram(removeMode=True)
                R.reshape_upstates()

                fig1, ax1 = plt.subplots(5, 1, num=1, figsize=(5, 9),
                                         gridspec_kw={'height_ratios': [3, 2, 1, 1, 1]},
                                         sharex=True)
                R.plot_spike_raster(ax1[0])  # uses RNG but with a separate random seed
                R.plot_firing_rate(ax1[1])
                ax1[1].set_ylim(0, 30)
                R.plot_voltage_detail(ax1[2], unitType='Exc', useStateInd=0)
                R.plot_updur_lines(ax1[2])
                R.plot_voltage_detail(ax1[3], unitType='Inh', useStateInd=0)
                R.plot_updur_lines(ax1[3])
                R.plot_voltage_detail(ax1[4], unitType='Exc', useStateInd=1)
                R.plot_updur_lines(ax1[4])
                ax1[3].set(xlabel='Time (s)')
                R.plot_voltage_histogram_sideways(ax1[2], 'Exc')
                R.plot_voltage_histogram_sideways(ax1[3], 'Inh')
                fig1.suptitle(R.p['simName'] + '_' + p['useRule'] + '_t' + str(trialInd + 1))
                plt.savefig(pdfObject, format='pdf')
                if pickleThisFigure:
                    pickle.dump(fig1,
                                open(
                                    R.p['saveFolder'] + '/' + R.rID + '_' + p['useRule'] + '_t' + str(
                                        trialInd + 1) + '.pickle',
                                    'wb'))

                if p['recordMovieVariables']:
                    self.selectTrialVExc[saveTrialDummy, :] = R.stateMonExcV[0, ::self.frameMult]
                    self.selectTrialVInh[saveTrialDummy, :] = R.stateMonInhV[0, ::self.frameMult]
                    self.selectTrialFRExc[saveTrialDummy, :] = R.FRExc
                    self.selectTrialFRInh[saveTrialDummy, :] = R.FRInh
                    self.selectTrialSpikeExcI[saveTrialDummy] = R.spikeMonExcI
                    self.selectTrialSpikeExcT[saveTrialDummy] = R.spikeMonExcT
                    self.selectTrialSpikeInhI[saveTrialDummy] = R.spikeMonInhI
                    self.selectTrialSpikeInhT[saveTrialDummy] = R.spikeMonInhT

                saveTrialDummy += 1

            # heck it, print those values
            print(meanWeightMsgFormatter.format(self.trialUpFRExc[trialInd], self.trialUpFRInh[trialInd],
                                                self.trialwEE[trialInd], self.trialwIE[trialInd],
                                                self.trialwEI[trialInd], self.trialwII[trialInd]))

            # calculate the moving average of the up FRs
            if movAvgUpFRExc:
                movAvgUpFRExc += (-movAvgUpFRExc + self.trialUpFRExc[trialInd] * Hz) / p['tauUpFRTrials']
                movAvgUpFRInh += (-movAvgUpFRInh + self.trialUpFRInh[trialInd] * Hz) / p['tauUpFRTrials']

                movAvgUpFRExcUnits += (-movAvgUpFRExcUnits +
                                       self.trialUpFRExcUnits[trialInd, :] * Hz) / p['tauUpFRTrials']
                movAvgUpFRInhUnits += (-movAvgUpFRInhUnits +
                                       self.trialUpFRInhUnits[trialInd, :] * Hz) / p['tauUpFRTrials']

            else:  # this only gets run the first self.trial (when they are None)
                movAvgUpFRExc = self.trialUpFRExc[trialInd] * Hz  # initialize at the first measured
                movAvgUpFRInh = self.trialUpFRInh[trialInd] * Hz

                movAvgUpFRExcUnits = self.trialUpFRExcUnits[trialInd, :] * Hz  # initialize at the first measured
                movAvgUpFRInhUnits = self.trialUpFRInhUnits[trialInd, :] * Hz

            if p['useRule'] == 'cross-homeo':

                # separately by synapse...
                # must create a vector in which each element represents the average FR
                # of the cells that synapse onto the post
                # for wEI & wEE, mean fr of inh units that target each E unit
                # for wIE & wII, mean fr of exc units that target each I unit
                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nIncomingInhOntoEachExc
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh

                movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = \
                    2 * p['setUpFRInh']
                movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = \
                    2 * p['setUpFRExc']
                # movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                # movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # check if this is less than 1... if so, make it be 1 Hz
                # movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc < 1 * Hz] = 1 * Hz
                # movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh < 1 * Hz] = 1 * Hz

                # check if this is greater than 2 * set-point, if so, make it be less
                # movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']
                # movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # proposed weight changes take the form of an array which we interpret as a row vector
                # each element proposes the change to a column of the weight matrix
                # (because the addition broadcasts the row across the columns,
                # the same change is applied to all elements of a column)
                # in other words, all of the incoming synapses to one unit get scaled the same amount
                # depending on the average FR  of the sensor units presynaptic to that unit
                dwEE = p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)
                dwEI = -p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)
                dwIE = -p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)
                dwII = p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE / pA
                self.trialdwEIUnits[trialInd, :] = dwEI / pA
                self.trialdwIEUnits[trialInd, :] = dwIE / pA
                self.trialdwIIUnits[trialInd, :] = dwII / pA

                # this broadcasts the addition across the ROWS (the 1d dwEE arrays are row vectors)
                # this applies the same weight change to all incoming synapses onto a single post-synaptic unit
                # but it's a different value for each post-synaptic unit
                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-outer':

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nIncomingInhOntoEachExc
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p[
                    'setUpFRInh']
                movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p[
                    'setUpFRExc']

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # outer product of pre Fr and average error of pre units in opposite pop
                # outer strips units for some reason, but the unit should technically be Hz ^ 2
                # given the units of alpha we multiply by Hz to convert to amps
                # can interpret as each synapse changes according to the product of the presynaptic FR
                # and the average error of the presynaptic units in the opposite pop (since it's cross-homeo)
                # imagine there is a somatic sensor for each unit's FR (sensor 1, Ca++ based)
                # and a separate somatic sensor that integrates from metabotropic sensors activated by the opposite pop
                # (sensor 2: in Exc cells, a GABA-B based sensor, and in Inh cells, an mGluR based sensor)
                dwEE = p['alpha1'] * np.outer(movAvgUpFRExcUnits,
                                              (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)) * Hz
                dwEI = -p['alpha1'] * np.outer(movAvgUpFRInhUnits,
                                               (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)) * Hz
                dwIE = -p['alpha1'] * np.outer(movAvgUpFRExcUnits,
                                               (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)) * Hz
                dwII = p['alpha1'] * np.outer(movAvgUpFRInhUnits,
                                              (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)) * Hz

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE.mean() / pA
                self.trialdwEIUnits[trialInd, :] = dwEI.mean() / pA
                self.trialdwIEUnits[trialInd, :] = dwIE.mean() / pA
                self.trialdwIIUnits[trialInd, :] = dwII.mean() / pA

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # save the mean absolute delta in pA
                # dwEE et al MUST be matrices of the right size here
                self.trialMAdwEE[trialInd] = np.mean(np.fabs(dwEE[JN.preEE, JN.posEE] / pA))
                self.trialMAdwEI[trialInd] = np.mean(np.fabs(dwEI[JN.preEI, JN.posEI] / pA))
                self.trialMAdwIE[trialInd] = np.mean(np.fabs(dwIE[JN.preIE, JN.posIE] / pA))
                self.trialMAdwII[trialInd] = np.mean(np.fabs(dwII[JN.preII, JN.posII] / pA))

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-outer-careful':

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nIncomingInhOntoEachExc
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p[
                    'setUpFRInh']
                movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p[
                    'setUpFRExc']

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # outer product of pre Fr and average error of pre units in opposite pop
                # outer strips units for some reason, but the unit should technically be Hz ^ 2
                # given the units of alpha we multiply by Hz to convert to amps
                # can interpret as each synapse changes according to the product of the presynaptic FR
                # and the average error of the presynaptic units in the opposite pop (since it's cross-homeo)
                # imagine there is a somatic sensor for each unit's FR (sensor 1, Ca++ based)
                # and a separate somatic sensor that integrates from metabotropic sensors activated by the opposite pop
                # (sensor 2: in Exc cells, a GABA-B based sensor, and in Inh cells, an mGluR based sensor)
                dwEE2 = np.full(wEEMat.shape, np.nan, dtype=np.float32)
                dwEI2 = np.full(wEIMat.shape, np.nan, dtype=np.float32)
                dwIE2 = np.full(wIEMat.shape, np.nan, dtype=np.float32)
                dwII2 = np.full(wIIMat.shape, np.nan, dtype=np.float32)
                for preInd in range(wEEMat.shape[0]):
                    for postInd in range(wEEMat.shape[1]):
                        dwEE2[preInd, postInd] = p['alpha1'] * movAvgUpFRExcUnits[preInd] * (
                                p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc[postInd])
                for preInd in range(wEIMat.shape[0]):
                    for postInd in range(wEIMat.shape[1]):
                        dwEI2[preInd, postInd] = -p['alpha1'] * movAvgUpFRInhUnits[preInd] * (
                                p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc[postInd])
                for preInd in range(wIEMat.shape[0]):
                    for postInd in range(wIEMat.shape[1]):
                        dwIE2[preInd, postInd] = -p['alpha1'] * movAvgUpFRExcUnits[preInd] * (
                                    p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh[postInd])
                for preInd in range(wIIMat.shape[0]):
                    for postInd in range(wIIMat.shape[1]):
                        dwII2[preInd, postInd] = p['alpha1'] * movAvgUpFRInhUnits[preInd] * (
                                    p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh[postInd])

                dwEE = p['alpha1'] * np.outer(movAvgUpFRExcUnits,
                                              (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)) * Hz
                dwEI = -p['alpha1'] * np.outer(movAvgUpFRInhUnits,
                                               (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)) * Hz
                dwIE = -p['alpha1'] * np.outer(movAvgUpFRExcUnits,
                                               (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)) * Hz
                dwII = p['alpha1'] * np.outer(movAvgUpFRInhUnits,
                                              (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)) * Hz

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE.mean() / pA
                self.trialdwEIUnits[trialInd, :] = dwEI.mean() / pA
                self.trialdwIEUnits[trialInd, :] = dwIE.mean() / pA
                self.trialdwIIUnits[trialInd, :] = dwII.mean() / pA

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # perhaps based on a gaseous messenger that diffuses into all cells
                # the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                dwEECH1 = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEICH1 = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIECH1 = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwIICH1 = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                dwEE = np.tile(dwEECH1.reshape(-1, 1), wEEMat.shape[1])
                dwEI = np.tile(dwEICH1.reshape(-1, 1), wEIMat.shape[1])
                dwIE = np.tile(dwIECH1.reshape(-1, 1), wIEMat.shape[1])
                dwII = np.tile(dwIICH1.reshape(-1, 1), wIIMat.shape[1])

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE.mean() / pA
                self.trialdwEIUnits[trialInd, :] = dwEI.mean() / pA
                self.trialdwIEUnits[trialInd, :] = dwIE.mean() / pA
                self.trialdwIIUnits[trialInd, :] = dwII.mean() / pA

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # save the mean absolute delta in pA
                # dwEE et al MUST be matrices of the right size here
                self.trialMAdwEE[trialInd] = np.mean(np.fabs(dwEE[JN.preEE, JN.posEE] / pA))
                self.trialMAdwEI[trialInd] = np.mean(np.fabs(dwEI[JN.preEI, JN.posEI] / pA))
                self.trialMAdwIE[trialInd] = np.mean(np.fabs(dwIE[JN.preIE, JN.posIE] / pA))
                self.trialMAdwII[trialInd] = np.mean(np.fabs(dwII[JN.preII, JN.posII] / pA))

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar-reMean':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                dwEE1 = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEI1 = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIE1 = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwII1 = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                dwEE = np.tile(dwEE1.reshape(-1, 1), wEEMat.shape[1])
                dwEI = np.tile(dwEI1.reshape(-1, 1), wEIMat.shape[1])
                dwIE = np.tile(dwIE1.reshape(-1, 1), wIEMat.shape[1])
                dwII = np.tile(dwII1.reshape(-1, 1), wIIMat.shape[1])

                # re-mean the dW mats -- idea is that the isotropy of the proposed weight changes is only preserved
                # if the mean of the subset is the same as the mean of the superset
                dwEES = dwEE[JN.preEE, JN.posEE]
                dwEIS = dwEI[JN.preEI, JN.posEI]
                dwIES = dwIE[JN.preIE, JN.posIE]
                dwIIS = dwII[JN.preII, JN.posII]

                dwEERM = dwEE + (dwEE.mean() - dwEES.mean())
                dwEIRM = dwEI + (dwEI.mean() - dwEIS.mean())
                dwIERM = dwIE + (dwIE.mean() - dwIES.mean())
                dwIIRM = dwII + (dwII.mean() - dwIIS.mean())

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE.mean() / pA
                self.trialdwEIUnits[trialInd, :] = dwEI.mean() / pA
                self.trialdwIEUnits[trialInd, :] = dwIE.mean() / pA
                self.trialdwIIUnits[trialInd, :] = dwII.mean() / pA

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                wEEMat += dwEERM / pA * JN.p['wEEScale']
                wEIMat += dwEIRM / pA * JN.p['wEIScale']
                wIEMat += dwIERM / pA * JN.p['wIEScale']
                wIIMat += dwIIRM / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-scalar':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # perhaps based on a gaseous messenger that diffuses into all cells
                # the error term is a scalar
                # since there is no presynaptic multiplier, the same value is broadcast across all weights
                # we divide by Hz because of the units of alpha to convert to amps
                dwEE = p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEI = -p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIE = -p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwII = p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE / pA
                self.trialdwEIUnits[trialInd, :] = dwEI / pA
                self.trialdwIEUnits[trialInd, :] = dwIE / pA
                self.trialdwIIUnits[trialInd, :] = dwII / pA

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to an array
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-elementwise':

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nIncomingInhOntoEachExc
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh
                movAvgUpFRInhUnitsPreToPostInh = np.matmul(movAvgUpFRInhUnits, aII) / nIncomingInhOntoEachInh
                movAvgUpFRExcUnitsPreToPostExc = np.matmul(movAvgUpFRExcUnits, aEE) / nIncomingExcOntoEachExc

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p[
                    'setUpFRInh']
                movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p[
                    'setUpFRExc']
                movAvgUpFRInhUnitsPreToPostInh[movAvgUpFRInhUnitsPreToPostInh > 2 * p['setUpFRInh']] = 2 * p[
                    'setUpFRInh']
                movAvgUpFRExcUnitsPreToPostExc[movAvgUpFRExcUnitsPreToPostExc > 2 * p['setUpFRExc']] = 2 * p[
                    'setUpFRExc']
                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # elementwise product results in units of Hz ^ 2
                # given units of alpha we divide by Hz again to convert to amps
                dwEE = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc) / Hz
                dwEI = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostInh) / Hz
                dwIE = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostExc) / Hz
                dwII = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh) / Hz

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE / pA
                self.trialdwEIUnits[trialInd, :] = dwEI.mean() / pA
                self.trialdwIEUnits[trialInd, :] = dwIE.mean() / pA
                self.trialdwIIUnits[trialInd, :] = dwII / pA

                # this special reshaping is require to make sure the right dimension is aligned for broadcast addition
                wEEMat += dwEE.reshape(-1, 1) / pA * JN.p['wEEScale']
                wEIMat += dwEI.reshape(-1, 1) / pA * JN.p['wEIScale']
                wIEMat += dwIE.reshape(-1, 1) / pA * JN.p['wIEScale']
                wIIMat += dwII.reshape(-1, 1) / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-outer-homeo':

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nIncomingInhOntoEachExc
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p[
                    'setUpFRInh']
                movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p[
                    'setUpFRExc']

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # cross-homeo with presynaptic multiplier (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEECH = p['alpha1'] * np.outer(movAvgUpFRExcUnits,
                                                (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)) * Hz
                dwEICH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits,
                                                 (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)) * Hz
                dwIECH = -p['alpha1'] * np.outer(movAvgUpFRExcUnits,
                                                 (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)) * Hz
                dwIICH = p['alpha1'] * np.outer(movAvgUpFRInhUnits,
                                                (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)) * Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwIIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                dwEE = dwEECH + dwEEH
                dwEI = dwEICH + dwEIH
                dwIE = dwIECH + dwIEH
                dwII = dwIICH + dwIIH

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE.mean() / pA
                self.trialdwEIUnits[trialInd, :] = dwEI.mean() / pA
                self.trialdwIEUnits[trialInd, :] = dwIE.mean() / pA
                self.trialdwIIUnits[trialInd, :] = dwII.mean() / pA

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # save the mean absolute delta in pA
                # dwEE et al MUST be matrices of the right size here
                self.trialMAdwEE[trialInd] = np.mean(np.fabs(dwEE[JN.preEE, JN.posEE] / pA))
                self.trialMAdwEI[trialInd] = np.mean(np.fabs(dwEI[JN.preEI, JN.posEI] / pA))
                self.trialMAdwIE[trialInd] = np.mean(np.fabs(dwIE[JN.preIE, JN.posIE] / pA))
                self.trialMAdwII[trialInd] = np.mean(np.fabs(dwII[JN.preII, JN.posII] / pA))

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'homeo':

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nIncomingInhOntoEachExc
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p[
                    'setUpFRInh']
                movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p[
                    'setUpFRExc']

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEE = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEI = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIE = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwII = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE.mean() / pA
                self.trialdwEIUnits[trialInd, :] = dwEI.mean() / pA
                self.trialdwIEUnits[trialInd, :] = dwIE.mean() / pA
                self.trialdwIIUnits[trialInd, :] = dwII.mean() / pA

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-scalar-homeo':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                dwEECH = p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean())
                dwEICH = -p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean())
                dwIECH = -p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean())
                dwIICH = p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean())

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnits)
                dwEIH = -p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnits)
                dwIEH = p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInhUnits)
                dwIIH = -p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInhUnits)

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEECH + dwEEH
                dwEI = dwEICH + dwEIH
                dwIE = dwIECH + dwIEH
                dwII = dwIICH + dwIIH

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE / pA
                self.trialdwEIUnits[trialInd, :] = dwEI / pA
                self.trialdwIEUnits[trialInd, :] = dwIE / pA
                self.trialdwIIUnits[trialInd, :] = dwII / pA

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > (2 * p['setUpFRExc'] - 1 * Hz)] = (2 * p['setUpFRExc'] - 1 * Hz)
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > (2 * p['setUpFRInh'] - 1 * Hz)] = (2 * p['setUpFRInh'] - 1 * Hz)

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                dwEECH = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEICH = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIECH = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwIICH = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwIIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEECHUnits'):
                        self.trialdwEECHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEICHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIECHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIICHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                        self.trialdwEEHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save both the CH contribution and the H contribution
                    self.trialdwEECHUnits[trialInd, :] = dwEECH / pA
                    self.trialdwEICHUnits[trialInd, :] = dwEICH / pA
                    self.trialdwIECHUnits[trialInd, :] = dwIECH / pA
                    self.trialdwIICHUnits[trialInd, :] = dwIICH / pA

                    dwEEH_tmp = dwEEH / pA
                    dwEIH_tmp = dwEIH / pA
                    dwIEH_tmp = dwIEH / pA
                    dwIIH_tmp = dwIIH / pA

                    dwEEH_tmp[np.isnan(wEEMat)] = np.nan
                    dwEIH_tmp[np.isnan(wEIMat)] = np.nan
                    dwIEH_tmp[np.isnan(wIEMat)] = np.nan
                    dwIIH_tmp[np.isnan(wIIMat)] = np.nan

                    self.trialdwEEHUnits[trialInd, :] = np.nanmean(dwEEH_tmp, 1)  # 1 for outgoing
                    self.trialdwEIHUnits[trialInd, :] = np.nanmean(dwEIH_tmp, 1)
                    self.trialdwIEHUnits[trialInd, :] = np.nanmean(dwIEH_tmp, 1)
                    self.trialdwIIHUnits[trialInd, :] = np.nanmean(dwIIH_tmp, 1)

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEECH.reshape(-1, 1) + dwEEH
                dwEI = dwEICH.reshape(-1, 1) + dwEIH
                dwIE = dwIECH.reshape(-1, 1) + dwIEH
                dwII = dwIICH.reshape(-1, 1) + dwIIH

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEEOUnits'):
                        self.trialdwEEOUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIOUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEOUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIOUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save the proposed weight change in pA
                    self.trialdwEEOUnits[trialInd, :] = np.nanmean(dwEE / pA, 1)
                    self.trialdwEIOUnits[trialInd, :] = np.nanmean(dwEI / pA, 1)
                    self.trialdwIEOUnits[trialInd, :] = np.nanmean(dwIE / pA, 1)
                    self.trialdwIIOUnits[trialInd, :] = np.nanmean(dwII / pA, 1)

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = np.nanmean(dwEE / pA, 0)
                self.trialdwEIUnits[trialInd, :] = np.nanmean(dwEI / pA, 0)
                self.trialdwIEUnits[trialInd, :] = np.nanmean(dwIE / pA, 0)
                self.trialdwIIUnits[trialInd, :] = np.nanmean(dwII / pA, 0)

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # save the mean absolute delta in pA
                # dwEE et al MUST be matrices of the right size here
                self.trialMAdwEE[trialInd] = np.mean(np.fabs(dwEE[JN.preEE, JN.posEE] / pA))
                self.trialMAdwEI[trialInd] = np.mean(np.fabs(dwEI[JN.preEI, JN.posEI] / pA))
                self.trialMAdwIE[trialInd] = np.mean(np.fabs(dwIE[JN.preIE, JN.posIE] / pA))
                self.trialMAdwII[trialInd] = np.mean(np.fabs(dwII[JN.preII, JN.posII] / pA))

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo-reMean':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > (2 * p['setUpFRExc'] - 1 * Hz)] = (2 * p['setUpFRExc'] - 1 * Hz)
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > (2 * p['setUpFRInh'] - 1 * Hz)] = (2 * p['setUpFRInh'] - 1 * Hz)

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                dwEECH = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEICH = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIECH = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwIICH = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwIIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEECHUnits'):
                        self.trialdwEECHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEICHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIECHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIICHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                        self.trialdwEEHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save both the CH contribution and the H contribution
                    self.trialdwEECHUnits[trialInd, :] = dwEECH / pA
                    self.trialdwEICHUnits[trialInd, :] = dwEICH / pA
                    self.trialdwIECHUnits[trialInd, :] = dwIECH / pA
                    self.trialdwIICHUnits[trialInd, :] = dwIICH / pA

                    dwEEH_tmp = dwEEH / pA
                    dwEIH_tmp = dwEIH / pA
                    dwIEH_tmp = dwIEH / pA
                    dwIIH_tmp = dwIIH / pA

                    dwEEH_tmp[np.isnan(wEEMat)] = np.nan
                    dwEIH_tmp[np.isnan(wEIMat)] = np.nan
                    dwIEH_tmp[np.isnan(wIEMat)] = np.nan
                    dwIIH_tmp[np.isnan(wIIMat)] = np.nan

                    self.trialdwEEHUnits[trialInd, :] = np.nanmean(dwEEH_tmp, 1)  # 1 for outgoing
                    self.trialdwEIHUnits[trialInd, :] = np.nanmean(dwEIH_tmp, 1)
                    self.trialdwIEHUnits[trialInd, :] = np.nanmean(dwIEH_tmp, 1)
                    self.trialdwIIHUnits[trialInd, :] = np.nanmean(dwIIH_tmp, 1)

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEECH.reshape(-1, 1) + dwEEH
                dwEI = dwEICH.reshape(-1, 1) + dwEIH
                dwIE = dwIECH.reshape(-1, 1) + dwIEH
                dwII = dwIICH.reshape(-1, 1) + dwIIH

                # re-mean the dW mats -- idea is that the isotropy of the proposed weight changes is only preserved
                # if the mean of the subset is the same as the mean of the superset
                dwEES = dwEE[JN.preEE, JN.posEE]
                dwEIS = dwEI[JN.preEI, JN.posEI]
                dwIES = dwIE[JN.preIE, JN.posIE]
                dwIIS = dwII[JN.preII, JN.posII]

                dwEERM = dwEE + (dwEE.mean() - dwEES.mean())
                dwEIRM = dwEI + (dwEI.mean() - dwEIS.mean())
                dwIERM = dwIE + (dwIE.mean() - dwIES.mean())
                dwIIRM = dwII + (dwII.mean() - dwIIS.mean())

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEEOUnits'):
                        self.trialdwEEOUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIOUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEOUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIOUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save the proposed weight change in pA
                    self.trialdwEEOUnits[trialInd, :] = np.nanmean(dwEE / pA, 1)
                    self.trialdwEIOUnits[trialInd, :] = np.nanmean(dwEI / pA, 1)
                    self.trialdwIEOUnits[trialInd, :] = np.nanmean(dwIE / pA, 1)
                    self.trialdwIIOUnits[trialInd, :] = np.nanmean(dwII / pA, 1)

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = np.nanmean(dwEE / pA, 0)
                self.trialdwEIUnits[trialInd, :] = np.nanmean(dwEI / pA, 0)
                self.trialdwIEUnits[trialInd, :] = np.nanmean(dwIE / pA, 0)
                self.trialdwIIUnits[trialInd, :] = np.nanmean(dwII / pA, 0)

                wEEMat += dwEERM / pA * JN.p['wEEScale']
                wEIMat += dwEIRM / pA * JN.p['wEIScale']
                wIEMat += dwIERM / pA * JN.p['wIEScale']
                wIIMat += dwIIRM / pA * JN.p['wIIScale']

                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo-normCH':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > (2 * p['setUpFRExc'] - 1 * Hz)] = (2 * p['setUpFRExc'] - 1 * Hz)
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > (2 * p['setUpFRInh'] - 1 * Hz)] = (2 * p['setUpFRInh'] - 1 * Hz)

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                dwEECH1 = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEICH1 = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIECH1 = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwIICH1 = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                dwEECH = np.tile(dwEECH1.reshape(-1, 1), wEEMat.shape[1])
                dwEICH = np.tile(dwEICH1.reshape(-1, 1), wEIMat.shape[1])
                dwIECH = np.tile(dwIECH1.reshape(-1, 1), wIEMat.shape[1])
                dwIICH = np.tile(dwIICH1.reshape(-1, 1), wIIMat.shape[1])

                # normalize by incoming
                dwEECH *= ((p['nExc'] * p['propConnect']) / nIncomingExcOntoEachExc).reshape(1, -1)
                dwEICH *= ((p['nInh'] * p['propConnect']) / nIncomingInhOntoEachExc).reshape(1, -1)
                dwIECH *= ((p['nExc'] * p['propConnect']) / nIncomingExcOntoEachInh).reshape(1, -1)
                dwIICH *= ((p['nInh'] * p['propConnect']) / nIncomingInhOntoEachInh).reshape(1, -1)

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwIIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEECHUnits'):
                        self.trialdwEECHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEICHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIECHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIICHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                        self.trialdwEEHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save both the CH contribution and the H contribution
                    self.trialdwEECHUnits[trialInd, :] = dwEECH / pA
                    self.trialdwEICHUnits[trialInd, :] = dwEICH / pA
                    self.trialdwIECHUnits[trialInd, :] = dwIECH / pA
                    self.trialdwIICHUnits[trialInd, :] = dwIICH / pA

                    dwEEH_tmp = dwEEH / pA
                    dwEIH_tmp = dwEIH / pA
                    dwIEH_tmp = dwIEH / pA
                    dwIIH_tmp = dwIIH / pA

                    dwEEH_tmp[np.isnan(wEEMat)] = np.nan
                    dwEIH_tmp[np.isnan(wEIMat)] = np.nan
                    dwIEH_tmp[np.isnan(wIEMat)] = np.nan
                    dwIIH_tmp[np.isnan(wIIMat)] = np.nan

                    self.trialdwEEHUnits[trialInd, :] = np.nanmean(dwEEH_tmp, 1)  # 1 for outgoing
                    self.trialdwEIHUnits[trialInd, :] = np.nanmean(dwEIH_tmp, 1)
                    self.trialdwIEHUnits[trialInd, :] = np.nanmean(dwIEH_tmp, 1)
                    self.trialdwIIHUnits[trialInd, :] = np.nanmean(dwIIH_tmp, 1)

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEECH + dwEEH
                dwEI = dwEICH + dwEIH
                dwIE = dwIECH + dwIEH
                dwII = dwIICH + dwIIH

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEEOUnits'):
                        self.trialdwEEOUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIOUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEOUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIOUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save the proposed weight change in pA
                    self.trialdwEEOUnits[trialInd, :] = np.nanmean(dwEE / pA, 1)
                    self.trialdwEIOUnits[trialInd, :] = np.nanmean(dwEI / pA, 1)
                    self.trialdwIEOUnits[trialInd, :] = np.nanmean(dwIE / pA, 1)
                    self.trialdwIIOUnits[trialInd, :] = np.nanmean(dwII / pA, 1)

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = np.nanmean(dwEE / pA, 0)
                self.trialdwEIUnits[trialInd, :] = np.nanmean(dwEI / pA, 0)
                self.trialdwIEUnits[trialInd, :] = np.nanmean(dwIE / pA, 0)
                self.trialdwIIUnits[trialInd, :] = np.nanmean(dwII / pA, 0)

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo-normSum':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > (2 * p['setUpFRExc'] - 1 * Hz)] = (2 * p['setUpFRExc'] - 1 * Hz)
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > (2 * p['setUpFRInh'] - 1 * Hz)] = (2 * p['setUpFRInh'] - 1 * Hz)

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                dwEECH = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEICH = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIECH = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwIICH = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwIIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEECHUnits'):
                        self.trialdwEECHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEICHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIECHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIICHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                        self.trialdwEEHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save both the CH contribution and the H contribution
                    self.trialdwEECHUnits[trialInd, :] = dwEECH / pA
                    self.trialdwEICHUnits[trialInd, :] = dwEICH / pA
                    self.trialdwIECHUnits[trialInd, :] = dwIECH / pA
                    self.trialdwIICHUnits[trialInd, :] = dwIICH / pA

                    dwEEH_tmp = dwEEH / pA
                    dwEIH_tmp = dwEIH / pA
                    dwIEH_tmp = dwIEH / pA
                    dwIIH_tmp = dwIIH / pA

                    dwEEH_tmp[np.isnan(wEEMat)] = np.nan
                    dwEIH_tmp[np.isnan(wEIMat)] = np.nan
                    dwIEH_tmp[np.isnan(wIEMat)] = np.nan
                    dwIIH_tmp[np.isnan(wIIMat)] = np.nan

                    self.trialdwEEHUnits[trialInd, :] = np.nanmean(dwEEH_tmp, 1)  # 1 for outgoing
                    self.trialdwEIHUnits[trialInd, :] = np.nanmean(dwEIH_tmp, 1)
                    self.trialdwIEHUnits[trialInd, :] = np.nanmean(dwIEH_tmp, 1)
                    self.trialdwIIHUnits[trialInd, :] = np.nanmean(dwIIH_tmp, 1)

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = (dwEECH.reshape(-1, 1) + dwEEH) * normMatEE
                dwEI = (dwEICH.reshape(-1, 1) + dwEIH) * normMatEI
                dwIE = (dwIECH.reshape(-1, 1) + dwIEH) * normMatIE
                dwII = (dwIICH.reshape(-1, 1) + dwIIH) * normMatII

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEEOUnits'):
                        self.trialdwEEOUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIOUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEOUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIOUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save the proposed weight change in pA
                    self.trialdwEEOUnits[trialInd, :] = np.nanmean(dwEE / pA, 1)
                    self.trialdwEIOUnits[trialInd, :] = np.nanmean(dwEI / pA, 1)
                    self.trialdwIEOUnits[trialInd, :] = np.nanmean(dwIE / pA, 1)
                    self.trialdwIIOUnits[trialInd, :] = np.nanmean(dwII / pA, 1)

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = np.nanmean(dwEE / pA, 0)
                self.trialdwEIUnits[trialInd, :] = np.nanmean(dwEI / pA, 0)
                self.trialdwIEUnits[trialInd, :] = np.nanmean(dwIE / pA, 0)
                self.trialdwIIUnits[trialInd, :] = np.nanmean(dwII / pA, 0)

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo-noCap':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                # movAvgUpFRExcUnits[movAvgUpFRExcUnits > (2 * p['setUpFRExc'] - 1 * Hz)] = (2 * p['setUpFRExc'] - 1 * Hz)
                # movAvgUpFRInhUnits[movAvgUpFRInhUnits > (2 * p['setUpFRInh'] - 1 * Hz)] = (2 * p['setUpFRInh'] - 1 * Hz)

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                dwEECH = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEICH = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIECH = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwIICH = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwIIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEECHUnits'):
                        self.trialdwEECHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEICHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIECHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIICHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                        self.trialdwEEHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save both the CH contribution and the H contribution
                    self.trialdwEECHUnits[trialInd, :] = dwEECH / pA
                    self.trialdwEICHUnits[trialInd, :] = dwEICH / pA
                    self.trialdwIECHUnits[trialInd, :] = dwIECH / pA
                    self.trialdwIICHUnits[trialInd, :] = dwIICH / pA

                    dwEEH_tmp = dwEEH / pA
                    dwEIH_tmp = dwEIH / pA
                    dwIEH_tmp = dwIEH / pA
                    dwIIH_tmp = dwIIH / pA

                    dwEEH_tmp[np.isnan(wEEMat)] = np.nan
                    dwEIH_tmp[np.isnan(wEIMat)] = np.nan
                    dwIEH_tmp[np.isnan(wIEMat)] = np.nan
                    dwIIH_tmp[np.isnan(wIIMat)] = np.nan

                    self.trialdwEEHUnits[trialInd, :] = np.nanmean(dwEEH_tmp, 1)  # 1 for outgoing
                    self.trialdwEIHUnits[trialInd, :] = np.nanmean(dwEIH_tmp, 1)
                    self.trialdwIEHUnits[trialInd, :] = np.nanmean(dwIEH_tmp, 1)
                    self.trialdwIIHUnits[trialInd, :] = np.nanmean(dwIIH_tmp, 1)

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEECH.reshape(-1, 1) + dwEEH
                dwEI = dwEICH.reshape(-1, 1) + dwEIH
                dwIE = dwIECH.reshape(-1, 1) + dwIEH
                dwII = dwIICH.reshape(-1, 1) + dwIIH

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEEOUnits'):
                        self.trialdwEEOUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIOUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEOUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIOUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save the proposed weight change in pA
                    self.trialdwEEOUnits[trialInd, :] = np.nanmean(dwEE / pA, 1)
                    self.trialdwEIOUnits[trialInd, :] = np.nanmean(dwEI / pA, 1)
                    self.trialdwIEOUnits[trialInd, :] = np.nanmean(dwIE / pA, 1)
                    self.trialdwIIOUnits[trialInd, :] = np.nanmean(dwII / pA, 1)

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = np.nanmean(dwEE / pA, 0)
                self.trialdwEIUnits[trialInd, :] = np.nanmean(dwEI / pA, 0)
                self.trialdwIEUnits[trialInd, :] = np.nanmean(dwIE / pA, 0)
                self.trialdwIIUnits[trialInd, :] = np.nanmean(dwII / pA, 0)

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo-flip':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > (2 * p['setUpFRExc'] - 1 * Hz)] = (2 * p['setUpFRExc'] - 1 * Hz)
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > (2 * p['setUpFRInh'] - 1 * Hz)] = (2 * p['setUpFRInh'] - 1 * Hz)

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                dwEECH = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEICH = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIECH = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwIICH = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwIIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEECHUnits'):
                        self.trialdwEECHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEICHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIECHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIICHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                        self.trialdwEEHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save both the CH contribution and the H contribution
                    self.trialdwEECHUnits[trialInd, :] = dwEECH / pA
                    self.trialdwEICHUnits[trialInd, :] = dwEICH / pA
                    self.trialdwIECHUnits[trialInd, :] = dwIECH / pA
                    self.trialdwIICHUnits[trialInd, :] = dwIICH / pA

                    dwEEH_tmp = dwEEH / pA
                    dwEIH_tmp = dwEIH / pA
                    dwIEH_tmp = dwIEH / pA
                    dwIIH_tmp = dwIIH / pA

                    dwEEH_tmp[np.isnan(wEEMat)] = np.nan
                    dwEIH_tmp[np.isnan(wEIMat)] = np.nan
                    dwIEH_tmp[np.isnan(wIEMat)] = np.nan
                    dwIIH_tmp[np.isnan(wIIMat)] = np.nan

                    self.trialdwEEHUnits[trialInd, :] = np.nanmean(dwEEH_tmp, 1)  # 1 for outgoing
                    self.trialdwEIHUnits[trialInd, :] = np.nanmean(dwEIH_tmp, 1)
                    self.trialdwIEHUnits[trialInd, :] = np.nanmean(dwIEH_tmp, 1)
                    self.trialdwIIHUnits[trialInd, :] = np.nanmean(dwIIH_tmp, 1)

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEECH.reshape(-1, 1) + dwEEH
                dwEI = dwEICH.reshape(-1, 1) + dwEIH
                dwIE = dwIECH.reshape(-1, 1) + dwIEH
                dwII = dwIICH.reshape(-1, 1) + dwIIH

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEEOUnits'):
                        self.trialdwEEOUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIOUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEOUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIOUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save the proposed weight change in pA
                    self.trialdwEEOUnits[trialInd, :] = np.nanmean(dwEE / pA, 1)
                    self.trialdwEIOUnits[trialInd, :] = np.nanmean(dwEI / pA, 1)
                    self.trialdwIEOUnits[trialInd, :] = np.nanmean(dwIE / pA, 1)
                    self.trialdwIIOUnits[trialInd, :] = np.nanmean(dwII / pA, 1)

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = np.nanmean(dwEE / pA, 0)
                self.trialdwEIUnits[trialInd, :] = np.nanmean(dwEI / pA, 0)
                self.trialdwIEUnits[trialInd, :] = np.nanmean(dwIE / pA, 0)
                self.trialdwIIUnits[trialInd, :] = np.nanmean(dwII / pA, 0)

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo-flipCH':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > (2 * p['setUpFRExc'] - 1 * Hz)] = (2 * p['setUpFRExc'] - 1 * Hz)
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > (2 * p['setUpFRInh'] - 1 * Hz)] = (2 * p['setUpFRInh'] - 1 * Hz)

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                dwEECH = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEICH = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIECH = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwIICH = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwIIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEECHUnits'):
                        self.trialdwEECHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEICHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIECHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIICHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                        self.trialdwEEHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save both the CH contribution and the H contribution
                    self.trialdwEECHUnits[trialInd, :] = dwEECH / pA
                    self.trialdwEICHUnits[trialInd, :] = dwEICH / pA
                    self.trialdwIECHUnits[trialInd, :] = dwIECH / pA
                    self.trialdwIICHUnits[trialInd, :] = dwIICH / pA

                    dwEEH_tmp = dwEEH / pA
                    dwEIH_tmp = dwEIH / pA
                    dwIEH_tmp = dwIEH / pA
                    dwIIH_tmp = dwIIH / pA

                    dwEEH_tmp[np.isnan(wEEMat)] = np.nan
                    dwEIH_tmp[np.isnan(wEIMat)] = np.nan
                    dwIEH_tmp[np.isnan(wIEMat)] = np.nan
                    dwIIH_tmp[np.isnan(wIIMat)] = np.nan

                    self.trialdwEEHUnits[trialInd, :] = np.nanmean(dwEEH_tmp, 1)  # 1 for outgoing
                    self.trialdwEIHUnits[trialInd, :] = np.nanmean(dwEIH_tmp, 1)
                    self.trialdwIEHUnits[trialInd, :] = np.nanmean(dwIEH_tmp, 1)
                    self.trialdwIIHUnits[trialInd, :] = np.nanmean(dwIIH_tmp, 1)

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEECH.reshape(-1, 1) + dwEEH
                dwEI = dwEICH.reshape(-1, 1) + dwEIH
                dwIE = dwIECH.reshape(-1, 1) + dwIEH
                dwII = dwIICH.reshape(-1, 1) + dwIIH

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEEOUnits'):
                        self.trialdwEEOUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIOUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEOUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIOUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save the proposed weight change in pA
                    self.trialdwEEOUnits[trialInd, :] = np.nanmean(dwEE / pA, 1)
                    self.trialdwEIOUnits[trialInd, :] = np.nanmean(dwEI / pA, 1)
                    self.trialdwIEOUnits[trialInd, :] = np.nanmean(dwIE / pA, 1)
                    self.trialdwIIOUnits[trialInd, :] = np.nanmean(dwII / pA, 1)

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = np.nanmean(dwEE / pA, 0)
                self.trialdwEIUnits[trialInd, :] = np.nanmean(dwEI / pA, 0)
                self.trialdwIEUnits[trialInd, :] = np.nanmean(dwIE / pA, 0)
                self.trialdwIIUnits[trialInd, :] = np.nanmean(dwII / pA, 0)

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo-flipH':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > (2 * p['setUpFRExc'] - 1 * Hz)] = (2 * p['setUpFRExc'] - 1 * Hz)
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > (2 * p['setUpFRInh'] - 1 * Hz)] = (2 * p['setUpFRInh'] - 1 * Hz)

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                dwEECH = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEICH = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIECH = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwIICH = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = -p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEIH = p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwIIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEECHUnits'):
                        self.trialdwEECHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEICHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIECHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIICHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                        self.trialdwEEHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save both the CH contribution and the H contribution
                    self.trialdwEECHUnits[trialInd, :] = dwEECH / pA
                    self.trialdwEICHUnits[trialInd, :] = dwEICH / pA
                    self.trialdwIECHUnits[trialInd, :] = dwIECH / pA
                    self.trialdwIICHUnits[trialInd, :] = dwIICH / pA

                    dwEEH_tmp = dwEEH / pA
                    dwEIH_tmp = dwEIH / pA
                    dwIEH_tmp = dwIEH / pA
                    dwIIH_tmp = dwIIH / pA

                    dwEEH_tmp[np.isnan(wEEMat)] = np.nan
                    dwEIH_tmp[np.isnan(wEIMat)] = np.nan
                    dwIEH_tmp[np.isnan(wIEMat)] = np.nan
                    dwIIH_tmp[np.isnan(wIIMat)] = np.nan

                    self.trialdwEEHUnits[trialInd, :] = np.nanmean(dwEEH_tmp, 1)  # 1 for outgoing
                    self.trialdwEIHUnits[trialInd, :] = np.nanmean(dwEIH_tmp, 1)
                    self.trialdwIEHUnits[trialInd, :] = np.nanmean(dwIEH_tmp, 1)
                    self.trialdwIIHUnits[trialInd, :] = np.nanmean(dwIIH_tmp, 1)

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEECH.reshape(-1, 1) + dwEEH
                dwEI = dwEICH.reshape(-1, 1) + dwEIH
                dwIE = dwIECH.reshape(-1, 1) + dwIEH
                dwII = dwIICH.reshape(-1, 1) + dwIIH

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEEOUnits'):
                        self.trialdwEEOUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIOUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEOUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIOUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save the proposed weight change in pA
                    self.trialdwEEOUnits[trialInd, :] = np.nanmean(dwEE / pA, 1)
                    self.trialdwEIOUnits[trialInd, :] = np.nanmean(dwEI / pA, 1)
                    self.trialdwIEOUnits[trialInd, :] = np.nanmean(dwIE / pA, 1)
                    self.trialdwIIOUnits[trialInd, :] = np.nanmean(dwII / pA, 1)

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = np.nanmean(dwEE / pA, 0)
                self.trialdwEIUnits[trialInd, :] = np.nanmean(dwEI / pA, 0)
                self.trialdwIEUnits[trialInd, :] = np.nanmean(dwIE / pA, 0)
                self.trialdwIIUnits[trialInd, :] = np.nanmean(dwII / pA, 0)

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo-corrected':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > (2 * p['setUpFRExc'] - 1 * Hz)] = (2 * p['setUpFRExc'] - 1 * Hz)
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > (2 * p['setUpFRInh'] - 1 * Hz)] = (2 * p['setUpFRInh'] - 1 * Hz)

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                # going to try fixing units by reversing sign when applicable
                dwEECH = p['alpha1'] * (
                            +.01970 * Hz + movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz)
                dwEICH = -p['alpha1'] * (
                            +0.0674 * Hz + movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz)
                dwIECH = -p['alpha1'] * (
                            -0.0076 * Hz + movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz)
                dwIICH = p['alpha1'] * (
                            -0.0194 * Hz + movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz)

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * (
                            +.01500 * Hz + np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz)
                dwEIH = -p['alpha1'] * (
                            -0.1430 * Hz + np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz)
                dwIEH = p['alpha1'] * (
                            +0.0927 * Hz + np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz)
                dwIIH = -p['alpha1'] * (
                            -0.2130 * Hz + np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz)

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEECHUnits'):
                        self.trialdwEECHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEICHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIECHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                         dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIICHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                        self.trialdwEEHUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIHUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEHUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIHUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save both the CH contribution and the H contribution
                    self.trialdwEECHUnits[trialInd, :] = dwEECH / pA
                    self.trialdwEICHUnits[trialInd, :] = dwEICH / pA
                    self.trialdwIECHUnits[trialInd, :] = dwIECH / pA
                    self.trialdwIICHUnits[trialInd, :] = dwIICH / pA

                    dwEEH_tmp = dwEEH / pA
                    dwEIH_tmp = dwEIH / pA
                    dwIEH_tmp = dwIEH / pA
                    dwIIH_tmp = dwIIH / pA

                    dwEEH_tmp[np.isnan(wEEMat)] = np.nan
                    dwEIH_tmp[np.isnan(wEIMat)] = np.nan
                    dwIEH_tmp[np.isnan(wIEMat)] = np.nan
                    dwIIH_tmp[np.isnan(wIIMat)] = np.nan

                    self.trialdwEEHUnits[trialInd, :] = np.nanmean(dwEEH_tmp, 1)  # 1 for outgoing
                    self.trialdwEIHUnits[trialInd, :] = np.nanmean(dwEIH_tmp, 1)
                    self.trialdwIEHUnits[trialInd, :] = np.nanmean(dwIEH_tmp, 1)
                    self.trialdwIIHUnits[trialInd, :] = np.nanmean(dwIIH_tmp, 1)

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEECH.reshape(-1, 1) + dwEEH
                dwEI = dwEICH.reshape(-1, 1) + dwEIH
                dwIE = dwIECH.reshape(-1, 1) + dwIEH
                dwII = dwIICH.reshape(-1, 1) + dwIIH

                if self.p['saveTermsSeparately']:
                    if not hasattr(self, 'trialdwEEOUnits'):
                        self.trialdwEEOUnits = np.empty((self.p['nTrials'], self.p['nExc']), dtype='float32')
                        self.trialdwEIOUnits = np.empty((self.p['nTrials'], self.p['nInh']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIEOUnits = np.empty((self.p['nTrials'], self.p['nExc']),
                                                        dtype='float32')  # SWAPPED SINCE IT'S OUTGOING
                        self.trialdwIIOUnits = np.empty((self.p['nTrials'], self.p['nInh']), dtype='float32')

                    # save the proposed weight change in pA
                    self.trialdwEEOUnits[trialInd, :] = np.nanmean(dwEE / pA, 1)
                    self.trialdwEIOUnits[trialInd, :] = np.nanmean(dwEI / pA, 1)
                    self.trialdwIEOUnits[trialInd, :] = np.nanmean(dwIE / pA, 1)
                    self.trialdwIIOUnits[trialInd, :] = np.nanmean(dwII / pA, 1)

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = np.nanmean(dwEE / pA, 0)
                self.trialdwEIUnits[trialInd, :] = np.nanmean(dwEI / pA, 0)
                self.trialdwIEUnits[trialInd, :] = np.nanmean(dwIE / pA, 0)
                self.trialdwIIUnits[trialInd, :] = np.nanmean(dwII / pA, 0)

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo-norm':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # here we assume there is a global sensed value of the average FR of E or I units,
                # so the error term is a scalar
                # we divide by Hz because of the units of alpha to convert to amps
                setUpE = p['setUpFRExc'] / Hz
                setUpI = p['setUpFRInh'] / Hz

                dwEECH = p['alpha1'] * movAvgUpFRExcUnits / setUpE * (
                            p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / setUpI / Hz
                dwEICH = -p['alpha1'] * movAvgUpFRInhUnits / setUpI * (
                            p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / setUpI / Hz
                dwIECH = -p['alpha1'] * movAvgUpFRExcUnits / setUpE * (
                            p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / setUpE / Hz
                dwIICH = p['alpha1'] * movAvgUpFRInhUnits / setUpI * (
                            p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / setUpE / Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits / setUpE,
                                               (p['setUpFRExc'] - movAvgUpFRExcUnits) / setUpE) * Hz
                dwEIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits / setUpI,
                                                (p['setUpFRExc'] - movAvgUpFRExcUnits) / setUpE) * Hz
                dwIEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits / setUpE,
                                               (p['setUpFRInh'] - movAvgUpFRInhUnits) / setUpI) * Hz
                dwIIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits / setUpI,
                                                (p['setUpFRInh'] - movAvgUpFRInhUnits) / setUpI) * Hz

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEECH.reshape(-1, 1) + dwEEH
                dwEI = dwEICH.reshape(-1, 1) + dwEIH
                dwIE = dwIECH.reshape(-1, 1) + dwIEH
                dwII = dwIICH.reshape(-1, 1) + dwIIH

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE.mean() / pA
                self.trialdwEIUnits[trialInd, :] = dwEI.mean() / pA
                self.trialdwIEUnits[trialInd, :] = dwIE.mean() / pA
                self.trialdwIIUnits[trialInd, :] = dwII.mean() / pA

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-elementwise-homeo':

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nIncomingInhOntoEachExc
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh
                movAvgUpFRInhUnitsPreToPostInh = np.matmul(movAvgUpFRInhUnits, aII) / nIncomingInhOntoEachInh
                movAvgUpFRExcUnitsPreToPostExc = np.matmul(movAvgUpFRExcUnits, aEE) / nIncomingExcOntoEachExc

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p[
                    'setUpFRInh']
                movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p[
                    'setUpFRExc']
                movAvgUpFRInhUnitsPreToPostInh[movAvgUpFRInhUnitsPreToPostInh > 2 * p['setUpFRInh']] = 2 * p[
                    'setUpFRInh']
                movAvgUpFRExcUnitsPreToPostExc[movAvgUpFRExcUnitsPreToPostExc > 2 * p['setUpFRExc']] = 2 * p[
                    'setUpFRExc']
                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # cross-homeo with presynaptic multiplier (elementwise)
                # elementwise product results in units of Hz ^ 2
                # given units of alpha we divide by Hz again to convert to amps
                dwEECH = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc) / Hz
                dwEICH = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostInh) / Hz
                dwIECH = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostExc) / Hz
                dwIICH = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh) / Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIEH = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwIIH = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                dwEE = dwEECH.reshape(-1, 1) + dwEEH
                dwEI = dwEICH.reshape(-1, 1) + dwEIH
                dwIE = dwIECH.reshape(-1, 1) + dwIEH
                dwII = dwIICH.reshape(-1, 1) + dwIIH

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE.mean() / pA
                self.trialdwEIUnits[trialInd, :] = dwEI.mean() / pA
                self.trialdwIEUnits[trialInd, :] = dwIE.mean() / pA
                self.trialdwIIUnits[trialInd, :] = dwII.mean() / pA

                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo2':

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nIncomingInhOntoEachExc
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh
                rEI = aEI * movAvgUpFRInhUnits.reshape(-1, 1)
                rIE = aIE * movAvgUpFRExcUnits.reshape(-1, 1)

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                dwEE = p['alpha1'] * (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)
                dwEI = -p['alpha1'] * (p['setUpFRInh'] - rEI)
                dwIE = -p['alpha1'] * (p['setUpFRExc'] - rIE)
                dwII = p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE / pA
                self.trialdwEIUnits[trialInd, :] = dwEI.mean(0) / pA  # this is wrong
                self.trialdwIEUnits[trialInd, :] = dwIE.mean(0) / pA
                self.trialdwIIUnits[trialInd, :] = dwII / pA

                # print(dwEE.shape, dwEI.shape, dwIE.shape, dwII.shape)
                wEEMat += dwEE / pA * JN.p['wEEScale']
                wEIMat += dwEI / pA * JN.p['wEIScale']
                wIEMat += dwIE / pA * JN.p['wIEScale']
                wIIMat += dwII / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'balance':

                # customized weight change version

                # start by converting weights to matrices
                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # calculate the average firing rate of exc units that are presynaptic to each exc unit
                # movAvgUpFRExcUnitsPreToPostExc = np.matmul(movAvgUpFRExcUnits, aEE) / nIncomingExcOntoEachExc
                movAvgUpFRExcUnitsPreToPostExc = movAvgUpFRExcUnits.copy()
                # movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh

                # check if this is less than 1... if so, make it be 1 Hz
                # is this correct? we're sort of forcing all units to be involved...
                movAvgUpFRExcUnitsPreToPostExc[movAvgUpFRExcUnitsPreToPostExc < 1 * Hz] = 1 * Hz
                # movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                # movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                # check if this is greater than 2 * set-point, if so, make it be 2 * set-point
                movAvgUpFRExcUnitsPreToPostExc[movAvgUpFRExcUnitsPreToPostExc > 2 * p['setUpFRExc']] = \
                    2 * p['setUpFRExc']
                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # take the log to prevent insanely large weight modifications during explosions...
                if p['applyLogToFR']:
                    movAvgUpFRExcUnitsPreToPostExcLog = np.log2(movAvgUpFRExcUnitsPreToPostExc / Hz + 1) * Hz
                else:
                    movAvgUpFRExcUnitsPreToPostExcLog = movAvgUpFRExcUnitsPreToPostExc

                # weight change is proportional to
                # the error in the post-synaptic FR times the pre-synaptic FR
                # and takes the form of an outer product
                # the first array is simply the error in the post-synaptic FR
                # each element of the second array is the average FR across E units pre-synaptic to that E unit

                # the pre-unit avg FR and post-unit FR errors are both vectors
                # we take the outer product with the pre-unit avg first (column)
                # times the post-unit error second (row)
                # our weight mats are formatted (pre, post) so this works...
                # i.e. each element represents how much to change that weight

                # when we do np.outer, the units fall off (would have been in Hz^2)
                # alpha says how much to change the weight in pA / Hz / Hz

                # (simpler version...)
                # dwEE = p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnits)
                # wEEMat += dwEE / pA * JN.p['wEEScale']

                # (complex version...)
                dwEEMat = p['alpha1'] * np.outer(movAvgUpFRExcUnitsPreToPostExcLog,
                                                 (p['setUpFRExc'] - movAvgUpFRExcUnits))
                wEEMat += dwEEMat / pA * JN.p['wEEScale']

                wEE = wEEMat[JN.preEE, JN.posEE] * pA

                # (simpler version...)
                # dwIE = p['alpha2'] * (p['setUpFRInh'] - movAvgUpFRInhUnits)
                # wIEMat += dwIE / pA * JN.p['wIEScale']

                # (complex version...)
                dwIEMat = p['alpha2'] * np.outer(movAvgUpFRExcUnitsPreToPostExcLog,
                                                 (p['setUpFRInh'] - movAvgUpFRInhUnits))
                wIEMat += dwIEMat / pA * JN.p['wIEScale']

                wIE = wIEMat[JN.preIE, JN.posIE] * pA

                # given the total excitatory input to each E unit,
                # there is an exact amount of inhibition that will result in the the desired relation
                # between the setpoint FRs in the "steady state" (i.e. given that external input
                # is resulting in the desired setpoint FRs)

                # given that we thus know how much inhibitory current is required for each post cell
                # we should modify its incoming inhibitory weights so that they split that equally

                sumExcInputToExc = np.nansum(wEEMat, 0) * pA
                sumInhInputToExc = p['setUpFRExc'] / p['setUpFRInh'] * sumExcInputToExc - \
                                   p['setUpFRExc'] / p['setUpFRInh'] / p['gainExc'] / second - \
                                   p['threshExc'] / p['setUpFRInh'] / second  # amp

                normwEIMat = wEIMat / np.nansum(wEIMat, 0)  # unitless (amp / amp)
                normlinewEI = normwEIMat * sumInhInputToExc  # amp
                dwEIMat = p['alphaBalance'] * (normlinewEI - wEIMat * pA)
                wEIMat += dwEIMat / pA * JN.p['wEIScale']
                wEI = wEIMat[JN.preEI, JN.posEI] * pA

                sumExcInputToInh = np.nansum(wIEMat, 0) * pA
                sumInhInputToInh = p['setUpFRExc'] / p['setUpFRInh'] * sumExcInputToInh - \
                                   1 / p['gainInh'] / second - \
                                   p['threshInh'] / p['setUpFRInh'] / second

                normwIIMat = wIIMat / np.nansum(wIIMat, 0)
                normlinewII = normwIIMat * sumInhInputToInh
                dwIIMat = p['alphaBalance'] * (normlinewII - wIIMat * pA)
                wIIMat += dwIIMat / pA * JN.p['wIIScale']
                wII = wIIMat[JN.preII, JN.posII] * pA

                # (complex version)...
                self.trialdwEEUnits[trialInd, :] = np.nansum(dwEEMat, 0) / pA
                self.trialdwEIUnits[trialInd, :] = np.nansum(dwEIMat, 0) / pA
                self.trialdwIEUnits[trialInd, :] = np.nansum(dwIEMat, 0) / pA
                self.trialdwIIUnits[trialInd, :] = np.nansum(dwIIMat, 0) / pA

            elif p['useRule'] == 'balance2':

                # customized weight change version

                # start by converting weights to matrices
                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # calculate the average firing rate of exc units that are presynaptic to each exc unit
                # movAvgUpFRExcUnitsPreToPostExc = np.matmul(movAvgUpFRExcUnits, aEE) / nIncomingExcOntoEachExc
                movAvgUpFRExcUnitsPreToPostExc = movAvgUpFRExcUnits.copy()
                # movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh

                # check if this is less than 1... if so, make it be 1 Hz
                # is this correct? we're sort of forcing all units to be involved...
                movAvgUpFRExcUnitsPreToPostExc[movAvgUpFRExcUnitsPreToPostExc < 1 * Hz] = 1 * Hz
                # movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                # movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                # check if this is greater than 2 * set-point, if so, make it be 2 * set-point
                movAvgUpFRExcUnitsPreToPostExc[movAvgUpFRExcUnitsPreToPostExc > 2 * p['setUpFRExc']] = \
                    2 * p['setUpFRExc']
                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # take the log to prevent insanely large weight modifications during explosions...
                if p['applyLogToFR']:
                    movAvgUpFRExcUnitsPreToPostExcLog = np.log2(movAvgUpFRExcUnitsPreToPostExc / Hz + 1) * Hz
                else:
                    movAvgUpFRExcUnitsPreToPostExcLog = movAvgUpFRExcUnitsPreToPostExc

                # weight change is proportional to
                # the error in the post-synaptic FR times the pre-synaptic FR
                # and takes the form of an outer product
                # the first array is simply the error in the post-synaptic FR
                # each element of the second array is the average FR across E units pre-synaptic to that E unit

                # the pre-unit avg FR and post-unit FR errors are both vectors
                # we take the outer product with the pre-unit avg first (column)
                # times the post-unit error second (row)
                # our weight mats are formatted (pre, post) so this works...
                # i.e. each element represents how much to change that weight

                # when we do np.outer, the units fall off (would have been in Hz^2)
                # alpha says how much to change the weight in pA / Hz / Hz

                # (simpler version...)
                # dwEE = p['alpha1'] * (p['setUpFRExc'] - movAvgUpFRExcUnits)
                # wEEMat += dwEE / pA * JN.p['wEEScale']

                # (complex version...)
                dwEEMat = p['alpha1'] * np.outer(movAvgUpFRExcUnitsPreToPostExcLog,
                                                 (p['setUpFRExc'] - movAvgUpFRExcUnits))
                wEEMat += dwEEMat / pA * JN.p['wEEScale']

                wEE = wEEMat[JN.preEE, JN.posEE] * pA

                # (simpler version...)
                # dwIE = p['alpha2'] * (p['setUpFRInh'] - movAvgUpFRInhUnits)
                # wIEMat += dwIE / pA * JN.p['wIEScale']

                # (complex version...)
                dwIEMat = p['alpha2'] * np.outer(movAvgUpFRExcUnitsPreToPostExcLog,
                                                 (p['setUpFRInh'] - movAvgUpFRInhUnits))
                wIEMat += dwIEMat / pA * JN.p['wIEScale']

                wIE = wIEMat[JN.preIE, JN.posIE] * pA

                # given the total excitatory input to each E unit,
                # there is an exact amount of inhibition that will result in the the desired relation
                # between the setpoint FRs in the "steady state" (i.e. given that external input
                # is resulting in the desired setpoint FRs)

                # given that we thus know how much inhibitory current is required for each post cell
                # we should modify its incoming inhibitory weights so that they split that equally

                # here we could actually calculate the total charge by estimating
                # summed excitatory weights * setpoint FR * duration

                # these are empirically estimated
                sustainingWeightImbalanceExc = 98 * nA
                sustainingWeightImbalanceInh = 108 * nA
                sustainingWeightRatioExc = 0.658
                sustainingWeightRatioInh = 0.564
                slopeExc = p['setUpFRExc'] / p['setUpFRInh']
                slopeInh = p['setUpFRExc'] / p['setUpFRInh']
                offsetExc = 86.5 * nA
                offsetInh = 51.4 * nA

                sumExcInputToExc = np.nansum(wEEMat, 0) * float(
                    p['setUpFRExc']) * pA  # in 1 second... should be multiplied by duration???

                # sumInhInputToExc = (sumExcInputToExc - sustainingWeightImbalanceExc) / float(p['setUpFRInh'])
                # sumInhInputToExc = sumExcInputToExc * sustainingWeightRatioExc / float(p['setUpFRInh'])
                sumInhInputToExc = (offsetExc + slopeExc * sumExcInputToExc) / float(p['setUpFRInh'])

                normwEIMat = wEIMat / np.nansum(wEIMat, 0)  # unitless (amp / amp)
                normlinewEI = normwEIMat * sumInhInputToExc  # amp
                dwEIMat = p['alphaBalance'] * (normlinewEI - wEIMat * pA)
                wEIMat += dwEIMat / pA * JN.p['wEIScale']
                wEI = wEIMat[JN.preEI, JN.posEI] * pA

                sumExcInputToInh = np.nansum(wIEMat, 0) * float(p['setUpFRExc']) * pA

                # sumInhInputToInh = (sumExcInputToInh - sustainingWeightImbalanceInh) / float(p['setUpFRInh'])
                # sumInhInputToInh = sumExcInputToInh * sustainingWeightRatioInh / float(p['setUpFRInh'])
                sumInhInputToInh = (offsetInh + slopeInh * sumExcInputToInh) / float(p['setUpFRInh'])

                normwIIMat = wIIMat / np.nansum(wIIMat, 0)
                normlinewII = normwIIMat * sumInhInputToInh
                dwIIMat = p['alphaBalance'] * (normlinewII - wIIMat * pA)
                wIIMat += dwIIMat / pA * JN.p['wIIScale']
                wII = wIIMat[JN.preII, JN.posII] * pA

                # (complex version)...
                self.trialdwEEUnits[trialInd, :] = np.nansum(dwEEMat, 0) / pA
                self.trialdwEIUnits[trialInd, :] = np.nansum(dwEIMat, 0) / pA
                self.trialdwIEUnits[trialInd, :] = np.nansum(dwIEMat, 0) / pA
                self.trialdwIIUnits[trialInd, :] = np.nansum(dwIIMat, 0) / pA

            wEETooSmall = wEE < p['minAllowedWEE']
            wIETooSmall = wIE < p['minAllowedWIE']
            wEITooSmall = wEI < p['minAllowedWEI']
            wIITooSmall = wII < p['minAllowedWII']
            if wEETooSmall.any():
                print('at least one weight was below the minimum allowed')
                wEE[wEETooSmall] = p['minAllowedWEE']
            if wIETooSmall.any():
                print('at least one weight was below the minimum allowed')
                wIE[wIETooSmall] = p['minAllowedWIE']
            if wEITooSmall.any():
                print('at least one weight was below the minimum allowed')
                wEI[wEITooSmall] = p['minAllowedWEI']
            if wIITooSmall.any():
                print('at least one weight was below the minimum allowed')
                wII[wIITooSmall] = p['minAllowedWII']

            if p['useRule'][:5] == 'cross' or p['useRule'] == 'homeo':
                print(sumWeightMsgFormatter.format(movAvgUpFRExc, movAvgUpFRInh, dwEE.sum() * JN.p['wEEScale'] / pA,
                                                   dwIE.sum() * JN.p['wIEScale'] / pA,
                                                   dwEI.sum() * JN.p['wEIScale'] / pA,
                                                   dwII.sum() * JN.p['wIIScale'] / pA))
                print(meanWeightChangeMsgFormatter.format(dwEE.mean() * JN.p['wEEScale'] / pA,
                                                          dwIE.mean() * JN.p['wIEScale'] / pA,
                                                          dwEI.mean() * JN.p['wEIScale'] / pA,
                                                          dwII.mean() * JN.p['wIIScale'] / pA))
            elif p['useRule'][:7] == 'balance':
                print(sumWeightMsgFormatter.format(movAvgUpFRExc, movAvgUpFRInh,
                                                   np.nansum(dwEEMat) * JN.p['wEEScale'] / pA,
                                                   np.nansum(dwIEMat) * JN.p['wIEScale'] / pA,
                                                   np.nansum(dwEIMat) * JN.p['wEIScale'] / pA,
                                                   np.nansum(dwIIMat) * JN.p['wIIScale'] / pA))
                print(meanWeightChangeMsgFormatter.format(np.nanmean(dwEEMat) * JN.p['wEEScale'] / pA,
                                                          np.nanmean(dwIEMat) * JN.p['wIEScale'] / pA,
                                                          np.nanmean(dwEIMat) * JN.p['wEIScale'] / pA,
                                                          np.nanmean(dwIIMat) * JN.p['wIIScale'] / pA))

        # close pdf
        pdfObject.close()

        #
        t1_overall = datetime.now()
        print('the whole training session took:', t1_overall - t0_overall)

        # assign some objects to pass to saving the results
        self.wEE = wEE
        self.wIE = wIE
        self.wEI = wEI
        self.wII = wII
        self.preEE = JN.preEE
        self.preIE = JN.preIE
        self.preEI = JN.preEI
        self.preII = JN.preII
        self.posEE = JN.posEE
        self.posIE = JN.posIE
        self.posEI = JN.posEI
        self.posII = JN.posII

    def run_upCrit(self):

        JN = self.JN

        wEE = self.wEE_init.copy()
        wEI = self.wEI_init.copy()
        wIE = self.wIE_init.copy()
        wII = self.wII_init.copy()

        # set the weights (separately for each unit)
        JN.synapsesEE.jEE = wEE
        JN.synapsesEI.jEI = wEI
        JN.synapsesIE.jIE = wIE
        JN.synapsesII.jII = wII

        # run the simulation
        JN.run()

        # assign some objects to pass to saving the results
        self.wEE = wEE
        self.wIE = wIE
        self.wEI = wEI
        self.wII = wII
        self.preEE = JN.preEE
        self.preIE = JN.preIE
        self.preEI = JN.preEI
        self.preII = JN.preII
        self.posEE = JN.posEE
        self.posIE = JN.posIE
        self.posEI = JN.posEI
        self.posII = JN.posII

    def save_params(self):
        savePath = os.path.join(self.p['saveFolder'], self.saveName + '_params.pkl')
        with open(savePath, 'wb') as f:
            dill.dump(self.p, f)

    def save_results(self):
        savePath = os.path.join(self.p['saveFolder'], self.saveName + '_results.npz')

        saveDict = {
            'trialUpFRExc': self.trialUpFRExc,
            'trialUpFRInh': self.trialUpFRInh,
            'trialUpDur': self.trialUpDur,
            'trialwEE': self.trialwEE,
            'trialwEI': self.trialwEI,
            'trialwIE': self.trialwIE,
            'trialwII': self.trialwII,
            'trialdwEEUnits': self.trialdwEEUnits,
            'trialdwEIUnits': self.trialdwEIUnits,
            'trialdwIEUnits': self.trialdwIEUnits,
            'trialdwIIUnits': self.trialdwIIUnits,
            'trialUpFRExcUnits': self.trialUpFRExcUnits,
            'trialUpFRInhUnits': self.trialUpFRInhUnits,
            'wEE_init': self.wEE_init / pA,
            'wIE_init': self.wIE_init / pA,
            'wEI_init': self.wEI_init / pA,
            'wII_init': self.wII_init / pA,
            'wEE_final': self.wEE / pA,
            'wIE_final': self.wIE / pA,
            'wEI_final': self.wEI / pA,
            'wII_final': self.wII / pA,
            'preEE': self.preEE,
            'preIE': self.preIE,
            'preEI': self.preEI,
            'preII': self.preII,
            'posEE': self.posEE,
            'posIE': self.posIE,
            'posEI': self.posEI,
            'posII': self.posII,
        }

        if hasattr(self, 'trialdwEECHUnits'):
            saveDictAdd = {
                'trialdwEECHUnits': self.trialdwEECHUnits,
                'trialdwEICHUnits': self.trialdwEICHUnits,
                'trialdwIECHUnits': self.trialdwIECHUnits,
                'trialdwIICHUnits': self.trialdwIICHUnits,
            }
            saveDict.update(saveDictAdd)

        if hasattr(self, 'trialdwEEHUnits'):
            saveDictAdd = {
                'trialdwEEHUnits': self.trialdwEEHUnits,
                'trialdwEIHUnits': self.trialdwEIHUnits,
                'trialdwIEHUnits': self.trialdwIEHUnits,
                'trialdwIIHUnits': self.trialdwIIHUnits,
            }
            saveDict.update(saveDictAdd)

        if hasattr(self, 'trialdwEEOUnits'):
            saveDictAdd = {
                'trialdwEEOUnits': self.trialdwEEOUnits,
                'trialdwEIOUnits': self.trialdwEIOUnits,
                'trialdwIEOUnits': self.trialdwIEOUnits,
                'trialdwIIOUnits': self.trialdwIIOUnits,
            }
            saveDict.update(saveDictAdd)

        if self.p['recordMovieVariables']:
            saveDictMovie = {
                'selectTrialT': self.selectTrialT,
                'selectTrialVExc': self.selectTrialVExc,
                'selectTrialVInh': self.selectTrialVInh,
                'selectTrialFRExc': self.selectTrialFRExc,
                'selectTrialFRInh': self.selectTrialFRInh,
                'selectTrialSpikeExcI': self.selectTrialSpikeExcI,
                'selectTrialSpikeExcT': self.selectTrialSpikeExcT,
                'selectTrialSpikeInhI': self.selectTrialSpikeInhI,
                'selectTrialSpikeInhT': self.selectTrialSpikeInhT,
            }
            saveDict.update(saveDictMovie)

        np.savez(savePath, **saveDict)

    def save_results_upCrit(self):
        savePath = os.path.join(self.p['saveFolder'], self.saveName + '_results.npz')

        useDType = np.single

        spikeMonExcT = np.array(self.JN.spikeMonExc.t, dtype=useDType)
        spikeMonExcI = np.array(self.JN.spikeMonExc.i, dtype=useDType)
        spikeMonInhT = np.array(self.JN.spikeMonInh.t, dtype=useDType)
        spikeMonInhI = np.array(self.JN.spikeMonInh.i, dtype=useDType)
        stateMonExcV = np.array(self.JN.stateMonExc.v / mV, dtype=useDType)
        stateMonInhV = np.array(self.JN.stateMonInh.v / mV, dtype=useDType)

        saveDict = {
            'wEE_init': self.wEE_init / pA,
            'wIE_init': self.wIE_init / pA,
            'wEI_init': self.wEI_init / pA,
            'wII_init': self.wII_init / pA,
            'preEE': self.preEE,
            'preIE': self.preIE,
            'preEI': self.preEI,
            'preII': self.preII,
            'posEE': self.posEE,
            'posIE': self.posIE,
            'posEI': self.posEI,
            'posII': self.posII,
            'spikeMonExcT': spikeMonExcT,
            'spikeMonExcI': spikeMonExcI,
            'spikeMonInhT': spikeMonInhT,
            'spikeMonInhI': spikeMonInhI,
            'stateMonExcV': stateMonExcV,
            'stateMonInhV': stateMonInhV,
        }

        np.savez(savePath, **saveDict)
