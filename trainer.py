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
from brian2 import ms, nA, pA, Hz, second
from network import JercogEphysNetwork, JercogNetwork
from results import ResultsEphys, Results
from generate import lognormal_positive_weights, normal_positive_weights, adjacency_matrix_from_flat_inds, weight_matrix_from_flat_inds_weights


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
            self.wEE_init = self.JN.synapsesEE.jEE[:] * normal_positive_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng'])
            self.wIE_init = self.JN.synapsesIE.jIE[:] * normal_positive_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng'])
            self.wEI_init = self.JN.synapsesEI.jEI[:] * normal_positive_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng'])
            self.wII_init = self.JN.synapsesII.jII[:] * normal_positive_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng'])
        elif self.p['initWeightMethod'] == 'defaultNormalScaled':
            self.wEE_init = self.JN.synapsesEE.jEE[:] * normal_positive_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng']) * self.p['jEEScaleRatio']
            self.wIE_init = self.JN.synapsesIE.jIE[:] * normal_positive_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng']) * self.p['jIEScaleRatio']
            self.wEI_init = self.JN.synapsesEI.jEI[:] * normal_positive_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng']) * self.p['jEIScaleRatio']
            self.wII_init = self.JN.synapsesII.jII[:] * normal_positive_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng']) * self.p['jIIScaleRatio']
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
            self.wEE_init = wEE_mean * normal_positive_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * normal_positive_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * normal_positive_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * normal_positive_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessGoodWeights2e3p025':
            wEE_mean = 120
            wIE_mean = 90
            wEI_mean = 104
            wII_mean = 54
            self.wEE_init = wEE_mean * normal_positive_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * normal_positive_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * normal_positive_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * normal_positive_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessGoodWeights2e3p025LogNormal':
            wEE_mean = 120
            wIE_mean = 90
            wEI_mean = 104
            wII_mean = 54
            self.wEE_init = wEE_mean * lognormal_positive_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognormal_positive_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognormal_positive_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognormal_positive_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessZeroActivityWeights2e3p025':
            wEE_mean = 62
            wIE_mean = 62
            wEI_mean = 250
            wII_mean = 250
            self.wEE_init = wEE_mean * normal_positive_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * normal_positive_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * normal_positive_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * normal_positive_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'] == 'guessLowActivityWeights2e3p025':
            wEE_mean = 114
            wIE_mean = 82
            wEI_mean = 78
            wII_mean = 20
            self.wEE_init = wEE_mean * normal_positive_weights(self.JN.synapsesEE.jEE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * normal_positive_weights(self.JN.synapsesIE.jIE[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * normal_positive_weights(self.JN.synapsesEI.jEI[:].size, 1, 0.2, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * normal_positive_weights(self.JN.synapsesII.jII[:].size, 1, 0.2, rng=self.p['rng']) * pA
        elif self.p['initWeightMethod'][:4] == 'seed':
            excMeanWeightsPossible = (75, 112.5, 150)
            inhMeanWeightsPossible = (700, 450, 200)
            # mappingStringSeq = (('wIE', 'wEE'), ('wII', 'wEI'),)
            excWeightTupleList = list(product(excMeanWeightsPossible, excMeanWeightsPossible))
            inhWeightTupleList = list(product(inhMeanWeightsPossible, inhMeanWeightsPossible))
            useSeed = int(self.p['initWeightMethod'][-1])  # should be a value 0-8
            wIE_mean, wEE_mean = excWeightTupleList[useSeed]
            wII_mean, wEI_mean = inhWeightTupleList[useSeed]
            self.wEE_init = wEE_mean * lognormal_positive_weights(self.JN.synapsesEE.jEE[:].size, rng=self.p['rng']) * pA
            self.wIE_init = wIE_mean * lognormal_positive_weights(self.JN.synapsesIE.jIE[:].size, rng=self.p['rng']) * pA
            self.wEI_init = wEI_mean * lognormal_positive_weights(self.JN.synapsesEI.jEI[:].size, rng=self.p['rng']) * pA
            self.wII_init = wII_mean * lognormal_positive_weights(self.JN.synapsesII.jII[:].size, rng=self.p['rng']) * pA

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
        aIE = adjacency_matrix_from_flat_inds(p['nExc'], p['nInh'], JN.preIE, JN.posIE)
        aEI = adjacency_matrix_from_flat_inds(p['nInh'], p['nExc'], JN.preEI, JN.posEI)
        aII = adjacency_matrix_from_flat_inds(p['nInh'], p['nInh'], JN.preII, JN.posII)
        nPreEE = aEE.sum(0)
        nPreEI = aEI.sum(0)
        nPreIE = aIE.sum(0)
        nPreII = aII.sum(0)

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
                break
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
                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nPreEI
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nPreIE

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

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nPreEI
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nPreIE

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']
                movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']

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
                dwEE = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)) * Hz
                dwEI = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)) * Hz
                dwIE = -p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)) * Hz
                dwII = p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)) * Hz

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
                dwEE = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwEI = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / Hz
                dwIE = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz
                dwII = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / Hz

                # save the proposed weight change in pA
                self.trialdwEEUnits[trialInd, :] = dwEE.mean() / pA
                self.trialdwEIUnits[trialInd, :] = dwEI.mean() / pA
                self.trialdwIEUnits[trialInd, :] = dwIE.mean() / pA
                self.trialdwIIUnits[trialInd, :] = dwII.mean() / pA

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                wEEMat += dwEE.reshape(-1, 1) / pA * JN.p['wEEScale']
                wEIMat += dwEI.reshape(-1, 1) / pA * JN.p['wEIScale']
                wIEMat += dwIE.reshape(-1, 1) / pA * JN.p['wIEScale']
                wIIMat += dwII.reshape(-1, 1) / pA * JN.p['wIIScale']

                # reshape back to a matrix
                wEE = wEEMat[JN.preEE, JN.posEE] * pA
                wEI = wEIMat[JN.preEI, JN.posEI] * pA
                wIE = wIEMat[JN.preIE, JN.posIE] * pA
                wII = wIIMat[JN.preII, JN.posII] * pA

            elif p['useRule'] == 'cross-homeo-pre-elementwise':

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nPreEI
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nPreIE
                movAvgUpFRInhUnitsPreToPostInh = np.matmul(movAvgUpFRInhUnits, aII) / nPreII
                movAvgUpFRExcUnitsPreToPostExc = np.matmul(movAvgUpFRExcUnits, aEE) / nPreEE

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']
                movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnitsPreToPostInh[movAvgUpFRInhUnitsPreToPostInh > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']
                movAvgUpFRExcUnitsPreToPostExc[movAvgUpFRExcUnitsPreToPostExc > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
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

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nPreEI
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nPreIE

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']
                movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']

                movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

                # convert flat weight arrays into matrices in units of pA
                wEEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nExc'], JN.preEE, JN.posEE, wEE / pA)
                wEIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nExc'], JN.preEI, JN.posEI, wEI / pA)
                wIEMat = weight_matrix_from_flat_inds_weights(p['nExc'], p['nInh'], JN.preIE, JN.posIE, wIE / pA)
                wIIMat = weight_matrix_from_flat_inds_weights(p['nInh'], p['nInh'], JN.preII, JN.posII, wII / pA)

                # cross-homeo with presynaptic multiplier (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEE1 = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)) * Hz
                dwEI1 = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc)) * Hz
                dwIE1 = -p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)) * Hz
                dwII1 = p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh)) * Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEE2 = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEI2 = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIE2 = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwII2 = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                dwEE = dwEE1 + dwEE2
                dwEI = dwEI1 + dwEI2
                dwIE = dwIE1 + dwIE2
                dwII = dwII1 + dwII2

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

            elif p['useRule'] == 'homeo':

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nPreEI
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nPreIE

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']
                movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']

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

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo':

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

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEE2 = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEI2 = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIE2 = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwII2 = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEE1.reshape(-1, 1) + dwEE2
                dwEI = dwEI1.reshape(-1, 1) + dwEI2
                dwIE = dwIE1.reshape(-1, 1) + dwIE2
                dwII = dwII1.reshape(-1, 1) + dwII2

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

                dwEE1 = p['alpha1'] * movAvgUpFRExcUnits / setUpE * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / setUpI / Hz
                dwEI1 = -p['alpha1'] * movAvgUpFRInhUnits / setUpI * (p['setUpFRInh'] - movAvgUpFRInhUnits.mean()) / setUpI / Hz
                dwIE1 = -p['alpha1'] * movAvgUpFRExcUnits / setUpE * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / setUpE / Hz
                dwII1 = p['alpha1'] * movAvgUpFRInhUnits / setUpI * (p['setUpFRExc'] - movAvgUpFRExcUnits.mean()) / setUpE / Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEE2 = p['alpha1'] * np.outer(movAvgUpFRExcUnits / setUpE, (p['setUpFRExc'] - movAvgUpFRExcUnits) / setUpE) * Hz
                dwEI2 = -p['alpha1'] * np.outer(movAvgUpFRInhUnits / setUpI, (p['setUpFRExc'] - movAvgUpFRExcUnits) / setUpE) * Hz
                dwIE2 = p['alpha1'] * np.outer(movAvgUpFRExcUnits / setUpE, (p['setUpFRInh'] - movAvgUpFRInhUnits) / setUpI) * Hz
                dwII2 = -p['alpha1'] * np.outer(movAvgUpFRInhUnits / setUpI, (p['setUpFRInh'] - movAvgUpFRInhUnits) / setUpI) * Hz

                # this broadcasts the addition across the COLUMNS (the 1d dw arrays are column vectors)
                # this applies the same weight change to all OUTGOING synapses from a single unit
                # but it's a different value for each unit
                dwEE = dwEE1.reshape(-1, 1) + dwEE2
                dwEI = dwEI1.reshape(-1, 1) + dwEI2
                dwIE = dwIE1.reshape(-1, 1) + dwIE2
                dwII = dwII1.reshape(-1, 1) + dwII2

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

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nPreEI
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nPreIE
                movAvgUpFRInhUnitsPreToPostInh = np.matmul(movAvgUpFRInhUnits, aII) / nPreII
                movAvgUpFRExcUnitsPreToPostExc = np.matmul(movAvgUpFRExcUnits, aEE) / nPreEE

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
                dwEE1 = p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostExc) / Hz
                dwEI1 = -p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRInh'] - movAvgUpFRInhUnitsPreToPostInh) / Hz
                dwIE1 = -p['alpha1'] * movAvgUpFRExcUnits * (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostExc) / Hz
                dwII1 = p['alpha1'] * movAvgUpFRInhUnits * (p['setUpFRExc'] - movAvgUpFRExcUnitsPreToPostInh) / Hz

                # regular homeo (outer product)
                # since outer strips units and because of alpha we multiply by Hz to convert to amps
                dwEE2 = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwEI2 = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRExc'] - movAvgUpFRExcUnits)) * Hz
                dwIE2 = p['alpha1'] * np.outer(movAvgUpFRExcUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz
                dwII2 = -p['alpha1'] * np.outer(movAvgUpFRInhUnits, (p['setUpFRInh'] - movAvgUpFRInhUnits)) * Hz

                dwEE = dwEE1.reshape(-1, 1) + dwEE2
                dwEI = dwEI1.reshape(-1, 1) + dwEI2
                dwIE = dwIE1.reshape(-1, 1) + dwIE2
                dwII = dwII1.reshape(-1, 1) + dwII2

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

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nPreEI
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nPreIE
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
                # movAvgUpFRExcUnitsPreToPostExc = np.matmul(movAvgUpFRExcUnits, aEE) / nPreEE
                movAvgUpFRExcUnitsPreToPostExc = movAvgUpFRExcUnits.copy()
                # movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nPreIE

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
                dwEEMat = p['alpha1'] * np.outer(movAvgUpFRExcUnitsPreToPostExcLog, (p['setUpFRExc'] - movAvgUpFRExcUnits))
                wEEMat += dwEEMat / pA * JN.p['wEEScale']

                wEE = wEEMat[JN.preEE, JN.posEE] * pA

                # (simpler version...)
                # dwIE = p['alpha2'] * (p['setUpFRInh'] - movAvgUpFRInhUnits)
                # wIEMat += dwIE / pA * JN.p['wIEScale']

                # (complex version...)
                dwIEMat = p['alpha2'] * np.outer(movAvgUpFRExcUnitsPreToPostExcLog, (p['setUpFRInh'] - movAvgUpFRInhUnits))
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
                # movAvgUpFRExcUnitsPreToPostExc = np.matmul(movAvgUpFRExcUnits, aEE) / nPreEE
                movAvgUpFRExcUnitsPreToPostExc = movAvgUpFRExcUnits.copy()
                # movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nPreIE

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
                dwEEMat = p['alpha1'] * np.outer(movAvgUpFRExcUnitsPreToPostExcLog, (p['setUpFRExc'] - movAvgUpFRExcUnits))
                wEEMat += dwEEMat / pA * JN.p['wEEScale']

                wEE = wEEMat[JN.preEE, JN.posEE] * pA

                # (simpler version...)
                # dwIE = p['alpha2'] * (p['setUpFRInh'] - movAvgUpFRInhUnits)
                # wIEMat += dwIE / pA * JN.p['wIEScale']

                # (complex version...)
                dwIEMat = p['alpha2'] * np.outer(movAvgUpFRExcUnitsPreToPostExcLog, (p['setUpFRInh'] - movAvgUpFRInhUnits))
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

                sumExcInputToExc = np.nansum(wEEMat, 0) * float(p['setUpFRExc']) * pA  # in 1 second... should be multiplied by duration???

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
                wEE[wEETooSmall] = p['minAllowedWEE']
            if wIETooSmall.any():
                wIE[wIETooSmall] = p['minAllowedWIE']
            if wEITooSmall.any():
                wEI[wEITooSmall] = p['minAllowedWEI']
            if wIITooSmall.any():
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
