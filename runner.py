"""
a class that consumes a network and some parameters,
then runs an experiment and saves the results.
slightly redundant here because it was mainly developed for doing training sessions
"""

import os
from datetime import datetime
import dill
import numpy as np
from brian2 import ms, nA, pA, mV
from network import JercogEphysNetwork, JercogNetwork
from results import ResultsEphys


class JercogRunner(object):

    def __init__(self, p):
        self.p = p
        self.p['initTime'] = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
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

    def set_up_network(self, priorResults=None, recordAllVoltage=False):
        # set up network, experiment, and start recording
        JN = JercogNetwork(self.p)
        JN.initialize_network()
        JN.initialize_units()
        JN.prepare_upPoisson_experiment(poissonLambda=self.p['poissonLambda'],
                                        duration=self.p['duration'],
                                        spikeUnits=self.p['nUnitsToSpike'],
                                        rng=self.p['rng'],
                                        currentAmp=self.p['spikeInputAmplitude'])
        if priorResults is not None:
            JN.initialize_synapses_results(priorResults)
        else:
            JN.initialize_synapses()

        if recordAllVoltage:
            JN.create_monitors_allVoltage()
        else:
            JN.create_monitors()

        self.JN = JN

    def initialize_weight_matrices(self):

        if self.p['initWeightMethod'] == 'resumePrior':
            self.wEE_init = self.JN.synapsesEE.jEE[:]
            self.wIE_init = self.JN.synapsesIE.jIE[:]
            self.wEI_init = self.JN.synapsesEI.jEI[:]
            self.wII_init = self.JN.synapsesII.jII[:]

    def run(self):

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
