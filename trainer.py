"""
a class that consumes a network and some parameters,
then uses them to run many trials,
modifying weights in between each trial...
"""

import os
from datetime import datetime
import dill
import numpy as np
from brian2 import ms, nA, pA, Hz, second, mV
from network import JercogEphysNetwork, JercogNetwork, DestexheNetwork
from results import ResultsEphys, Results
from generate import norm_weights, adjacency_matrix_from_flat_inds, \
    weight_matrix_from_flat_inds_weights


class JercogTrainer(object):

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

        # testing an experimental modification to these values...
        self.p['threshExc'] = RE.threshExc  # / 13
        self.p['threshInh'] = RE.threshInh  # / 13
        self.p['gainExc'] = RE.gainExc
        self.p['gainInh'] = RE.gainInh

    def set_up_network(self, priorResults=None):
        # set up network, experiment, and start recording
        JN = JercogNetwork(self.p)
        JN.initialize_network()
        JN.initialize_units()

        if self.p['kickType'] == 'kick':
            JN.set_kicked_units(onlyKickExc=self.p['onlyKickExc'])
        elif self.p['kickType'] == 'spike':
            JN.prepare_upCrit_experiment2(minUnits=self.p['nUnitsToSpike'], maxUnits=self.p['nUnitsToSpike'],
                                          unitSpacing=5,  # unitSpacing is a useless input in this context
                                          timeSpacing=self.p['timeAfterSpiked'], startTime=self.p['timeToSpike'],
                                          currentAmp=self.p['spikeInputAmplitude'])
            # JN.prepare_upCrit_random(nUnits=self.p['nUnitsToSpike'], timeSpacing=self.p['timeAfterSpiked'],
            #                          startTime=self.p['timeToSpike'], currentAmp=self.p['spikeInputAmplitude'])
        if priorResults is not None:
            JN.initialize_synapses_from_results(priorResults)
        else:
            JN.initialize_synapses()

        JN.create_monitors()
        self.JN = JN

    def set_up_network_upCrit(self, priorResults=None, recordAllVoltage=False):
        JN = JercogNetwork(self.p)
        JN.initialize_network()
        JN.initialize_units()

        if self.p['kickType'] == 'kick':
            JN.set_kicked_units(onlyKickExc=self.p['onlyKickExc'])
        elif self.p['kickType'] == 'spike':
            JN.prepare_upCrit_experiment2(minUnits=self.p['nUnitsToSpike'], maxUnits=self.p['nUnitsToSpike'],
                                          unitSpacing=5,  # unitSpacing is a useless input in this context
                                          timeSpacing=self.p['timeAfterSpiked'], startTime=self.p['timeToSpike'],
                                          currentAmp=self.p['spikeInputAmplitude'])
        elif self.p['kickType'] == 'barrage':
            JN.initialize_external_input_uncorrelated(currentAmpExc=self.p['poissonUncorrInputAmpExc'],
                                                      currentAmpInh=self.p['poissonUncorrInputAmpInh'],)

        if priorResults is not None:
            JN.initialize_synapses_from_results(priorResults)
        else:
            JN.initialize_synapses()

        if recordAllVoltage:
            JN.create_monitors_allVoltage()
        else:
            JN.create_monitors()

        self.JN = JN

    def set_up_network_Poisson(self, priorResults=None, recordAllVoltage=False):
        # set up network, experiment, and start recording
        JN = JercogNetwork(self.p)
        JN.initialize_network()
        JN.initialize_units()
        JN.prepare_upPoisson_experiment(poissonLambda=self.p['poissonLambda'],
                                        duration=self.p['duration'],
                                        spikeUnits=self.p['nUnitsToSpike'],
                                        rng=self.p['rng'])
        if priorResults is not None:
            JN.initialize_synapses_from_results(priorResults)
        else:
            JN.initialize_synapses()

        if recordAllVoltage:
            JN.create_monitors_allVoltage()
        else:
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

        if self.p['initWeightMethod'] == 'resumePrior':
            self.wEE_init = self.JN.synapsesEE.jEE[:]
            self.wIE_init = self.JN.synapsesIE.jIE[:]
            self.wEI_init = self.JN.synapsesEI.jEI[:]
            self.wII_init = self.JN.synapsesII.jII[:]
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
        elif self.p['initWeightMethod'] == 'defaultUniform':
            self.wEE_init = self.p['rng'].random(self.JN.synapsesEE.jEE[:].size) * 2 * self.JN.synapsesEE.jEE[0]
            self.wIE_init = self.p['rng'].random(self.JN.synapsesIE.jIE[:].size) * 2 * self.JN.synapsesIE.jIE[0]
            self.wEI_init = self.p['rng'].random(self.JN.synapsesEI.jEI[:].size) * 2 * self.JN.synapsesEI.jEI[0]
            self.wII_init = self.p['rng'].random(self.JN.synapsesII.jII[:].size) * 2 * self.JN.synapsesII.jII[0]

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

        # define message formatters
        meanWeightMsgFormatter = ('upstateFRExc: {:.2f} Hz, upstateFRInh: {:.2f}'
                                  ' Hz, wEE: {:.2f} pA, wIE: {:.2f} pA, wEI: {:.2f} pA, wII: {:.2f} pA')
        sumWeightMsgFormatter = ('movAvgUpFRExc: {:.2f} Hz, movAvgUpFRInh: {:.2f} Hz, '
                                 'dwEE: {:.2f} pA, dwEI: {:.2f} pA, dwIE: {:.2f} pA, dwII: {:.2f} pA')
        meanWeightChangeMsgFormatter = 'mean dwEE: {:.2f} pA, dwEI: {:.2f} pA, dwIE: {:.2f} pA, dwII: {:.2f} pA'

        saveTrialDummy = 0
        for trialInd in range(p['nTrials']):

            print('starting trial {}'.format(trialInd))

            # restore the initial network state
            JN.N.restore()

            # set the weights
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
                # if there were no Up states, just take the avg FR within the time period
                # of nonzero FR
                print('there were no up states')
                spikesPerUnitExc = np.bincount(R.spikeMonExcI.astype(int), minlength=self.p['nExc'])
                spikesPerUnitInh = np.bincount(R.spikeMonInhI.astype(int), minlength=self.p['nInh'])

                # since we are not in the Up state range, this would be considered an event
                # and this is mainly for situations in which WEE is too low to cause any recurrent firing
                # (i.e. only the directly stimulated units fire, for a very short duration)
                # however, it's possible to also have recurrent excitatory spikes but no Up state
                # here, we consider the Up state duration to be the set of indices in which
                # some spikes took place
                # since it should be the same for both, we will take the max (although Exc will almost always win)
                eventDuration = np.fmax((R.FRExc > 0).sum(), (R.FRInh > 0).sum()) * self.p['dtHistPSTH']

                unstimFRExcUnits = spikesPerUnitExc / eventDuration
                unstimFRInhUnits = spikesPerUnitInh / eventDuration

                # here, we avoid using the directly stimulated Exc units
                # and consider their FR to be the average of all others
                unstimFRExcUnits[:p['nUnitsToSpike']] = unstimFRExcUnits[p['nUnitsToSpike']:].mean()

                self.trialUpFRExc[trialInd] = unstimFRExcUnits.mean()
                self.trialUpFRInh[trialInd] = unstimFRInhUnits.mean()
                self.trialUpDur[trialInd] = eventDuration
                self.trialUpFRExcUnits[trialInd, :] = unstimFRExcUnits  # in Hz
                self.trialUpFRInhUnits[trialInd, :] = unstimFRInhUnits  # in Hz

            # separate by unit, record average weight?
            self.trialwEE[trialInd] = wEE.mean() / pA
            self.trialwEI[trialInd] = wEI.mean() / pA
            self.trialwIE[trialInd] = wIE.mean() / pA
            self.trialwII[trialInd] = wII.mean() / pA

            # save numerical results and/or plots!!!
            if saveThisTrial:

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

            if p['useRule'] == 'cross-homeo':  # aka cross-homeo-outer

                # separately by synapse...
                # must create a vector in which each element represents the average FR
                # of the cells that synapse onto the post
                # for wEI & wEE, mean fr of inh units that target each E unit
                # for wIE & wII, mean fr of exc units that target each I unit
                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nIncomingInhOntoEachExc
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh

                # movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = \
                #     2 * p['setUpFRInh']
                # movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = \
                #     2 * p['setUpFRExc']
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

                # movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p[
                #     'setUpFRInh']
                # movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p[
                #     'setUpFRExc']
                #
                # movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                # movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

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

            elif p['useRule'] == 'cross-homeo-pre-scalar':

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                # movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                # movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

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
                wEEMat += dwEE / pA
                wEIMat += dwEI / pA
                wIEMat += dwIE / pA
                wIIMat += dwII / pA

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

            elif p['useRule'] == 'cross-homeo-pre-outer-homeo':

                movAvgUpFRInhUnitsPreToPostExc = np.matmul(movAvgUpFRInhUnits, aEI) / nIncomingInhOntoEachExc
                movAvgUpFRExcUnitsPreToPostInh = np.matmul(movAvgUpFRExcUnits, aIE) / nIncomingExcOntoEachInh

                movAvgUpFRExcUnits[movAvgUpFRExcUnits < 1 * Hz] = 1 * Hz
                movAvgUpFRInhUnits[movAvgUpFRInhUnits < 1 * Hz] = 1 * Hz

                # movAvgUpFRInhUnitsPreToPostExc[movAvgUpFRInhUnitsPreToPostExc > 2 * p['setUpFRInh']] = 2 * p[
                #     'setUpFRInh']
                # movAvgUpFRExcUnitsPreToPostInh[movAvgUpFRExcUnitsPreToPostInh > 2 * p['setUpFRExc']] = 2 * p[
                #     'setUpFRExc']
                #
                # movAvgUpFRExcUnits[movAvgUpFRExcUnits > 2 * p['setUpFRExc']] = 2 * p['setUpFRExc']
                # movAvgUpFRInhUnits[movAvgUpFRInhUnits > 2 * p['setUpFRInh']] = 2 * p['setUpFRInh']

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

            elif p['useRule'] == 'cross-homeo-pre-scalar-homeo':

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

            elif p['useRule'] == 'balance':
                # this is not a good implementation because the y intercept / slope of the balance line are wrong

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

            wEETooBig = wEE > p['maxAllowedWEE']
            wIETooBig = wIE > p['maxAllowedWIE']
            wEITooBig = wEI > p['maxAllowedWEI']
            wIITooBig = wII > p['maxAllowedWII']
            if wEETooBig.any():
                print('at least one weight was above the maximum allowed')
                wEE[wEETooBig] = p['maxAllowedWEE']
            if wIETooBig.any():
                print('at least one weight was above the maximum allowed')
                wIE[wIETooBig] = p['maxAllowedWIE']
            if wEITooBig.any():
                print('at least one weight was above the maximum allowed')
                wEI[wEITooBig] = p['maxAllowedWEI']
            if wIITooBig.any():
                print('at least one weight was above the maximum allowed')
                wII[wIITooBig] = p['maxAllowedWII']

            if p['useRule'][:5] == 'cross' or p['useRule'] == 'homeo':
                print(sumWeightMsgFormatter.format(movAvgUpFRExc, movAvgUpFRInh, dwEE.sum() / pA,
                                                   dwEI.sum() / pA,
                                                   dwIE.sum() / pA,
                                                   dwII.sum() / pA))
                print(meanWeightChangeMsgFormatter.format(dwEE.mean() / pA,
                                                          dwEI.mean() / pA,
                                                          dwIE.mean() / pA,
                                                          dwII.mean() / pA))
            elif p['useRule'][:7] == 'balance':
                print(sumWeightMsgFormatter.format(movAvgUpFRExc, movAvgUpFRInh,
                                                   np.nansum(dwEEMat) / pA,
                                                   np.nansum(dwIEMat) / pA,
                                                   np.nansum(dwEIMat) / pA,
                                                   np.nansum(dwIIMat) / pA))
                print(meanWeightChangeMsgFormatter.format(np.nanmean(dwEEMat) / pA,
                                                          np.nanmean(dwIEMat) / pA,
                                                          np.nanmean(dwEIMat) / pA,
                                                          np.nanmean(dwIIMat) / pA))

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
            'trialMAdwEE': self.trialMAdwEE,
            'trialMAdwEI': self.trialMAdwEI,
            'trialMAdwIE': self.trialMAdwIE,
            'trialMAdwII': self.trialMAdwII,
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


class DestexheTrainer(object):

    def __init__(self, p):
        self.p = p
        self.p['initTime'] = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # construct name from parameterSet, nUnits, propConnect, useRule, initWeightMethod, nameSuffix, initTime
        self.saveName = '_'.join((p['paramSet'], str(int(p['nUnits'])), str(p['propConnect']).replace('.', 'p'),
                                  p['useRule'], p['initWeightMethod'], p['nameSuffix'], p['initTime']))

    def set_up_network_upCrit(self, priorResults=None, recordAllVoltage=False):
        # set up network, experiment, and start recording
        DN = DestexheNetwork(self.p)
        DN.initialize_network()
        DN.initialize_units()

        if self.p['kickType'] == 'barrage':
            DN.initialize_external_input_uncorrelated()

        # create synapses!
        DN.initialize_recurrent_synapses_4bundles()

        if recordAllVoltage:
            DN.create_monitors_allVoltage()
        else:
            DN.create_monitors()

        self.DN = DN

    def run_upCrit(self):

        # run the simulation
        self.DN.run()

    def save_params(self):
        savePath = os.path.join(self.p['saveFolder'], self.saveName + '_params.pkl')
        with open(savePath, 'wb') as f:
            dill.dump(self.p, f)

    def save_results_upCrit(self):
        savePath = os.path.join(self.p['saveFolder'], self.saveName + '_results.npz')

        useDType = np.single

        spikeMonExcT = np.array(self.DN.spikeMonExc.t, dtype=useDType)
        spikeMonExcI = np.array(self.DN.spikeMonExc.i, dtype=useDType)
        spikeMonInhT = np.array(self.DN.spikeMonInh.t, dtype=useDType)
        spikeMonInhI = np.array(self.DN.spikeMonInh.i, dtype=useDType)
        stateMonExcV = np.array(self.DN.stateMonExc.v / mV, dtype=useDType)
        stateMonInhV = np.array(self.DN.stateMonInh.v / mV, dtype=useDType)

        saveDict = {
            'spikeMonExcT': spikeMonExcT,
            'spikeMonExcI': spikeMonExcI,
            'spikeMonInhT': spikeMonInhT,
            'spikeMonInhI': spikeMonInhI,
            'stateMonExcV': stateMonExcV,
            'stateMonInhV': stateMonInhV,
        }

        np.savez(savePath, **saveDict)
