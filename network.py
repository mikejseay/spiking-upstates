"""
classes for representing networks of simulated neurons in Brian2.
classes consume a parameter dictionary,
but they also have numerous "hardcoded" aspects inside of them, such as the equations.
the classes do the following:
    create the units based on specific mathematical specification (given needed parameters)
    create the synapses and set their weights (external or recurrent)
    create monitors, which allow one to record spiking and state variables
    build the network out of the above Brian2 objects
    runs the network for a given duration at a given dt
    saves the results as a Results object
    saves the params in a pickle
can ALSO set up certain classic types of experiments for characterizing a network.
"""

from brian2 import start_scope, NeuronGroup, Synapses, SpikeMonitor, StateMonitor, Network, defaultclock, mV, ms, \
    volt, PoissonGroup, SpikeGeneratorGroup, Mohm, second, nS, TimedArray, Hz, pA
import dill
from datetime import datetime
import os
from generate import fixed_current_series, adjacency_matrix_from_flat_inds, \
    adjacency_indices_within, adjacency_indices_between, poisson_single
import numpy as np


class BaseNetwork(object):
    """ represents descriptive information needed to create network in Brian.
        creates the NeuronGroup, sets variable/randomized params of units,
        creates the Synapses, connects them, sets any variable/randomized params of synapses,
        creates Monitor objects,
        eventually saves all created objects in a Network object (Brian), which can be passed
        for various simulation goals
        SERVES AS A TEMPLATE AND IS NOT USED
    """

    def __init__(self, params):
        self.p = params
        self.p['initTime'] = datetime.now().strftime('%Y-%m-%d-%H-%M')
        saveName = self.p['simName']
        if self.p['saveWithDate']:
            saveName += '_' + self.p['initTime']
        self.saveName = saveName

    def build(self):
        # Because each network will do this differently, this will serve as a template and alwyas be overloaded.

        start_scope()

        unitModel = '''
        '''

        resetCode = '''
        '''

        threshCode = ''

        units = NeuronGroup(
            N=self.p['nUnits'],
            model=unitModel,
            method=self.p['updateMethod'],
            threshold=threshCode,
            reset=resetCode,
            refractory=self.p['refractoryPeriod'],
            dt=self.p['dt']
        )

        unitsExc = units[:self.p['nExc']]
        unitsInh = units[self.p['nExc']:]

        # set Exc/Inh parameters

        # create synapses

        synapsesExc = Synapses(
            source=unitsExc,
            target=units,
            on_pre='',
        )
        synapsesInh = Synapses(
            source=unitsInh,
            target=units,
            on_pre='',
        )
        synapsesExc.connect(p=self.p['propConnect'])
        synapsesInh.connect(p=self.p['propConnect'])

        spikeMonExc = SpikeMonitor(unitsExc[:self.p['nExcSpikemon']])
        spikeMonInh = SpikeMonitor(unitsInh[:self.p['nInhSpikemon']])

        stateMonExc = StateMonitor(unitsExc, self.p['recordStateVariables'],
                                   record=list(range(self.p['nRecordStateExc'])))
        stateMonInh = StateMonitor(unitsInh, self.p['recordStateVariables'],
                                   record=self.p['indsRecordStateInh'])

        N = Network(units, synapsesExc, synapsesInh, spikeMonExc, spikeMonInh, stateMonExc, stateMonInh)

        self.N = N
        self.spikeMonExc = spikeMonExc
        self.spikeMonInh = spikeMonInh
        self.stateMonExc = stateMonExc
        self.stateMonInh = stateMonInh

    def run(self):
        # Because each network will do this differently, this will serve as a template and alwyas be overloaded.

        # instantiate variable names needed for the unitModel (i.e. those not hardcoded or parameterized for all units)

        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def save_results(self):
        useDType = np.single

        spikeMonExcT = np.array(self.spikeMonExc.t, dtype=useDType)
        spikeMonExcI = np.array(self.spikeMonExc.i, dtype=useDType)
        spikeMonInhT = np.array(self.spikeMonInh.t, dtype=useDType)
        spikeMonInhI = np.array(self.spikeMonInh.i, dtype=useDType)
        stateMonExcV = np.array(self.stateMonExc.v / mV, dtype=useDType)
        stateMonInhV = np.array(self.stateMonInh.v / mV, dtype=useDType)

        savePath = os.path.join(self.p['saveFolder'],
                                self.saveName + '_results.npz')

        np.savez(savePath, spikeMonExcT=spikeMonExcT, spikeMonExcI=spikeMonExcI, spikeMonInhT=spikeMonInhT,
                 spikeMonInhI=spikeMonInhI, stateMonExcV=stateMonExcV, stateMonInhV=stateMonInhV)

    def save_params(self):
        savePath = os.path.join(self.p['saveFolder'],
                                self.saveName + '_params.pkl')
        with open(savePath, 'wb') as f:
            dill.dump(self.p, f)


class JercogNetwork(object):
    """ represents a network inspired by:
        Jercog, D., Roxin, A., Barthó, P., Luczak, A., Compte, A., & De La Rocha, J. (2017).
        UP-DOWN cortical dynamics reflect state transitions in a bistable network.
        ELife, 6, 1–33. https://doi.org/10.7554/eLife.22425 """

    def __init__(self, params):
        self.p = params
        self.p['initTime'] = datetime.now().strftime('%Y-%m-%d-%H-%M')
        saveName = self.p['simName']
        if self.p['saveWithDate']:
            saveName += '_' + self.p['initTime']
        self.saveName = saveName

    def initialize_network(self):
        start_scope()
        self.N = Network()

    def calculate_connectivity_stats(self):
        # get some adjacency matrix and nPre
        aEE = adjacency_matrix_from_flat_inds(self.p['nExc'], self.p['nExc'], self.preEE, self.posEE)
        aEI = adjacency_matrix_from_flat_inds(self.p['nInh'], self.p['nExc'], self.preEI, self.posEI)
        aIE = adjacency_matrix_from_flat_inds(self.p['nExc'], self.p['nInh'], self.preIE, self.posIE)
        aII = adjacency_matrix_from_flat_inds(self.p['nInh'], self.p['nInh'], self.preII, self.posII)
        nIncomingExcOntoEachExc = aEE.sum(0)
        nIncomingInhOntoEachExc = aEI.sum(0)
        nIncomingExcOntoEachInh = aIE.sum(0)
        nIncomingInhOntoEachInh = aII.sum(0)
        self.nIncomingExcOntoEachExc = nIncomingExcOntoEachExc
        self.nIncomingInhOntoEachExc = nIncomingInhOntoEachExc
        self.nIncomingExcOntoEachInh = nIncomingExcOntoEachInh
        self.nIncomingInhOntoEachInh = nIncomingInhOntoEachInh

    def initialize_units(self):
        unitModel = '''
                dv/dt = (gl * (eLeak - v) - iAdapt +
                         sE - sI + sExt +
                         kKick * iKick) / Cm +
                         noiseSigma * (Cm / gl)**-0.5 * xi: volt (unless refractory)
                diAdapt/dt = -iAdapt / tauAdapt : amp

                dsE/dt = (-sE + uE) / tauFallE : amp
                duE/dt = -uE / tauRiseE : amp
                dsI/dt = (-sI + uI) / tauFallI : amp
                duI/dt = -uI / tauRiseI : amp
                dsExt/dt = (-sExt + uExt) / tauFallE : amp
                duExt/dt = -uExt / tauRiseE : amp

                eLeak : volt
                kKick : amp
                iKick = iKickRecorded(t) : 1
                vReset : volt
                vThresh : volt
                betaAdapt : amp * second
                gl : siemens
                Cm : farad
                '''

        resetCode = '''
        v = vReset
        iAdapt += betaAdapt / tauAdapt 
        '''

        threshCode = 'v >= vThresh'

        self.p['nInh'] = int(self.p['propInh'] * self.p['nUnits'])
        self.p['nExc'] = int(self.p['nUnits'] - self.p['nInh'])

        unitsExc = NeuronGroup(
            N=self.p['nExc'],
            model=unitModel,
            method=self.p['updateMethod'],
            threshold=threshCode,
            reset=resetCode,
            refractory=self.p['refractoryPeriodExc'],
            clock=defaultclock,
        )
        unitsInh = NeuronGroup(
            N=self.p['nInh'],
            model=unitModel,
            method=self.p['updateMethod'],
            threshold=threshCode,
            reset=resetCode,
            refractory=self.p['refractoryPeriodInh'],
            clock=defaultclock,
        )

        self.p['nExcSpikemon'] = int(self.p['nExc'] * self.p['propSpikemon'])
        self.p['nInhSpikemon'] = int(self.p['nInh'] * self.p['propSpikemon'])

        unitsExc.v = self.p['eLeakExc']
        unitsExc.vReset = self.p['vResetExc']
        unitsExc.vThresh = self.p['vThreshExc']
        unitsExc.betaAdapt = self.p['betaAdaptExc']
        unitsExc.eLeak = self.p['eLeakExc']
        unitsExc.Cm = self.p['membraneCapacitanceExc']
        unitsExc.gl = self.p['gLeakExc']

        unitsInh.v = self.p['eLeakInh']
        unitsInh.vReset = self.p['vResetInh']
        unitsInh.vThresh = self.p['vThreshInh']
        unitsInh.betaAdapt = self.p['betaAdaptInh']
        unitsInh.eLeak = self.p['eLeakInh']
        unitsInh.Cm = self.p['membraneCapacitanceInh']
        unitsInh.gl = self.p['gLeakInh']

        if 'useSecondPopExc' not in self.p:
            self.p['useSecondPopExc'] = False

        if self.p['useSecondPopExc']:
            startInd = self.p['startIndSecondPopExc']
            endInd = startInd + self.p['nUnitsSecondPopExc']

            unitsExc2 = unitsExc[startInd:endInd]

            unitsExc2.v = self.p['eLeakExc2']
            unitsExc2.vReset = self.p['vResetExc2']
            unitsExc2.vThresh = self.p['vThreshExc2']
            unitsExc2.betaAdapt = self.p['betaAdaptExc2']
            unitsExc2.eLeak = self.p['eLeakExc2']
            unitsExc2.Cm = self.p['membraneCapacitanceExc2']
            unitsExc2.gl = self.p['gLeakExc2']

        self.unitsExc = unitsExc
        self.unitsInh = unitsInh
        self.N.add(unitsExc)
        self.N.add(unitsInh)

    def set_kicked_units(self, onlyKickExc=False):
        ''' note that this refers to kicks by current injection, not spikes '''

        if onlyKickExc:
            unitsExcKicked = self.unitsExc[:int(self.p['nUnits'] * self.p['propKicked'])]
            unitsExcKicked.kKick = self.p['kickAmplitudeExc']
        else:
            unitsExcKicked = self.unitsExc[:int(self.p['nExc'] * self.p['propKicked'])]
            unitsExcKicked.kKick = self.p['kickAmplitudeExc']
            unitsInhKicked = self.unitsInh[:int(self.p['nInh'] * self.p['propKicked'])]
            unitsInhKicked.kKick = self.p['kickAmplitudeInh']

    def set_paradoxical_kicked(self):
        unitsInhKicked = self.unitsInh[:int(self.p['nInh'] * self.p['propKicked'])]
        unitsInhKicked.kKick = self.p['kickAmplitudeInh']
        unitsExcKicked = self.unitsExc[:int(self.p['nExc'] * self.p['propKicked'])]
        unitsExcKicked.kKick = self.p['kickAmplitudeExc']

    def set_spiked_units(self, onlySpikeExc=True, critExc=0.784 * volt, critInh=0.67625 * volt):

        nSpikedUnits = int(self.p['nUnits'] * self.p['propKicked'])

        tauRiseE = self.p['tauRiseExc']

        multExc = critExc / (self.unitsExc.jE[0] * 100 * Mohm)
        multInh = critInh / (self.unitsInh.jE[0] * 100 * Mohm)

        indices = []
        times = []
        for kickTime in self.p['kickTimes']:
            indices.extend(list(range(nSpikedUnits)))
            times.extend([float(kickTime), ] * nSpikedUnits)

        Spikers = SpikeGeneratorGroup(nSpikedUnits, np.array(indices), np.array(times) * second)

        if onlySpikeExc:
            feedforwardUpExc = Synapses(
                source=Spikers,
                target=self.unitsExc,
                on_pre='uE_post += ' + str(multExc / tauRiseE * ms),
            )
            feedforwardUpExc.connect('i==j')
            self.N.add(Spikers, feedforwardUpExc)
        else:
            nTargetExcUnits = int(self.p['nUnits'] * self.p['propKicked'] * (1 - self.p['propInh']))
            # nTargetInhUnits = int(self.p['nUnits'] * self.p['propKicked'] * self.p['propInh'])
            feedforwardUpExc = Synapses(
                source=Spikers[:nTargetExcUnits],
                target=self.unitsExc[:nTargetExcUnits],
                on_pre='uE_post += ' + str(multExc / tauRiseE * ms),
            )
            feedforwardUpInh = Synapses(
                source=Spikers[nTargetExcUnits:],
                target=self.unitsExc[nTargetExcUnits:],
                on_pre='uE_post += ' + str(multExc / tauRiseE * ms),
            )
            feedforwardUpExc.connect('i==j')
            feedforwardUpInh.connect('i==j')
            self.N.add(Spikers, feedforwardUpExc, feedforwardUpInh)

        self.p['iKickRecorded'] = fixed_current_series(0, self.p['duration'], self.p['dt'])

    def initialize_external_input_uncorrelated(self, currentAmpExc=1, currentAmpInh=1):

        inputGroupUncorrelated = PoissonGroup(int(self.p['nPoissonUncorrInputUnits']), self.p['poissonUncorrInputRate'])
        feedforwardSynapsesUncorrExc = Synapses(inputGroupUncorrelated, self.unitsExc,
                                                on_pre='uExt_post += ' + str(currentAmpExc) + ' * pA')
        feedforwardSynapsesUncorrInh = Synapses(inputGroupUncorrelated, self.unitsInh,
                                                on_pre='uExt_post += ' + str(currentAmpInh) + ' * pA')

        feedforwardSynapsesUncorrExc.connect('i==j')
        feedforwardSynapsesUncorrInh.connect('i==j')

        self.N.add(inputGroupUncorrelated, feedforwardSynapsesUncorrExc, feedforwardSynapsesUncorrInh)

    def initialize_synapses(self):

        tauRiseEOverMS = self.p['tauRiseExc'] / ms
        tauRiseIOverMS = self.p['tauRiseInh'] / ms
        vTauExcOverMS = self.p['membraneCapacitanceExc'] / self.p['gLeakExc'] / ms
        vTauInhOverMS = self.p['membraneCapacitanceInh'] / self.p['gLeakInh'] / ms

        usejEE = self.p['jEE'] / self.p['nIncExc'] * vTauExcOverMS
        usejIE = self.p['jIE'] / self.p['nIncExc'] * vTauInhOverMS
        usejEI = self.p['jEI'] / self.p['nIncInh'] * vTauExcOverMS
        usejII = self.p['jII'] / self.p['nIncInh'] * vTauInhOverMS
        onPreStrings = ('uE_post += jEE / tauRiseEOverMS',
                        'uE_post += jIE / tauRiseEOverMS',
                        'uI_post += jEI / tauRiseIOverMS',
                        'uI_post += jII / tauRiseIOverMS',)

        # from E to E
        synapsesEE = Synapses(
            model='jEE: amp',
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre='uE_post += jEE / tauRiseEOverMS',
        )
        preInds, postInds = adjacency_indices_within(self.p['nExc'], self.p['propConnect'],
                                                     allowAutapses=self.p['allowAutapses'], rng=self.p['rng'])
        synapsesEE.connect(i=preInds, j=postInds)
        self.preEE = preInds
        self.posEE = postInds
        synapsesEE.jEE = usejEE

        # from E to I
        synapsesIE = Synapses(
            model='jIE: amp',
            source=self.unitsExc,
            target=self.unitsInh,
            on_pre='uE_post += jIE / tauRiseEOverMS',
        )
        preInds, postInds = adjacency_indices_between(self.p['nExc'], self.p['nInh'],
                                                      self.p['propConnect'], rng=self.p['rng'])
        synapsesIE.connect(i=preInds, j=postInds)
        self.preIE = preInds
        self.posIE = postInds
        synapsesIE.jIE = usejIE

        # from I to E
        synapsesEI = Synapses(
            model='jEI: amp',
            source=self.unitsInh,
            target=self.unitsExc,
            on_pre='uI_post += jEI / tauRiseIOverMS',
        )
        preInds, postInds = adjacency_indices_between(self.p['nInh'], self.p['nExc'],
                                                      self.p['propConnect'], rng=self.p['rng'])
        synapsesEI.connect(i=preInds, j=postInds)
        self.preEI = preInds
        self.posEI = postInds
        synapsesEI.jEI = usejEI

        # from I to I
        synapsesII = Synapses(
            model='jII: amp',
            source=self.unitsInh,
            target=self.unitsInh,
            on_pre='uI_post += jII / tauRiseIOverMS',
        )
        preInds, postInds = adjacency_indices_within(self.p['nInh'], self.p['propConnect'],
                                                     allowAutapses=self.p['allowAutapses'], rng=self.p['rng'])
        synapsesII.connect(i=preInds, j=postInds)
        self.preII = preInds
        self.posII = postInds
        synapsesII.jII = usejII

        synapsesEE.delay = ((self.p['rng'].random(synapsesEE.delay.shape[0]) * self.p['delayExc'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesIE.delay = ((self.p['rng'].random(synapsesIE.delay.shape[0]) * self.p['delayExc'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesEI.delay = ((self.p['rng'].random(synapsesEI.delay.shape[0]) * self.p['delayInh'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesII.delay = ((self.p['rng'].random(synapsesII.delay.shape[0]) * self.p['delayInh'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self.synapsesEE = synapsesEE
        self.synapsesIE = synapsesIE
        self.synapsesEI = synapsesEI
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesIE, synapsesEI, synapsesII)

    def initialize_synapses_from_results(self, R):

        tauRiseEOverMS = R.p['tauRiseExc'] / ms
        tauRiseIOverMS = R.p['tauRiseInh'] / ms
        vTauExcOverMS = R.p['membraneCapacitanceExc'] / R.p['gLeakExc'] / ms
        vTauInhOverMS = R.p['membraneCapacitanceInh'] / R.p['gLeakInh'] / ms

        # from E to E
        synapsesEE = Synapses(
            model='jEE: amp',
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre='uE_post += jEE / tauRiseEOverMS',
        )
        if R.p['propConnect'] == 1:
            synapsesEE.connect('i!=j', p=R.p['propConnect'])
        else:
            preInds, postInds = R.preEE, R.posEE
            synapsesEE.connect(i=preInds, j=postInds)
            self.preEE = preInds
            self.posEE = postInds
        synapsesEE.jEE = R.wEE_final * pA

        # from E to I
        synapsesIE = Synapses(
            model='jIE: amp',
            source=self.unitsExc,
            target=self.unitsInh,
            on_pre='uE_post += jIE / tauRiseEOverMS',
        )
        if R.p['propConnect'] == 1:
            synapsesIE.connect('i!=j', p=R.p['propConnect'])
        else:
            preInds, postInds = R.preIE, R.posIE
            synapsesIE.connect(i=preInds, j=postInds)
            self.preIE = preInds
            self.posIE = postInds
        synapsesIE.jIE = R.wIE_final * pA

        # from I to E
        synapsesEI = Synapses(
            model='jEI: amp',
            source=self.unitsInh,
            target=self.unitsExc,
            on_pre='uI_post += jEI / tauRiseIOverMS',
        )
        if R.p['propConnect'] == 1:
            synapsesEI.connect('i!=j', p=R.p['propConnect'])
        else:
            preInds, postInds = R.preEI, R.posEI
            synapsesEI.connect(i=preInds, j=postInds)
            self.preEI = preInds
            self.posEI = postInds
        synapsesEI.jEI = R.wEI_final * pA

        # from I to I
        synapsesII = Synapses(
            model='jII: amp',
            source=self.unitsInh,
            target=self.unitsInh,
            on_pre='uI_post += jII / tauRiseIOverMS',
        )
        if R.p['propConnect'] == 1:
            synapsesII.connect('i!=j', p=R.p['propConnect'])
        else:
            preInds, postInds = R.preII, R.posII
            synapsesII.connect(i=preInds, j=postInds)
            self.preII = preInds
            self.posII = postInds
        synapsesII.jII = R.wII_final * pA

        synapsesEE.delay = ((self.p['rng'].random(synapsesEE.delay.shape[0]) * self.p['delayExc'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesIE.delay = ((self.p['rng'].random(synapsesIE.delay.shape[0]) * self.p['delayExc'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesEI.delay = ((self.p['rng'].random(synapsesEI.delay.shape[0]) * self.p['delayInh'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesII.delay = ((self.p['rng'].random(synapsesII.delay.shape[0]) * self.p['delayInh'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self.synapsesEE = synapsesEE
        self.synapsesIE = synapsesIE
        self.synapsesEI = synapsesEI
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesIE, synapsesEI, synapsesII)

    def create_monitors(self):

        spikeMonExc = SpikeMonitor(self.unitsExc[:self.p['nExcSpikemon']])
        spikeMonInh = SpikeMonitor(self.unitsInh[:self.p['nInhSpikemon']])

        stateMonExc = StateMonitor(self.unitsExc, self.p['recordStateVariables'],
                                   record=self.p['indsRecordStateExc'],
                                   clock=defaultclock)
        stateMonInh = StateMonitor(self.unitsInh, self.p['recordStateVariables'],
                                   record=self.p['indsRecordStateInh'],
                                   clock=defaultclock)

        self.spikeMonExc = spikeMonExc
        self.spikeMonInh = spikeMonInh
        self.stateMonExc = stateMonExc
        self.stateMonInh = stateMonInh
        self.N.add(spikeMonExc, spikeMonInh, stateMonExc, stateMonInh)

    def create_monitors_allVoltage(self):

        spikeMonExc = SpikeMonitor(self.unitsExc[:self.p['nExcSpikemon']])
        spikeMonInh = SpikeMonitor(self.unitsInh[:self.p['nInhSpikemon']])

        stateMonExc = StateMonitor(self.unitsExc, self.p['recordStateVariables'],
                                   record=True,
                                   dt=self.p['stateVariableDT'])
        stateMonInh = StateMonitor(self.unitsInh, self.p['recordStateVariables'],
                                   record=True,
                                   dt=self.p['stateVariableDT'])

        self.spikeMonExc = spikeMonExc
        self.spikeMonInh = spikeMonInh
        self.stateMonExc = stateMonExc
        self.stateMonInh = stateMonInh
        self.N.add(spikeMonExc, spikeMonInh, stateMonExc, stateMonInh)

    def run(self):

        iKickRecorded = self.p['iKickRecorded']  # disabling this to test brian2genn
        noiseSigma = self.p['noiseSigma']
        tauRiseE = self.p['tauRiseExc']
        tauFallE = self.p['tauFallExc']
        tauRiseI = self.p['tauRiseInh']
        tauFallI = self.p['tauFallInh']
        tauAdapt = self.p['adaptTau']
        tauRiseEOverMS = self.p['tauRiseExc'] / ms
        tauRiseIOverMS = self.p['tauRiseInh'] / ms
        vTauExcOverMS = self.p['membraneCapacitanceExc'] / self.p['gLeakExc'] / ms
        vTauInhOverMS = self.p['membraneCapacitanceInh'] / self.p['gLeakInh'] / ms

        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def save_results_to_file(self):
        useDType = np.single

        spikeMonExcT = np.array(self.spikeMonExc.t, dtype=useDType)
        spikeMonExcI = np.array(self.spikeMonExc.i, dtype=useDType)
        spikeMonInhT = np.array(self.spikeMonInh.t, dtype=useDType)
        spikeMonInhI = np.array(self.spikeMonInh.i, dtype=useDType)
        stateMonExcV = np.array(self.stateMonExc.v / mV, dtype=useDType)
        stateMonInhV = np.array(self.stateMonInh.v / mV, dtype=useDType)

        savePath = os.path.join(self.p['saveFolder'],
                                self.saveName + '_results.npz')

        np.savez(savePath, spikeMonExcT=spikeMonExcT, spikeMonExcI=spikeMonExcI, spikeMonInhT=spikeMonInhT,
                 spikeMonInhI=spikeMonInhI, stateMonExcV=stateMonExcV, stateMonInhV=stateMonInhV)

    def save_params_to_file(self):
        savePath = os.path.join(self.p['saveFolder'],
                                self.saveName + '_params.pkl')
        with open(savePath, 'wb') as f:
            dill.dump(self.p, f)

        # print('\a')

        # duration = 1000  # milliseconds
        # freq = 440  # Hz
        # winsound.Beep(freq, duration)

        # win32api.MessageBox(0, 'hello', 'title')

    def determine_fan_in(self, minUnits=21, maxUnits=40, unitSpacing=1, timeSpacing=250 * ms):

        # set the unit params that must be in the name space
        noiseSigma = 0 * self.p['noiseSigma']  # 0 for this experiment!!
        tauRiseE = self.p['tauRiseExc']
        tauFallE = self.p['tauFallExc']
        tauRiseI = self.p['tauRiseInh']
        tauFallI = self.p['tauFallInh']
        tauAdapt = self.p['adaptTau']

        # create the spike timings for the spike generator
        indices = []
        times = []
        dummyInd = 0
        useRange = range(minUnits, maxUnits + 1, unitSpacing)
        for unitInd in useRange:
            dummyInd += 1
            indices.extend(list(range(unitInd)))
            times.extend([float(timeSpacing) * dummyInd, ] * (unitInd))

        # create the spike generator and the synapses from it to the 2 units, connect them
        Fanners = SpikeGeneratorGroup(maxUnits, np.array(indices), np.array(times) * second)
        feedforwardFanExc = Synapses(
            source=Fanners,
            target=self.unitsExc,
            on_pre='uE_post += ' + str(1 / tauRiseE * ms),
        )
        feedforwardFanInh = Synapses(
            source=Fanners,
            target=self.unitsInh,
            on_pre='uE_post += ' + str(1 / tauRiseE * ms),
        )
        feedforwardFanExc.connect(p=1)
        feedforwardFanInh.connect(p=1)

        # add them to the network, set the run duration, create a bogus kick current
        self.N.add(Fanners, feedforwardFanExc, feedforwardFanInh)
        self.p['duration'] = (np.array(times).max() * second + timeSpacing)
        iKickRecorded = fixed_current_series(0, self.p['duration'], self.p['dt'])

        # set the weights
        self.unitsExc.jE = 7840 / 4 * pA  # jEE
        self.unitsExc.jI = 7840 / 4 * pA  # jEI
        self.unitsInh.jE = 7840 / 4 * pA  # jIE
        self.unitsInh.jI = 7840 / 4 * pA  # jII

        # all that's left is to monitor and run
        self.create_monitors()
        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def prepare_upCrit_experiment2(self, minUnits=170, maxUnits=180, unitSpacing=5, timeSpacing=3000 * ms,
                                   startTime=100 * ms, currentAmp=0.98):

        tauRiseE = self.p['tauRiseExc']

        # multExc = critExc / self.unitsExc.jE[0]
        # multInh = critInh / self.unitsInh.jE[0]

        indices = []
        times = []
        dummyInd = -1
        if minUnits <= maxUnits:
            useRange = range(minUnits, maxUnits + 1, unitSpacing)
            nUnits = maxUnits
        else:
            useRange = range(minUnits, maxUnits - 1, unitSpacing)
            nUnits = minUnits
        for unitInd in useRange:
            dummyInd += 1
            indices.extend(list(range(unitInd)))
            times.extend([float(startTime) + float(timeSpacing) * dummyInd, ] * (unitInd))

        Uppers = SpikeGeneratorGroup(nUnits, np.array(indices), np.array(times) * second)

        feedforwardUpExc = Synapses(
            source=Uppers,
            target=self.unitsExc,
            on_pre='uExt_post += ' + str(currentAmp) + ' * nA'
            # on_pre='uE_post += ' + str(currentAmp / nA) + ' * nA'
            #  + str(critExc / (100 * Mohm) / tauRiseE * ms),
        )
        feedforwardUpExc.connect('i==j')

        self.N.add(Uppers, feedforwardUpExc)

        self.p['duration'] = (np.array(times).max() * second + timeSpacing)
        self.p['iKickRecorded'] = fixed_current_series(0, self.p['duration'], self.p['dt'])

    def prepare_upPoisson_experiment(self, poissonLambda=0.025 * Hz, duration=30 * second, spikeUnits=100, rng=None):

        kickTimes = poisson_single(poissonLambda, self.p['dt'], duration, rng)
        # kickTimes = np.arange(1, float(duration), 3)  # just to check

        indices = []
        times = []
        for kickTime in kickTimes:
            indices.extend(list(range(spikeUnits)))
            times.extend([float(kickTime), ] * spikeUnits)

        self.p['upPoissonTimes'] = kickTimes

        Uppers = SpikeGeneratorGroup(spikeUnits, np.array(indices), np.array(times) * second)

        feedforwardUpExc = Synapses(
            source=Uppers,
            target=self.unitsExc,
            on_pre='uExt_post += 0.98 * nA'
            # on_pre='uE_post += ' + str(currentAmp / nA) + ' * nA'
            #  + str(critExc / (100 * Mohm) / tauRiseE * ms),
        )
        feedforwardUpExc.connect('i==j')

        self.N.add(Uppers, feedforwardUpExc)

        self.p['duration'] = duration
        self.p['iKickRecorded'] = fixed_current_series(0, self.p['duration'], self.p['dt'])


class DestexheNetwork(object):
    """ represents a network inspired by:
        Volo MD, Romagnoni A, Capone C, Destexhe A.
        Biologically Realistic Mean-Field Models of Conductance-Based Networks of Spiking Neurons with Adaptation.
        Neural Comput. 2019 Apr;31(4):653-680. doi: 10.1162/neco_a_01173. Epub 2019 Feb 14. PMID: 30764741. """

    def __init__(self, params):
        self.p = params
        self.p['initTime'] = datetime.now().strftime('%Y-%m-%d-%H-%M')
        saveName = self.p['simName']
        if self.p['saveWithDate']:
            saveName += '_' + self.p['initTime']
        self.saveName = saveName

    def initialize_network(self):
        start_scope()
        self.N = Network()

    def initialize_units(self):
        unitModel = '''
            dv/dt = (gl * (El - v) + gl * delta * exp((v - vThresh) / delta) - w +
                     gext * (Ee - v) + ge * (Ee - v) +  gi * (Ei - v) + kKick * iKick) / Cm: volt (unless refractory)
            dw/dt = (a * (v - El) - w) / tau_w : amp
            dge/dt = -ge / tau_e : siemens
            dgi/dt = -gi / tau_i : siemens
            dgext/dt = -gext / tau_e : siemens
            kKick : amp
            iKick = iKickRecorded(t) : 1
            El : volt
            delta: volt
            a : siemens
            b : amp
            vThresh : volt
            '''

        resetCode = '''
        v = El
        w += b
        '''

        threshCode = 'v > vThresh + 5 * delta'
        self.p['vThreshExc'] = self.p['vThresh'] + 5 * self.p['deltaVExc']
        self.p['vThreshInh'] = self.p['vThresh'] + 5 * self.p['deltaVInh']

        # threshCode = 'v > vThresh'
        # self.p['vThreshExc'] = self.p['vThresh']
        # self.p['vThreshInh'] = self.p['vThresh']

        units = NeuronGroup(
            N=self.p['nUnits'],
            model=unitModel,
            method=self.p['updateMethod'],
            threshold=threshCode,
            reset=resetCode,
            refractory=self.p['refractoryPeriod'],
            clock=defaultclock,
        )

        self.p['nInh'] = int(self.p['propInh'] * self.p['nUnits'])
        self.p['nExc'] = int(self.p['nUnits'] - self.p['nInh'])
        self.p['nExcSpikemon'] = int(self.p['nExc'] * self.p['propSpikemon'])
        self.p['nInhSpikemon'] = int(self.p['nInh'] * self.p['propSpikemon'])

        unitsExc = units[:self.p['nExc']]
        unitsInh = units[self.p['nExc']:]

        unitsExc.v = self.p['eLeakExc']
        unitsExc.El = self.p['eLeakExc']
        unitsExc.delta = self.p['deltaVExc']
        unitsExc.a = self.p['aExc']
        unitsExc.b = self.p['bExc']
        unitsExc.vThresh = self.p['vThresh']

        unitsInh.v = self.p['eLeakInh']
        unitsInh.El = self.p['eLeakInh']
        unitsInh.delta = self.p['deltaVInh']
        unitsInh.a = self.p['aInh']
        unitsInh.b = self.p['bInh']
        unitsInh.vThresh = self.p['vThresh']

        self.units = units
        self.unitsExc = unitsExc
        self.unitsInh = unitsInh
        self.N.add(units)

    def initialize_recurrent_synapses_4bundles(self):

        nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExc'] * self.p['propConnect'])
        nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInh'] * self.p['propConnect'])

        useQExc = self.p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
        useQInh = self.p['qInh'] / nRecurrentInhibitorySynapsesPerUnit

        print(self.p['qExc'], nRecurrentExcitatorySynapsesPerUnit, useQExc)
        print(self.p['qInh'], nRecurrentInhibitorySynapsesPerUnit, useQInh)

        # from E to E
        synapsesEE = Synapses(
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre='ge_post += ' + str(useQExc / nS) + ' * nS',
        )
        if self.p['propConnect'] == 1:
            synapsesEE.connect('i!=j', p=self.p['propConnect'])
        else:
            preInds, postInds = adjacency_indices_within(self.p['nExc'], self.p['propConnect'],
                                                         allowAutapses=False, rng=self.p['rng'])
            synapsesEE.connect(i=preInds, j=postInds)

        # from E to I
        synapsesIE = Synapses(
            source=self.unitsExc,
            target=self.unitsInh,
            on_pre='ge_post += ' + str(useQExc / nS) + ' * nS',
        )
        if self.p['propConnect'] == 1:
            synapsesIE.connect('i!=j', p=self.p['propConnect'])
        else:
            preInds, postInds = adjacency_indices_between(self.p['nExc'], self.p['nInh'],
                                                          self.p['propConnect'], rng=self.p['rng'])
            synapsesIE.connect(i=preInds, j=postInds)

        # from I to E
        synapsesEI = Synapses(
            source=self.unitsInh,
            target=self.unitsExc,
            on_pre='gi_post += ' + str(useQInh / nS) + ' * nS',
        )
        if self.p['propConnect'] == 1:
            synapsesEI.connect('i!=j', p=self.p['propConnect'])
        else:
            preInds, postInds = adjacency_indices_between(self.p['nInh'], self.p['nExc'],
                                                          self.p['propConnect'], rng=self.p['rng'])
            synapsesEI.connect(i=preInds, j=postInds)

        # from I to I
        synapsesII = Synapses(
            source=self.unitsInh,
            target=self.unitsInh,
            on_pre='gi_post += ' + str(useQInh / nS) + ' * nS',
        )
        if self.p['propConnect'] == 1:
            synapsesII.connect('i!=j', p=self.p['propConnect'])
        else:
            preInds, postInds = adjacency_indices_within(self.p['nInh'], self.p['propConnect'],
                                                         allowAutapses=False, rng=self.p['rng'])
            synapsesII.connect(i=preInds, j=postInds)

        self.synapsesEE = synapsesEE
        self.synapsesIE = synapsesIE
        self.synapsesEI = synapsesEI
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesIE, synapsesEI, synapsesII)

    def initialize_external_input_uncorrelated(self):

        inputGroupUncorrelated = PoissonGroup(int(self.p['nPoissonUncorrInputUnits']), self.p['poissonUncorrInputRate'])
        feedforwardSynapsesUncorr = Synapses(inputGroupUncorrelated, self.units,
                                             on_pre='ge_post += ' + str(self.p['qExcFeedforwardUncorr'] / nS) + ' * nS')

        feedforwardSynapsesUncorr.connect('i==j')

        self.N.add(inputGroupUncorrelated, feedforwardSynapsesUncorr)

    def initialize_external_input_correlated(self, targetExc=False, monitorProcesses=False):

        inputGroupCorrelated = PoissonGroup(int(self.p['nPoissonCorrInputUnits']), self.p['poissonCorrInputRate'])

        if targetExc:
            feedforwardSynapsesCorr = Synapses(inputGroupCorrelated, self.unitsExc,
                                               on_pre='ge_post += ' + str(self.p['qExcFeedforwardCorr'] / nS) + ' * nS')
        else:
            feedforwardSynapsesCorr = Synapses(inputGroupCorrelated, self.units,
                                               on_pre='ge_post += ' + str(self.p['qExcFeedforwardCorr'] / nS) + ' * nS')

        feedforwardSynapsesCorr.connect(p=self.p['propConnectFeedforwardProjectionCorr'])

        # self.inputGroupCorrelated = inputGroupCorrelated
        if monitorProcesses:
            self.monitorInpCorr = True
            spikeMonInpCorr = SpikeMonitor(inputGroupCorrelated)
            self.spikeMonInpCorr = spikeMonInpCorr
            self.N.add(inputGroupCorrelated, feedforwardSynapsesCorr, spikeMonInpCorr)
        else:
            self.N.add(inputGroupCorrelated, feedforwardSynapsesCorr)

    def set_spiked_units(self, onlySpikeExc=True):

        nSpikedUnits = int(self.p['nUnits'] * self.p['propKicked'])

        indices = []
        times = []
        for kickTime in self.p['kickTimes']:
            indices.extend(list(range(nSpikedUnits)))
            times.extend([float(kickTime), ] * nSpikedUnits)

        Spikers = SpikeGeneratorGroup(nSpikedUnits, np.array(indices), np.array(times) * second)

        if onlySpikeExc:
            feedforwardUpExc = Synapses(
                source=Spikers,
                target=self.unitsExc,
                on_pre='ge_post += 18.5 * nS',
            )
            feedforwardUpExc.connect('i==j')
            self.N.add(Spikers, feedforwardUpExc)
        else:
            pass

    def set_paradoxical_kicked(self):
        unitsInhKicked = self.unitsInh[:int(self.p['nInh'] * self.p['propKicked'])]
        unitsInhKicked.kKick = self.p['kickAmplitudeInh']

    def create_monitors(self):

        spikeMonExc = SpikeMonitor(self.unitsExc[:self.p['nExcSpikemon']])
        spikeMonInh = SpikeMonitor(self.unitsInh[:self.p['nInhSpikemon']])

        stateMonExc = StateMonitor(self.unitsExc, self.p['recordStateVariables'],
                                   record=self.p['indsRecordStateExc'],
                                   clock=defaultclock)
        stateMonInh = StateMonitor(self.unitsInh, self.p['recordStateVariables'],
                                   record=self.p['indsRecordStateInh'],
                                   clock=defaultclock)

        self.spikeMonExc = spikeMonExc
        self.spikeMonInh = spikeMonInh
        self.stateMonExc = stateMonExc
        self.stateMonInh = stateMonInh
        self.N.add(spikeMonExc, spikeMonInh, stateMonExc, stateMonInh)

    def create_monitors_allVoltage(self):

        spikeMonExc = SpikeMonitor(self.unitsExc[:self.p['nExcSpikemon']])
        spikeMonInh = SpikeMonitor(self.unitsInh[:self.p['nInhSpikemon']])

        stateMonExc = StateMonitor(self.unitsExc, self.p['recordStateVariables'],
                                   record=True,
                                   dt=self.p['stateVariableDT'])
        stateMonInh = StateMonitor(self.unitsInh, self.p['recordStateVariables'],
                                   record=True,
                                   dt=self.p['stateVariableDT'])

        self.spikeMonExc = spikeMonExc
        self.spikeMonInh = spikeMonInh
        self.stateMonExc = stateMonExc
        self.stateMonInh = stateMonInh
        self.N.add(spikeMonExc, spikeMonInh, stateMonExc, stateMonInh)

    def run(self):

        # vThresh = self.p['vThresh']
        iKickRecorded = self.p['iKickRecorded']
        Cm = self.p['membraneCapacitance']
        gl = self.p['gLeak']
        tau_w = self.p['adaptTau']
        Ee = self.p['eExcSyn']
        Ei = self.p['eInhSyn']
        tau_e = self.p['tauSynExc']
        tau_i = self.p['tauSynInh']
        Qe = self.p['qExc']
        Qi = self.p['qInh']

        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def save_results(self):
        useDType = np.float32

        savePath = os.path.join(self.p['saveFolder'],
                                self.saveName + '_results.npz')

        saveDict = dict()
        saveDict['spikeMonExcT'] = np.array(self.spikeMonExc.t, dtype=useDType)
        saveDict['spikeMonExcI'] = np.array(self.spikeMonExc.i, dtype=useDType)
        saveDict['spikeMonInhT'] = np.array(self.spikeMonInh.t, dtype=useDType)
        saveDict['spikeMonInhI'] = np.array(self.spikeMonInh.i, dtype=useDType)
        saveDict['stateMonExcV'] = np.array(self.stateMonExc.v / mV, dtype=useDType)
        saveDict['stateMonInhV'] = np.array(self.stateMonInh.v / mV, dtype=useDType)

        if hasattr(self, 'monitorInpCorr'):
            if self.monitorInpCorr == 'inherit':
                saveDict['spikeMonInpCorrT'] = self.spikeMonInpCorrT
                saveDict['spikeMonInpCorrI'] = self.spikeMonInpCorrI
            else:
                saveDict['spikeMonInpCorrT'] = np.array(self.spikeMonInpCorr.t, dtype=useDType)
                saveDict['spikeMonInpCorrI'] = np.array(self.spikeMonInpCorr.i, dtype=useDType)

        # these are big arrays so we'll be careful about how we save them
        if hasattr(self, 'weightsDistributed'):
            saveUInt16 = ['preInds_EE', 'preInds_IE', 'preInds_EI', 'preInds_II',
                          'postInds_EE', 'postInds_IE', 'postInds_EI', 'postInds_II', ]
            saveFloat32 = ['weights_EE', 'weights_IE', 'weights_EI', 'weights_II', ]
            for sA in saveUInt16:
                saveDict[sA] = getattr(self, sA).astype(np.uint16)
            for sA in saveFloat32:
                saveDict[sA] = getattr(self, sA).astype(np.float32)

        np.savez(savePath, **saveDict)

    def save_params(self):
        savePath = os.path.join(self.p['saveFolder'],
                                self.saveName + '_params.pkl')
        with open(savePath, 'wb') as f:
            dill.dump(self.p, f)

        # duration = 1000  # milliseconds
        # freq = 440  # Hz
        # winsound.Beep(freq, duration)

    def determine_fan_in(self, minUnits=21, maxUnits=40, unitSpacing=1, timeSpacing=250 * ms):

        nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExcFan'] * self.p['propConnect'])
        nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInhFan'] * self.p['propConnect'])

        useQExc = self.p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
        useQInh = self.p['qInh'] / nRecurrentInhibitorySynapsesPerUnit

        # set the unit params that must be in the name space
        Cm = self.p['membraneCapacitance']
        gl = self.p['gLeak']
        tau_w = self.p['adaptTau']
        Ee = self.p['eExcSyn']
        Ei = self.p['eInhSyn']
        tau_e = self.p['tauSynExc']
        tau_i = self.p['tauSynInh']
        Qe = self.p['qExc']
        Qi = self.p['qInh']

        # create the spike timings for the spike generator
        indices = []
        times = []
        dummyInd = 0
        useRange = range(minUnits, maxUnits + 1, unitSpacing)
        for unitInd in useRange:
            dummyInd += 1
            indices.extend(list(range(unitInd)))
            times.extend([float(timeSpacing) * dummyInd, ] * (unitInd))

        # create the spike generator and the synapses from it to the 2 units, connect them
        Fanners = SpikeGeneratorGroup(maxUnits, np.array(indices), np.array(times) * second)
        feedforwardFanExc = Synapses(
            source=Fanners,
            target=self.unitsExc,
            on_pre='ge_post += ' + str(useQExc / nS) + ' * nS',
        )
        feedforwardFanInh = Synapses(
            source=Fanners,
            target=self.unitsInh,
            on_pre='ge_post += ' + str(useQExc / nS) + ' * nS',
        )
        feedforwardFanExc.connect(p=1)
        feedforwardFanInh.connect(p=1)

        # add them to the network, set the run duration, create a bogus kick current
        self.N.add(Fanners, feedforwardFanExc, feedforwardFanInh)
        self.p['duration'] = (np.array(times).max() * second + timeSpacing)

        # this must be defined...
        # if this breaks, comment this out+
        iExtRecorded = fixed_current_series(1, self.p['duration'], self.p['dt'])

        # all that's left is to monitor and run
        self.create_monitors()
        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def prepare_upCrit_experiment(self, minUnits=170, maxUnits=180, unitSpacing=5, timeSpacing=3000 * ms,
                                  startTime=100 * ms):

        indices = []
        times = []
        dummyInd = -1
        useRange = range(minUnits, maxUnits + 1, unitSpacing)
        for unitInd in useRange:
            dummyInd += 1
            indices.extend(list(range(unitInd)))
            times.extend([float(startTime) + float(timeSpacing) * dummyInd, ] * (unitInd))

        Uppers = SpikeGeneratorGroup(maxUnits, np.array(indices), np.array(times) * second)

        feedforwardUpExc = Synapses(
            source=Uppers,
            target=self.unitsExc,
            on_pre='gext_post += 18.5 * nS',
        )
        feedforwardUpExc.connect('i==j')

        self.N.add(Uppers, feedforwardUpExc)

        self.p['duration'] = (np.array(times).max() * second + timeSpacing)


# THE BELOW ARE STRIPPED-DOWN VERSIONS OF THE ORIGINAL NETWORKS THAT ARE DESIGNED FOR TESTING EPHYS CHARACTERISTICS
# I.E. FIRING RATE / PATTERN GIVEN A SQUARE-WAVE CURRENT INJECTION

class JercogEphysNetwork(object):

    def __init__(self, params):
        self.p = params
        self.p['initTime'] = datetime.now().strftime('%Y-%m-%d-%H-%M')
        saveName = self.p['simName']
        if self.p['saveWithDate']:
            saveName += '_' + self.p['initTime']
        self.saveName = saveName

    def initialize_network(self):
        start_scope()
        self.N = Network()

    def initialize_units(self):
        unitModel = '''
        dv/dt = (gl * (eLeak - v) - iAdapt + iExt * iExtShape) / Cm: volt (unless refractory)
        diAdapt/dt = -iAdapt / tauAdapt : amp

        eLeak : volt
        vReset : volt
        vThresh : volt
        betaAdapt : amp * second
        iExt : amp
        iExtShape = iExtShapeRecorded(t) : 1
        gl : siemens
        Cm : farad
        '''

        resetCode = '''
        v = vReset
        iAdapt += betaAdapt / tauAdapt 
        '''

        threshCode = 'v >= vThresh'

        numIValues = len(self.p['iExtRange'])

        unitsExc = NeuronGroup(
            N=numIValues,
            model=unitModel,
            method=self.p['updateMethod'],
            threshold=threshCode,
            reset=resetCode,
            refractory=self.p['refractoryPeriodExc'],
            clock=defaultclock,
        )

        unitsInh = NeuronGroup(
            N=numIValues,
            model=unitModel,
            method=self.p['updateMethod'],
            threshold=threshCode,
            reset=resetCode,
            refractory=self.p['refractoryPeriodInh'],
            clock=defaultclock,
        )

        self.p['nInh'] = int(self.p['propInh'] * self.p['nUnits'])
        self.p['nExc'] = self.p['nUnits'] - self.p['nInh']
        self.p['nExcSpikemon'] = int(self.p['nExc'] * self.p['propSpikemon'])
        self.p['nInhSpikemon'] = int(self.p['nInh'] * self.p['propSpikemon'])

        unitsExc.v = self.p['eLeakExc']
        unitsExc.vReset = self.p['vResetExc']
        unitsExc.vThresh = self.p['vThreshExc']
        unitsExc.betaAdapt = self.p['betaAdaptExc']
        unitsExc.eLeak = self.p['eLeakExc']
        unitsExc.Cm = self.p['membraneCapacitanceExc']
        unitsExc.gl = self.p['gLeakExc']
        unitsExc.iExt = self.p['iExtRange']

        unitsInh.v = self.p['eLeakInh']
        unitsInh.vReset = self.p['vResetInh']
        unitsInh.vThresh = self.p['vThreshInh']
        unitsInh.betaAdapt = self.p['betaAdaptInh']
        unitsInh.eLeak = self.p['eLeakInh']
        unitsInh.Cm = self.p['membraneCapacitanceInh']
        unitsInh.gl = self.p['gLeakInh']
        unitsInh.iExt = self.p['iExtRange']

        self.unitsExc = unitsExc
        self.unitsInh = unitsInh
        self.N.add(unitsExc, unitsInh)

        if self.p['useSecondPopExc']:
            unitsExc2 = NeuronGroup(
                N=numIValues,
                model=unitModel,
                method=self.p['updateMethod'],
                threshold=threshCode,
                reset=resetCode,
                refractory=self.p['refractoryPeriodExc2'],
                clock=defaultclock,
            )
            unitsExc2.v = self.p['eLeakExc2']
            unitsExc2.vReset = self.p['vResetExc2']
            unitsExc2.vThresh = self.p['vThreshExc2']
            unitsExc2.betaAdapt = self.p['betaAdaptExc2']
            unitsExc2.eLeak = self.p['eLeakExc2']
            unitsExc2.Cm = self.p['membraneCapacitanceExc2']
            unitsExc2.gl = self.p['gLeakExc2']
            unitsExc2.iExt = self.p['iExtRange']
            self.unitsExc2 = unitsExc2
            self.N.add(unitsExc2)

    def create_monitors(self):
        spikeMonExc = SpikeMonitor(self.unitsExc)
        spikeMonInh = SpikeMonitor(self.unitsInh)

        # record voltage for all units
        stateMonExc = StateMonitor(self.unitsExc, self.p['recordStateVariables'], record=True)
        stateMonInh = StateMonitor(self.unitsInh, self.p['recordStateVariables'], record=True)

        self.spikeMonExc = spikeMonExc
        self.stateMonExc = stateMonExc
        self.spikeMonInh = spikeMonInh
        self.stateMonInh = stateMonInh
        self.N.add(spikeMonExc, spikeMonInh, stateMonExc, stateMonInh)

        if self.p['useSecondPopExc']:
            spikeMonExc2 = SpikeMonitor(self.unitsExc2)
            stateMonExc2 = StateMonitor(self.unitsExc2, self.p['recordStateVariables'], record=True)
            self.spikeMonExc2 = spikeMonExc2
            self.stateMonExc2 = stateMonExc2
            self.N.add(spikeMonExc2, stateMonExc2)

    def build_classic(self):
        self.initialize_network()
        self.initialize_units()
        self.create_monitors()

    def run(self):
        self.p['duration'] = self.p['baselineDur'] + self.p['iDur'] + self.p['afterDur']

        tauAdapt = self.p['adaptTau']
        iExtShapeArray = np.zeros((int(self.p['duration'] / self.p['dt']),))
        startCurrentIndex = int(self.p['baselineDur'] / self.p['dt'])
        endCurrentIndex = int((self.p['baselineDur'] + self.p['iDur']) / self.p['dt'])
        iExtShapeArray[startCurrentIndex:endCurrentIndex] = 1
        iExtShapeRecorded = TimedArray(iExtShapeArray, dt=self.p['dt'])

        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def save_results(self):
        useDType = np.single

        savePath = os.path.join(self.p['saveFolder'],
                                self.saveName + '_results.npz')

        spikeMonExcT = np.array(self.spikeMonExc.t, dtype=useDType)
        spikeMonExcI = np.array(self.spikeMonExc.i, dtype=useDType)
        spikeMonExcC = np.array(self.spikeMonExc.count, dtype=useDType)
        stateMonExcV = np.array(self.stateMonExc.v / mV, dtype=useDType)
        spikeTrainsExc = np.array(self.spikeMonExc.spike_trains(), dtype=object)

        spikeMonInhT = np.array(self.spikeMonInh.t, dtype=useDType)
        spikeMonInhI = np.array(self.spikeMonInh.i, dtype=useDType)
        spikeMonInhC = np.array(self.spikeMonInh.count, dtype=useDType)
        stateMonInhV = np.array(self.stateMonInh.v / mV, dtype=useDType)
        spikeTrainsInh = np.array(self.spikeMonInh.spike_trains(), dtype=object)

        saveDict = {
            'spikeMonExcT': spikeMonExcT,
            'spikeMonExcI': spikeMonExcI,
            'spikeMonExcC': spikeMonExcC,
            'stateMonExcV': stateMonExcV,
            'spikeTrainsExc': spikeTrainsExc,

            'spikeMonInhT': spikeMonInhT,
            'spikeMonInhI': spikeMonInhI,
            'spikeMonInhC': spikeMonInhC,
            'stateMonInhV': stateMonInhV,
            'spikeTrainsInh': spikeTrainsInh,
        }

        if self.p['useSecondPopExc']:
            spikeMonExc2T = np.array(self.spikeMonExc2.t, dtype=useDType)
            spikeMonExc2I = np.array(self.spikeMonExc2.i, dtype=useDType)
            spikeMonExc2C = np.array(self.spikeMonExc2.count, dtype=useDType)
            stateMonExc2V = np.array(self.stateMonExc2.v / mV, dtype=useDType)
            spikeTrainsExc2 = np.array(self.spikeMonExc2.spike_trains(), dtype=object)
            saveDictAdd = {
                'spikeMonExc2T': spikeMonExc2T,
                'spikeMonExc2I': spikeMonExc2I,
                'spikeMonExc2C': spikeMonExc2C,
                'stateMonExc2V': stateMonExc2V,
                'spikeTrainsExc2': spikeTrainsExc2,
            }
            saveDict.update(saveDictAdd)

        np.savez(savePath, **saveDict)

    def save_params(self):
        savePath = os.path.join(self.p['saveFolder'],
                                self.saveName + '_params.pkl')
        with open(savePath, 'wb') as f:
            dill.dump(self.p, f)


class DestexheEphysNetwork(object):

    def __init__(self, params):
        self.p = params
        self.p['initTime'] = datetime.now().strftime('%Y-%m-%d-%H-%M')
        saveName = self.p['simName']
        if self.p['saveWithDate']:
            saveName += '_' + self.p['initTime']
        self.saveName = saveName

    def initialize_network(self):
        start_scope()
        self.N = Network()

    def initialize_units(self):
        unitModel = '''
        dv/dt = (gl * (El - v) + gl * delta * exp((v - vThresh) / delta) - w + iExt +
                 ge * (Ee - v) + gi * (Ei - v)) / Cm: volt (unless refractory)
        dw/dt = (a * (v - El) - w) / tau_w : amp
        dge/dt = -ge / tau_e : siemens
        dgi/dt = -gi / tau_i : siemens
        El : volt
        delta: volt
        a : siemens
        b : amp
        vThresh : volt
        Cm : farad
        gl : siemens
        vReset : volt
        refractoryPeriod : second
        iExt : amp
        '''

        resetCode = '''
        v = vReset
        w += b
        '''

        threshCode = 'v > vThresh + 5 * delta'
        self.p['vTrueThreshExc'] = self.p['vThreshExc'] + 5 * self.p['deltaVExc']
        self.p['vTrueThreshInh'] = self.p['vThreshInh'] + 5 * self.p['deltaVInh']

        numIValues = len(self.p['iExtRange'])

        units = NeuronGroup(
            N=numIValues * 2,
            model=unitModel,
            method=self.p['updateMethod'],
            threshold=threshCode,
            reset=resetCode,
            refractory='refractoryPeriod',
            dt=self.p['dt']
        )

        self.p['nInh'] = int(self.p['propInh'] * self.p['nUnits'])
        self.p['nExc'] = int(self.p['nUnits'] - self.p['nInh'])
        self.p['nExcSpikemon'] = int(self.p['nExc'] * self.p['propSpikemon'])
        self.p['nInhSpikemon'] = int(self.p['nInh'] * self.p['propSpikemon'])

        unitsExc = units[:numIValues]
        unitsInh = units[numIValues:]

        unitsExc.v = self.p['eLeakExc']
        unitsExc.El = self.p['eLeakExc']
        unitsExc.delta = self.p['deltaVExc']
        unitsExc.a = self.p['aExc']
        unitsExc.b = self.p['bExc']
        unitsExc.vThresh = self.p['vThreshExc']
        unitsExc.Cm = self.p['membraneCapacitanceExc']
        unitsExc.gl = self.p['gLeakExc']
        unitsExc.vReset = self.p['vResetExc']
        unitsExc.refractoryPeriod = self.p['refractoryPeriodExc']
        unitsExc.iExt = self.p['iExtRange']

        unitsInh.v = self.p['eLeakInh']
        unitsInh.El = self.p['eLeakInh']
        unitsInh.delta = self.p['deltaVInh']
        unitsInh.a = self.p['aInh']
        unitsInh.b = self.p['bInh']
        unitsInh.vThresh = self.p['vThreshInh']
        unitsInh.Cm = self.p['membraneCapacitanceInh']
        unitsInh.gl = self.p['gLeakInh']
        unitsInh.vReset = self.p['vResetInh']
        unitsInh.refractoryPeriod = self.p['refractoryPeriodInh']
        unitsInh.iExt = self.p['iExtRange']

        self.units = units
        self.unitsExc = unitsExc
        self.unitsInh = unitsInh
        self.N.add(units)

    def create_monitors(self):
        spikeMonExc = SpikeMonitor(self.unitsExc)
        spikeMonInh = SpikeMonitor(self.unitsInh)

        # record voltage for all units
        stateMonExc = StateMonitor(self.unitsExc, self.p['recordStateVariables'], record=True)
        stateMonInh = StateMonitor(self.unitsInh, self.p['recordStateVariables'], record=True)

        self.spikeMonExc = spikeMonExc
        self.spikeMonInh = spikeMonInh
        self.stateMonExc = stateMonExc
        self.stateMonInh = stateMonInh
        self.N.add(spikeMonExc, spikeMonInh, stateMonExc, stateMonInh)

    def build_classic(self):
        self.initialize_network()
        self.initialize_units()
        self.create_monitors()

    def run(self):
        tau_w = self.p['adaptTau']
        Ee = self.p['eExcSyn']
        Ei = self.p['eInhSyn']
        tau_e = self.p['tauSynExc']
        tau_i = self.p['tauSynInh']
        Qe = self.p['qExc']
        Qi = self.p['qInh']

        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def save_results(self):
        useDType = np.single

        spikeMonExcT = np.array(self.spikeMonExc.t, dtype=useDType)
        spikeMonExcI = np.array(self.spikeMonExc.i, dtype=useDType)
        spikeMonExcC = np.array(self.spikeMonExc.count, dtype=useDType)
        spikeMonInhT = np.array(self.spikeMonInh.t, dtype=useDType)
        spikeMonInhI = np.array(self.spikeMonInh.i, dtype=useDType)
        spikeMonInhC = np.array(self.spikeMonInh.count, dtype=useDType)
        stateMonExcV = np.array(self.stateMonExc.v / mV, dtype=useDType)
        stateMonInhV = np.array(self.stateMonInh.v / mV, dtype=useDType)
        spikeTrainsExc = np.array(self.spikeMonExc.spike_trains(), dtype=object)
        spikeTrainsInh = np.array(self.spikeMonInh.spike_trains(), dtype=object)

        savePath = os.path.join(self.p['saveFolder'],
                                self.saveName + '_results.npz')

        np.savez(savePath, spikeMonExcT=spikeMonExcT, spikeMonExcI=spikeMonExcI, spikeMonInhT=spikeMonInhT,
                 spikeMonInhI=spikeMonInhI, stateMonExcV=stateMonExcV, stateMonInhV=stateMonInhV,
                 spikeMonExcC=spikeMonExcC, spikeMonInhC=spikeMonInhC,
                 spikeTrainsExc=spikeTrainsExc, spikeTrainsInh=spikeTrainsInh)

    def save_params(self):
        savePath = os.path.join(self.p['saveFolder'],
                                self.saveName + '_params.pkl')
        with open(savePath, 'wb') as f:
            dill.dump(self.p, f)
