"""
classes for representing networks of simulated neurons in Brian2.
classes consume a parameter dictionary,
but they also have numerous "hardcoded" aspects inside of them, such as the equations.
the classes do the following:
    create the units based on mathematical specification (given needed parameters)
    create the synapses and set their weights (external or recurrent)
    create monitors, which allow one to record spiking and state variables
    build the network out of the above Brian2 objects
    runs the network for a given duration at a given dt
    saves the results as a Results object
    saves the params in a pickle
can ALSO set up certain classic types of experiments for characterizing a network.
"""

from generate import fixed_current_series, adjacency_matrix_from_flat_inds, \
    adjacency_indices_within, adjacency_indices_between, poisson_single
from brian2 import start_scope, NeuronGroup, Synapses, SpikeMonitor, StateMonitor, Network, defaultclock, mV, ms, \
    SpikeGeneratorGroup, second, TimedArray, Hz, pA
import dill
import numpy as np
from datetime import datetime
import os


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
        elif self.p['useSecondPopExc']:
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

    def initialize_synapses(self):

        tauRiseEOverMS = self.p['tauRiseExc'] / ms
        tauRiseIOverMS = self.p['tauRiseInh'] / ms
        vTauExcOverMS = self.p['membraneCapacitanceExc'] / self.p['gLeakExc'] / ms
        vTauInhOverMS = self.p['membraneCapacitanceInh'] / self.p['gLeakInh'] / ms

        usejEE = self.p['jEE'] / self.p['nIncExc'] * vTauExcOverMS
        usejIE = self.p['jIE'] / self.p['nIncExc'] * vTauInhOverMS
        usejEI = self.p['jEI'] / self.p['nIncInh'] * vTauExcOverMS
        usejII = self.p['jII'] / self.p['nIncInh'] * vTauInhOverMS

        # from E to E
        synapsesEE = Synapses(
            model='jEE: amp',
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre='uE_post += jEE / tauRiseEOverMS',  # 'uE_post += 1 / tauRiseEOverMS'
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
        # if self.p['propConnect'] == 1:
        #     synapsesIE.connect('i!=j', p=self.p['propConnect'])
        # else:
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

    def initialize_synapses_results(self, R):

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

    def prepare_upPoisson_experiment(self, poissonLambda=0.025 * Hz, duration=30 * second, spikeUnits=100, rng=None,
                                     currentAmp=0.98):

        kickTimes = poisson_single(poissonLambda, self.p['dt'], duration, rng)
        # kickTimes = np.arange(1, duration / second, 4)

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
            on_pre='uExt_post += ' + str(currentAmp) + ' * nA'
        )
        feedforwardUpExc.connect('i==j')

        self.N.add(Uppers, feedforwardUpExc)

        self.p['duration'] = duration
        self.p['iKickRecorded'] = fixed_current_series(0, self.p['duration'], self.p['dt'])


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
