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

from brian2 import start_scope, NeuronGroup, Synapses, SpikeMonitor, StateMonitor, Network, defaultclock, mV, ms, volt, \
    PoissonGroup, SpikeGeneratorGroup, Mohm, second, nS, uS, TimedArray, Hz, nA, pA
import dill
from datetime import datetime
import os
from generate import set_spikes_from_time_varying_rate, fixed_current_series, \
    generate_adjacency_indices_within, generate_adjacency_indices_between, normal_positive_weights
import winsound
from results import Results
from functions import find_upstates
from results import bins_to_centers
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

    def initialize_units(self):
        unitModel = '''
        dv/dt = (gl * (eLeak - v) - iAdapt +
                 jE * sE - jI * sI) / Cm +
                 noiseSigma * (Cm / gl)**-0.5 * xi: volt
        diAdapt/dt = -iAdapt / tauAdapt : amp

        dsE/dt = (-sE + uE) / tauFallE : 1
        duE/dt = -uE / tauRiseE : 1
        dsI/dt = (-sI + uI) / tauFallI : 1
        duI/dt = -uI / tauRiseI : 1

        eLeak : volt
        jE : amp
        jI : amp
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

        vTauExc = self.p['membraneCapacitanceExc'] / self.p['gLeakExc']
        vTauInh = self.p['membraneCapacitanceInh'] / self.p['gLeakInh']

        # p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
        # p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])

        unitsExc.jE = vTauExc * self.p['jEE'] / self.p['nIncExc'] / ms  # jEE
        unitsExc.jI = vTauExc * self.p['jEI'] / self.p['nIncInh'] / ms  # jEI
        unitsInh.jE = vTauInh * self.p['jIE'] / self.p['nIncExc'] / ms  # jIE
        unitsInh.jI = vTauInh * self.p['jII'] / self.p['nIncInh'] / ms  # jII

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

        self.units = units
        self.unitsExc = unitsExc
        self.unitsInh = unitsInh
        self.N.add(units)

    def initialize_units_kickable(self):
        unitModel = '''
                dv/dt = (gl * (eLeak - v) - iAdapt +
                         jE * sE - jI * sI +
                         kKick * iKick) / Cm +
                         noiseSigma * (Cm / gl)**-0.5 * xi: volt
                diAdapt/dt = -iAdapt / tauAdapt : amp

                dsE/dt = (-sE + uE) / tauFallE : 1
                duE/dt = -uE / tauRiseE : 1
                dsI/dt = (-sI + uI) / tauFallI : 1
                duI/dt = -uI / tauRiseI : 1

                eLeak : volt
                jE : amp
                jI : amp
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
            refractory=self.p['refractoryPeriod'],
            clock=defaultclock,
        )
        unitsInh = NeuronGroup(
            N=self.p['nInh'],
            model=unitModel,
            method=self.p['updateMethod'],
            threshold=threshCode,
            reset=resetCode,
            refractory=self.p['refractoryPeriod'],
            clock=defaultclock,
        )

        self.p['nExcSpikemon'] = int(self.p['nExc'] * self.p['propSpikemon'])
        self.p['nInhSpikemon'] = int(self.p['nInh'] * self.p['propSpikemon'])

        vTauExc = self.p['membraneCapacitanceExc'] / self.p['gLeakExc']
        vTauInh = self.p['membraneCapacitanceInh'] / self.p['gLeakInh']

        unitsExc.jE = vTauExc * self.p['jEE'] / self.p['nIncExc'] / ms
        unitsExc.jI = vTauExc * self.p['jEI'] / self.p['nIncInh'] / ms
        unitsInh.jE = vTauInh * self.p['jIE'] / self.p['nIncExc'] / ms  # 50 nA * ~4 Hz = 200 nA / 100 = 2 nA
        unitsInh.jI = vTauInh * self.p['jII'] / self.p['nIncInh'] / ms  # 10 nA * ~8 Hz = 80 nA / 100 = 0.8 A

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

        self.unitsExc = unitsExc
        self.unitsInh = unitsInh
        self.N.add(unitsExc)
        self.N.add(unitsInh)

    def initialize_units_twice_kickable(self):
        unitModel = '''
                dv/dt = (gl * (eLeak - v) - iAdapt +
                         jE * sE - jI * sI + jExt * sExt +
                         kKick * iKick) / Cm +
                         noiseSigma * (Cm / gl)**-0.5 * xi: volt
                diAdapt/dt = -iAdapt / tauAdapt : amp

                dsE/dt = (-sE + uE) / tauFallE : 1
                duE/dt = -uE / tauRiseE : 1
                dsI/dt = (-sI + uI) / tauFallI : 1
                duI/dt = -uI / tauRiseI : 1
                dsExt/dt = (-sExt + uExt) / tauFallE : 1
                duExt/dt = -uExt / tauRiseE : 1

                eLeak : volt
                jE : amp
                jI : amp
                jExt : amp
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
            refractory=self.p['refractoryPeriod'],
            clock=defaultclock,
        )
        unitsInh = NeuronGroup(
            N=self.p['nInh'],
            model=unitModel,
            method=self.p['updateMethod'],
            threshold=threshCode,
            reset=resetCode,
            refractory=self.p['refractoryPeriod'],
            clock=defaultclock,
        )

        self.p['nExcSpikemon'] = int(self.p['nExc'] * self.p['propSpikemon'])
        self.p['nInhSpikemon'] = int(self.p['nInh'] * self.p['propSpikemon'])

        vTauExc = self.p['membraneCapacitanceExc'] / self.p['gLeakExc']
        vTauInh = self.p['membraneCapacitanceInh'] / self.p['gLeakInh']

        unitsExc.jExt = vTauExc * self.p['jEE'] / self.p['nIncExc'] / ms
        unitsInh.jExt = vTauInh * self.p['jIE'] / self.p['nIncExc'] / ms  # 50 nA * ~4 Hz = 200 nA / 100 = 2 nA

        weightScales = np.array([1 / self.p['nIncExc'] * vTauExc / ms,
                                 1 / self.p['nIncExc'] * vTauInh / ms,
                                 1 / self.p['nIncInh'] * vTauExc / ms,
                                 1 / self.p['nIncInh'] * vTauInh / ms])
        weightScales /= weightScales.max()

        self.p['wEEScale'] = weightScales[0]
        self.p['wIEScale'] = weightScales[1]
        self.p['wEIScale'] = weightScales[2]
        self.p['wIIScale'] = weightScales[3]

        # self.p['wEEScale'] = 1 / self.p['nIncExc'] * vTauExc / ms
        # self.p['wEIScale'] = 1 / self.p['nIncInh'] * vTauExc / ms
        # self.p['wIEScale'] = 1 / self.p['nIncExc'] * vTauInh / ms
        # self.p['wIIScale'] = 1 / self.p['nIncInh'] * vTauInh / ms

        unitsExc.jE = self.p['jEE'] / self.p['nIncExc'] * vTauExc / ms  # jEE
        unitsInh.jE = self.p['jIE'] / self.p['nIncExc'] * vTauInh / ms  # jIE # 50 nA * ~4 Hz = 200 nA / 100 = 2 nA
        unitsExc.jI = self.p['jEI'] / self.p['nIncInh'] * vTauExc / ms  # jEI
        unitsInh.jI = self.p['jII'] / self.p['nIncInh'] * vTauInh / ms  # jII # 10 nA * ~8 Hz = 80 nA / 100 = 0.8 A

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

        self.unitsExc = unitsExc
        self.unitsInh = unitsInh
        self.N.add(unitsExc)
        self.N.add(unitsInh)

    def initialize_units_twice_kickable2(self):
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

        self.unitsExc = unitsExc
        self.unitsInh = unitsInh
        self.N.add(unitsExc)
        self.N.add(unitsInh)

    def initialize_units_NMDA(self):
        unitModel = '''
        dv/dt = (gl * (eLeak - v) - iAdapt +
                 jE * sE - jI * sI +
                 jE_NMDA * s_NMDA_tot * int(v > vStepSigmoid) / (1 + Mg2 * exp(-kSigmoid * (v - vMidSigmoid) / mV)) +
                 kKick * iKick) / Cm +
                 noiseSigma * (Cm / gl)**-0.5 * xi: volt
        diAdapt/dt = -iAdapt / tauAdapt : amp

        dsE/dt = (-sE + uE) / tauFallE : 1
        duE/dt = -uE / tauRiseE : 1
        
        dsI/dt = (-sI + uI) / tauFallI : 1
        duI/dt = -uI / tauRiseI : 1

        eLeak : volt
        jE : amp
        jI : amp
        jE_NMDA : amp
        s_NMDA_tot : 1
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

        units = NeuronGroup(
            N=self.p['nUnits'],
            model=unitModel,
            method=self.p['updateMethod'],
            threshold=threshCode,
            reset=resetCode,
            refractory=self.p['refractoryPeriod'],
            dt=self.p['dt']
        )

        self.p['nInh'] = int(self.p['propInh'] * self.p['nUnits'])
        self.p['nExc'] = self.p['nUnits'] - self.p['nInh']
        self.p['nExcSpikemon'] = int(self.p['nExc'] * self.p['propSpikemon'])
        self.p['nInhSpikemon'] = int(self.p['nInh'] * self.p['propSpikemon'])

        unitsExc = units[:self.p['nExc']]
        unitsInh = units[self.p['nExc']:]

        vTauExc = self.p['membraneCapacitanceExc'] / self.p['gLeakExc']
        vTauInh = self.p['membraneCapacitanceInh'] / self.p['gLeakInh']

        unitsExc.jE = vTauExc * self.p['jEE'] / self.p['nIncExc'] / ms
        unitsExc.jI = vTauExc * self.p['jEI'] / self.p['nIncInh'] / ms
        unitsInh.jE = vTauInh * self.p['jIE'] / self.p['nIncExc'] / ms  # 50 nA * ~4 Hz = 200 nA / 100 = 2 nA
        unitsInh.jI = vTauInh * self.p['jII'] / self.p['nIncInh'] / ms  # 10 nA * ~8 Hz = 80 nA / 100 = 0.8 A

        unitsExc.jE_NMDA = vTauExc * self.p['jEE_NMDA'] / self.p['nIncExc'] / ms
        unitsInh.jE_NMDA = vTauInh * self.p['jIE_NMDA'] / self.p['nIncExc'] / ms

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

        self.units = units
        self.unitsExc = unitsExc
        self.unitsInh = unitsInh
        self.N.add(units)

    def initialize_units_NMDA2(self):
        unitModel = '''
        dv/dt = (gl * (eLeak - v) - iAdapt +
                 jE * sE - jI * sI +
                 jE_NMDA * sE_NMDA / (1 + exp(-kSigmoid * (v - vMidSigmoid) / mV)) * int(v > vStepSigmoid) +
                 kKick * iKick) / Cm +
                 noiseSigma * (Cm / gl)**-0.5 * xi: volt
        diAdapt/dt = -iAdapt / tauAdapt : amp

        dsE/dt = (-sE + uE) / tauFallE : 1
        duE/dt = -uE / tauRiseE : 1
        dsI/dt = (-sI + uI) / tauFallI : 1
        duI/dt = -uI / tauRiseI : 1
        dsE_NMDA/dt = -sE_NMDA / tauFallNMDA + alpha * uE_NMDA * (1 - sE_NMDA) : 1
        duE_NMDA/dt = -uE_NMDA / tauRiseNMDA : 1

        eLeak : volt
        jE : amp
        jI : amp
        jE_NMDA : amp
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

        units = NeuronGroup(
            N=self.p['nUnits'],
            model=unitModel,
            method=self.p['updateMethod'],
            threshold=threshCode,
            reset=resetCode,
            refractory=self.p['refractoryPeriod'],
            dt=self.p['dt']
        )

        self.p['nInh'] = int(self.p['propInh'] * self.p['nUnits'])
        self.p['nExc'] = self.p['nUnits'] - self.p['nInh']
        self.p['nExcSpikemon'] = int(self.p['nExc'] * self.p['propSpikemon'])
        self.p['nInhSpikemon'] = int(self.p['nInh'] * self.p['propSpikemon'])

        unitsExc = units[:self.p['nExc']]
        unitsInh = units[self.p['nExc']:]

        vTauExc = self.p['membraneCapacitanceExc'] / self.p['gLeakExc']
        vTauInh = self.p['membraneCapacitanceInh'] / self.p['gLeakInh']

        unitsExc.jE = vTauExc * self.p['jEE'] / self.p['nIncExc'] / ms
        unitsExc.jI = vTauExc * self.p['jEI'] / self.p['nIncInh'] / ms
        unitsInh.jE = vTauInh * self.p['jIE'] / self.p['nIncExc'] / ms  # 50 nA * ~4 Hz = 200 nA / 100 = 2 nA
        unitsInh.jI = vTauInh * self.p['jII'] / self.p['nIncInh'] / ms  # 10 nA * ~8 Hz = 80 nA / 100 = 0.8 A

        unitsExc.jE_NMDA = vTauExc * self.p['jEE_NMDA'] / self.p['nIncExc'] / ms
        unitsInh.jE_NMDA = vTauInh * self.p['jIE_NMDA'] / self.p['nIncExc'] / ms

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

        self.units = units
        self.unitsExc = unitsExc
        self.unitsInh = unitsInh
        self.N.add(units)

    def set_kicked_units(self, onlyKickExc=False):

        if onlyKickExc:
            unitsExcKicked = self.unitsExc[:int(self.p['nUnits'] * self.p['propKicked'])]
            unitsExcKicked.kKick = self.p['kickAmplitudeExc']
        else:
            unitsExcKicked = self.unitsExc[:int(self.p['nExc'] * self.p['propKicked'])]
            unitsExcKicked.kKick = self.p['kickAmplitudeExc']
            unitsInhKicked = self.unitsInh[:int(self.p['nInh'] * self.p['propKicked'])]
            unitsInhKicked.kKick = self.p['kickAmplitudeInh']

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

    def initialize_external_input_uncorrelated(self, excWeightMultiplier=1):

        inputGroupUncorrelated = PoissonGroup(int(self.p['nPoissonUncorrInputUnits']), self.p['poissonUncorrInputRate'])
        feedforwardSynapsesUncorr = Synapses(inputGroupUncorrelated, self.units,
                                             on_pre='uE_post += ' + str(
                                                 excWeightMultiplier / self.p['tauRiseExc'] * ms))

        feedforwardSynapsesUncorr.connect('i==j')

        self.N.add(inputGroupUncorrelated, feedforwardSynapsesUncorr)

    def initialize_recurrent_synapses(self):

        synapsesExc = Synapses(
            source=self.unitsExc,
            target=self.units,
            on_pre='uE_post += ' + str(1 / self.p['tauRiseExc'] * ms),
        )
        synapsesInh = Synapses(
            source=self.unitsInh,
            target=self.units,
            on_pre='uI_post += ' + str(1 / self.p['tauRiseInh'] * ms),
        )
        synapsesExc.connect(p=self.p['propConnect'])
        synapsesInh.connect(p=self.p['propConnect'])

        synapsesExc.delay = ((self.p['rng'].random(synapsesExc.delay.shape[0]) * self.p['delayExc'] /
                              defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesInh.delay = ((self.p['rng'].random(synapsesInh.delay.shape[0]) * self.p['delayInh'] /
                              defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self.synapsesExc = synapsesExc
        self.synapsesInh = synapsesInh
        self.N.add(synapsesExc, synapsesInh)

    def initialize_recurrent_excitation_NMDA(self):

        # implement this as an aspect of the pre/post unit
        # similar to the Vishwa/Dean solution for STP

        eqs_glut = '''
                s_NMDA_tot_post = w_NMDA * s_NMDA : 1 (summed)
                ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha * x * (1 - s_NMDA) : 1 (clock-driven)
                dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
                w_NMDA : 1
                '''

        eqs_pre_glut = '''
                x += 1
                '''

        synapsesExc = Synapses(
            source=self.unitsExc,
            target=self.units,
            model=eqs_glut,
            on_pre=eqs_pre_glut + 'uE_post += ' + str(1 / self.p['tauRiseExc'] * ms),
            method='euler',
        )
        synapsesExc.connect('i!=j', p=self.p['propConnect'])
        synapsesExc.w_NMDA[:] = 1

        synapsesExc.delay = ((self.p['rng'].random(synapsesExc.delay.shape[0]) * self.p['delayExc'] /
                              defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self.synapsesExc = synapsesExc
        self.N.add(synapsesExc)

    def initialize_recurrent_excitation_NMDA2(self):

        synapsesExc = Synapses(
            source=self.unitsExc,
            target=self.units,
            on_pre='uE_NMDA_post += ' + str(1 / self.p['tauRiseNMDA'] * ms) + '\n' + 'uE_post += ' + str(
                1 / self.p['tauRiseExc'] * ms),
        )
        synapsesExc.connect('i!=j', p=self.p['propConnect'])

        synapsesExc.delay = ((self.p['rng'].random(synapsesExc.delay.shape[0]) * self.p['delayExc'] /
                              defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self.synapsesExc = synapsesExc
        self.N.add(synapsesExc)

    def initialize_recurrent_inhibition(self):

        synapsesInh = Synapses(
            source=self.unitsInh,
            target=self.units,
            on_pre='uI_post += ' + str(1 / self.p['tauRiseInh'] * ms),
        )
        synapsesInh.connect(p=self.p['propConnect'])
        synapsesInh.delay = ((self.p['rng'].random(synapsesInh.delay.shape[0]) * self.p['delayInh'] /
                              defaultclock.dt).astype(int) + 1) * defaultclock.dt
        self.synapsesInh = synapsesInh
        self.N.add(synapsesInh)

    def initialize_recurrent_synapses2(self):

        # here we use pre-post notation

        synapsesEE = Synapses(
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre='uE_post += ' + str(1 / self.p['tauRiseExc'] * ms),
            # delay=self.p['delayExc'] / 2,
        )
        synapsesIE = Synapses(
            source=self.unitsExc,
            target=self.unitsInh,
            on_pre='uE_post += ' + str(1 / self.p['tauRiseExc'] * ms),
            # delay=self.p['delayExc'] / 2,
        )
        synapsesEI = Synapses(
            source=self.unitsInh,
            target=self.unitsExc,
            on_pre='uI_post += ' + str(1 / self.p['tauRiseInh'] * ms),
            # delay=self.p['delayInh'] / 2,
        )
        synapsesII = Synapses(
            source=self.unitsInh,
            target=self.unitsInh,
            on_pre='uI_post += ' + str(1 / self.p['tauRiseInh'] * ms),
            # delay=self.p['delayInh'] / 2,
        )
        synapsesEE.connect('i!=j', p=self.p['propConnect'])
        synapsesEI.connect('i!=j', p=self.p['propConnect'])
        synapsesIE.connect('i!=j', p=self.p['propConnect'])
        synapsesII.connect('i!=j', p=self.p['propConnect'])

        TESTING_GENN = False

        if TESTING_GENN:
            nEESynapses = np.round(self.p['nExc'] * self.p['nExc'] * self.p['propConnect'])
            nEISynapses = np.round(self.p['nInh'] * self.p['nExc'] * self.p['propConnect'])
            nIESynapses = np.round(self.p['nExc'] * self.p['nInh'] * self.p['propConnect'])
            nIISynapses = np.round(self.p['nInh'] * self.p['nInh'] * self.p['propConnect'])

            synapsesEE.delay = ((self.p['rng'].random(nEESynapses) * self.p['delayExc'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesEI.delay = ((self.p['rng'].random(nEISynapses) * self.p['delayInh'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesIE.delay = ((self.p['rng'].random(nIESynapses) * self.p['delayExc'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesII.delay = ((self.p['rng'].random(nIISynapses) * self.p['delayInh'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
        else:
            synapsesEE.delay = ((self.p['rng'].random(synapsesEE.delay.shape[0]) * self.p['delayExc'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesIE.delay = ((self.p['rng'].random(synapsesIE.delay.shape[0]) * self.p['delayExc'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesEI.delay = ((self.p['rng'].random(synapsesEI.delay.shape[0]) * self.p['delayInh'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesII.delay = ((self.p['rng'].random(synapsesII.delay.shape[0]) * self.p['delayInh'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self.synapsesEE = synapsesEE
        self.synapsesEI = synapsesEI
        self.synapsesIE = synapsesIE
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesEI, synapsesIE, synapsesII)

    def initialize_recurrent_synapses_4bundles_modifiable(self):

        # nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExc'] * self.p['propConnect'])
        # nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInh'] * self.p['propConnect'])
        tauRiseEOverMS = self.p['tauRiseExc'] / ms
        tauRiseIOverMS = self.p['tauRiseInh'] / ms
        vTauExcOverMS = self.p['membraneCapacitanceExc'] / self.p['gLeakExc'] / ms
        vTauInhOverMS = self.p['membraneCapacitanceInh'] / self.p['gLeakInh'] / ms

        if self.p['useOldWeightMagnitude']:
            usejEE = self.p['jEE'] / self.p['nIncExc'] * vTauExcOverMS
            usejIE = self.p['jIE'] / self.p['nIncExc'] * vTauInhOverMS
            usejEI = self.p['jEI'] / self.p['nIncInh'] * vTauExcOverMS
            usejII = self.p['jII'] / self.p['nIncInh'] * vTauInhOverMS
            onPreStrings = ('uE_post += jEE / tauRiseEOverMS',
                            'uE_post += jIE / tauRiseEOverMS',
                            'uI_post += jEI / tauRiseIOverMS',
                            'uI_post += jII / tauRiseIOverMS',)
        else:
            usejEE = self.p['jEE'] / self.p['nIncExc']
            usejIE = self.p['jIE'] / self.p['nIncExc']
            usejEI = self.p['jEI'] / self.p['nIncInh']
            usejII = self.p['jII'] / self.p['nIncInh']
            onPreStrings = ('uE_post += jEE * vTauExcOverMS / tauRiseEOverMS',
                            'uE_post += jIE * vTauInhOverMS / tauRiseEOverMS',
                            'uI_post += jEI * vTauExcOverMS / tauRiseIOverMS',
                            'uI_post += jII * vTauInhOverMS / tauRiseIOverMS',)

        # v1
        'uE_post += 1 / tauRiseEOverMS'

        # v2
        #
        # weightScales = np.array([1 / self.p['nIncExc'],
        #                          1 / self.p['nIncExc'],
        #                          1 / self.p['nIncInh'],
        #                          1 / self.p['nIncInh']])
        weightScales = np.array([1 / self.p['nIncExc'] * vTauExcOverMS,
                                 1 / self.p['nIncExc'] * vTauInhOverMS,
                                 1 / self.p['nIncInh'] * vTauExcOverMS,
                                 1 / self.p['nIncInh'] * vTauInhOverMS])
        # weightScales = np.array([1 / self.p['nIncExc'] * vTauExcOverMS / tauRiseEOverMS,
        #                          1 / self.p['nIncExc'] * vTauInhOverMS / tauRiseEOverMS,
        #                          1 / self.p['nIncInh'] * vTauExcOverMS / tauRiseIOverMS,
        #                          1 / self.p['nIncInh'] * vTauInhOverMS / tauRiseIOverMS])

        weightScales /= weightScales.max()
        self.p['wEEScale'] = weightScales[0]
        self.p['wIEScale'] = weightScales[1]
        self.p['wEIScale'] = weightScales[2]
        self.p['wIIScale'] = weightScales[3]

        # from E to E
        synapsesEE = Synapses(
            model='jEE: amp',
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre=onPreStrings[0],  # 'uE_post += 1 / tauRiseEOverMS'
        )
        # if self.p['propConnect'] == 1:
        #     synapsesEE.connect('i!=j', p=self.p['propConnect'])
        # else:
        preInds, postInds = generate_adjacency_indices_within(self.p['nExc'], self.p['propConnect'],
                                                              allowAutapses=False, rng=self.p['rng'])
        synapsesEE.connect(i=preInds, j=postInds)
        self.preEE = preInds
        self.posEE = postInds
        synapsesEE.jEE = usejEE

        # from E to I
        synapsesIE = Synapses(
            model='jIE: amp',
            source=self.unitsExc,
            target=self.unitsInh,
            on_pre=onPreStrings[1],
        )
        # if self.p['propConnect'] == 1:
        #     synapsesIE.connect('i!=j', p=self.p['propConnect'])
        # else:
        preInds, postInds = generate_adjacency_indices_between(self.p['nExc'], self.p['nInh'],
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
            on_pre=onPreStrings[2],
        )
        # if self.p['propConnect'] == 1:
        #     synapsesEI.connect('i!=j', p=self.p['propConnect'])
        # else:
        preInds, postInds = generate_adjacency_indices_between(self.p['nInh'], self.p['nExc'],
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
            on_pre=onPreStrings[3],
        )
        # if self.p['propConnect'] == 1:
        #     synapsesII.connect('i!=j', p=self.p['propConnect'])
        # else:
        preInds, postInds = generate_adjacency_indices_within(self.p['nInh'], self.p['propConnect'],
                                                              allowAutapses=False, rng=self.p['rng'])
        synapsesII.connect(i=preInds, j=postInds)
        self.preII = preInds
        self.posII = postInds
        synapsesII.jII = usejII

        TESTING_GENN = False

        if TESTING_GENN:
            nEESynapses = np.round(self.p['nExc'] * self.p['nExc'] * self.p['propConnect'])
            nEISynapses = np.round(self.p['nInh'] * self.p['nExc'] * self.p['propConnect'])
            nIESynapses = np.round(self.p['nExc'] * self.p['nInh'] * self.p['propConnect'])
            nIISynapses = np.round(self.p['nInh'] * self.p['nInh'] * self.p['propConnect'])

            synapsesEE.delay = ((self.p['rng'].random(nEESynapses) * self.p['delayExc'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesEI.delay = ((self.p['rng'].random(nEISynapses) * self.p['delayInh'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesIE.delay = ((self.p['rng'].random(nIESynapses) * self.p['delayExc'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesII.delay = ((self.p['rng'].random(nIISynapses) * self.p['delayInh'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
        else:
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

    def initialize_recurrent_synapses_4bundles_results(self, R):

        # nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExc'] * self.p['propConnect'])
        # nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInh'] * self.p['propConnect'])
        tauRiseEOverMS = R.p['tauRiseExc'] / ms
        tauRiseIOverMS = R.p['tauRiseInh'] / ms
        vTauExcOverMS = R.p['membraneCapacitanceExc'] / R.p['gLeakExc'] / ms
        vTauInhOverMS = R.p['membraneCapacitanceInh'] / R.p['gLeakInh'] / ms

        if R.p['useOldWeightMagnitude']:
            usejEE = R.p['jEE'] / R.p['nIncExc'] * vTauExcOverMS
            usejIE = R.p['jIE'] / R.p['nIncExc'] * vTauInhOverMS
            usejEI = R.p['jEI'] / R.p['nIncInh'] * vTauExcOverMS
            usejII = R.p['jII'] / R.p['nIncInh'] * vTauInhOverMS
            onPreStrings = ('uE_post += jEE / tauRiseEOverMS',
                            'uE_post += jIE / tauRiseEOverMS',
                            'uI_post += jEI / tauRiseIOverMS',
                            'uI_post += jII / tauRiseIOverMS',)
        else:
            usejEE = R.p['jEE'] / R.p['nIncExc']
            usejIE = R.p['jIE'] / R.p['nIncExc']
            usejEI = R.p['jEI'] / R.p['nIncInh']
            usejII = R.p['jII'] / R.p['nIncInh']
            onPreStrings = ('uE_post += jEE * vTauExcOverMS / tauRiseEOverMS',
                            'uE_post += jIE * vTauInhOverMS / tauRiseEOverMS',
                            'uI_post += jEI * vTauExcOverMS / tauRiseIOverMS',
                            'uI_post += jII * vTauInhOverMS / tauRiseIOverMS',)

        # weightScales = np.array([1 / R.p['nIncExc'],
        #                          1 / R.p['nIncExc'],
        #                          1 / R.p['nIncInh'],
        #                          1 / R.p['nIncInh']])
        weightScales = np.array([1 / R.p['nIncExc'] * vTauExcOverMS,
                                 1 / R.p['nIncExc'] * vTauInhOverMS,
                                 1 / R.p['nIncInh'] * vTauExcOverMS,
                                 1 / R.p['nIncInh'] * vTauInhOverMS])
        # weightScales = np.array([1 / R.p['nIncExc'] * vTauExcOverMS / tauRiseEOverMS,
        #                          1 / R.p['nIncExc'] * vTauInhOverMS / tauRiseEOverMS,
        #                          1 / R.p['nIncInh'] * vTauExcOverMS / tauRiseIOverMS,
        #                          1 / R.p['nIncInh'] * vTauInhOverMS / tauRiseIOverMS])

        weightScales /= weightScales.max()
        self.p['wEEScale'] = weightScales[0]
        self.p['wIEScale'] = weightScales[1]
        self.p['wEIScale'] = weightScales[2]
        self.p['wIIScale'] = weightScales[3]

        # from E to E
        synapsesEE = Synapses(
            model='jEE: amp',
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre=onPreStrings[0],
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
            on_pre=onPreStrings[1],
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
            on_pre=onPreStrings[2],
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
            on_pre=onPreStrings[3],
        )
        if R.p['propConnect'] == 1:
            synapsesII.connect('i!=j', p=R.p['propConnect'])
        else:
            preInds, postInds = R.preII, R.posII
            synapsesII.connect(i=preInds, j=postInds)
            self.preII = preInds
            self.posII = postInds
        synapsesII.jII = R.wII_final * pA

        TESTING_GENN = False

        if TESTING_GENN:
            nEESynapses = np.round(self.p['nExc'] * self.p['nExc'] * self.p['propConnect'])
            nEISynapses = np.round(self.p['nInh'] * self.p['nExc'] * self.p['propConnect'])
            nIESynapses = np.round(self.p['nExc'] * self.p['nInh'] * self.p['propConnect'])
            nIISynapses = np.round(self.p['nInh'] * self.p['nInh'] * self.p['propConnect'])

            synapsesEE.delay = ((self.p['rng'].random(nEESynapses) * self.p['delayExc'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesEI.delay = ((self.p['rng'].random(nEISynapses) * self.p['delayInh'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesIE.delay = ((self.p['rng'].random(nIESynapses) * self.p['delayExc'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
            synapsesII.delay = ((self.p['rng'].random(nIISynapses) * self.p['delayInh'] /
                                 defaultclock.dt).astype(int) + 1) * defaultclock.dt
        else:
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

    def initialize_recurrent_synapses3(self):

        # here we use pre-post notation

        synapsesEE = Synapses(
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre='uE_post += ' + str(1 / self.p['tauRiseExc'] * ms),
        )
        synapsesEI = Synapses(
            source=self.unitsInh,
            target=self.unitsExc,
            on_pre='uI_post += ' + str(1 / self.p['tauRiseInh'] * ms),
        )
        synapsesIE = Synapses(
            source=self.unitsExc,
            target=self.unitsInh,
            on_pre='uE_post += ' + str(1 / self.p['tauRiseExc'] * ms),
        )
        synapsesII = Synapses(
            source=self.unitsInh,
            target=self.unitsInh,
            on_pre='uI_post += ' + str(1 / self.p['tauRiseInh'] * ms),
        )

        CONNECTION_STYLE = 'organized'  # probabilistic or organized
        # OFFSET = int(self.p['nUnits'] * self.p['propKicked'])  # 1 by default
        OFFSET = 1

        if CONNECTION_STYLE == 'probabilistic':
            synapsesEE.connect(p=self.p['propConnect'])
            synapsesEI.connect(p=self.p['propConnect'])
            synapsesIE.connect(p=self.p['propConnect'])
            synapsesII.connect(p=self.p['propConnect'])
        elif CONNECTION_STYLE == 'organized':
            for synapseBundle in (synapsesEE, synapsesEI, synapsesIE, synapsesII):
                nPreSynapticUnits = synapseBundle.source.stop - synapseBundle.source.start
                nPostSynapticUnitsTotal = synapseBundle.target.stop - synapseBundle.target.start
                nPostSynapticUnitsTargeted = int(
                    (synapseBundle.target.stop - synapseBundle.target.start) * self.p['propConnect'])
                print('connecting {} presyn units to {} postsyn units'.format(nPreSynapticUnits,
                                                                              nPostSynapticUnitsTargeted))
                iList = []
                jList = []
                for presynUnitInd in range(nPreSynapticUnits):
                    # jList.append(
                    #     np.remainder(np.arange(presynUnitInd + OFFSET, presynUnitInd + nPostSynapticUnitsTargeted + OFFSET),
                    #                  nPostSynapticUnitsTotal))
                    stepper = int(nPostSynapticUnitsTotal / nPostSynapticUnitsTargeted)
                    jList.append(
                        np.remainder(
                            np.arange(presynUnitInd + OFFSET, presynUnitInd + OFFSET + nPostSynapticUnitsTotal,
                                      stepper), nPostSynapticUnitsTotal))
                    iList.append(np.ones((nPostSynapticUnitsTargeted,)) * presynUnitInd)
                iArray = np.concatenate(iList).astype(int)
                jArray = np.concatenate(jList).astype(int)
                synapseBundle.connect(i=iArray, j=jArray)

        synapsesEE.delay = ((self.p['rng'].random(synapsesEE.delay.shape[0]) * self.p['delayExc'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesIE.delay = ((self.p['rng'].random(synapsesIE.delay.shape[0]) * self.p['delayExc'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesEI.delay = ((self.p['rng'].random(synapsesEI.delay.shape[0]) * self.p['delayInh'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesII.delay = ((self.p['rng'].random(synapsesII.delay.shape[0]) * self.p['delayInh'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self.synapsesEE = synapsesEE
        self.synapsesEI = synapsesEI
        self.synapsesIE = synapsesIE
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesEI, synapsesIE, synapsesII)

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

    def build_classic(self):

        self.initialize_network()
        self.initialize_units()
        self.set_kicked_units()
        self.initialize_recurrent_synapses()
        self.create_monitors()

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

    def run_NMDA(self):

        iKickRecorded = self.p['iKickRecorded']
        noiseSigma = self.p['noiseSigma']
        tauRiseE = self.p['tauRiseExc']
        tauFallE = self.p['tauFallExc']
        tauRiseI = self.p['tauRiseInh']
        tauFallI = self.p['tauFallInh']
        tauAdapt = self.p['adaptTau']

        vStepSigmoid = self.p['vStepSigmoid']
        kSigmoid = self.p['kSigmoid']
        vMidSigmoid = self.p['vMidSigmoid']

        tau_NMDA_rise = self.p['tau_NMDA_rise']
        tau_NMDA_decay = self.p['tau_NMDA_decay']
        alpha = self.p['alpha']
        Mg2 = self.p['Mg2']

        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def run_NMDA2(self):

        iKickRecorded = self.p['iKickRecorded']
        noiseSigma = self.p['noiseSigma']
        tauRiseE = self.p['tauRiseExc']
        tauFallE = self.p['tauFallExc']
        tauRiseI = self.p['tauRiseInh']
        tauFallI = self.p['tauFallInh']
        tauAdapt = self.p['adaptTau']

        vStepSigmoid = self.p['vStepSigmoid']
        kSigmoid = self.p['kSigmoid']
        vMidSigmoid = self.p['vMidSigmoid']

        tauRiseNMDA = self.p['tauRiseNMDA']
        tauFallNMDA = self.p['tauFallNMDA']
        alpha = self.p['alpha']

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

        # all that's left is to monitor and run
        self.create_monitors()
        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def determine_fan_in_NMDA(self, minUnits=21, maxUnits=40, unitSpacing=1, timeSpacing=250 * ms):

        eqs_glut = '''
                s_NMDA_tot_post = w_NMDA * s_NMDA : 1 (summed)
                ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha * x * (1 - s_NMDA) : 1 (clock-driven)
                dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
                w_NMDA : 1
                '''

        eqs_pre_glut = '''
                x += 1
                '''

        # nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExcFan'] * self.p['propConnect'])
        # nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInhFan'] * self.p['propConnect'])

        # set the unit params that must be in the name space
        noiseSigma = 0 * self.p['noiseSigma']  # 0 for this experiment!!
        tauRiseE = self.p['tauRiseExc']
        tauFallE = self.p['tauFallExc']
        tauRiseI = self.p['tauRiseInh']
        tauFallI = self.p['tauFallInh']
        tauAdapt = self.p['adaptTau']

        ge_NMDA = 0
        # ge_NMDA = useQNMDA  # * 800. / self.p['nExc']
        tau_NMDA_rise = 2. * ms
        tau_NMDA_decay = 100. * ms
        alpha = 0.5 / ms
        Mg2 = 1.

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
            model=eqs_glut,
            on_pre=eqs_pre_glut + 'uE_post += ' + str(1 / tauRiseE * ms),
            method='euler',
        )
        feedforwardFanInh = Synapses(
            source=Fanners,
            target=self.unitsInh,
            model=eqs_glut,
            on_pre=eqs_pre_glut + 'uE_post += ' + str(1 / tauRiseE * ms),
            method='euler',
        )
        feedforwardFanExc.connect(p=1)
        feedforwardFanInh.connect(p=1)

        # add them to the network, set the run duration, create a bogus kick current
        self.N.add(Fanners, feedforwardFanExc, feedforwardFanInh)
        self.p['duration'] = (np.array(times).max() * second + timeSpacing)

        # this must be defined...
        iExtRecorded = fixed_current_series(1, self.p['duration'], self.p['dt'])

        # all that's left is to monitor and run
        self.create_monitors()
        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def prepare_upCrit_experiment(self, minUnits=170, maxUnits=180, unitSpacing=5, timeSpacing=3000 * ms,
                                  startTime=100 * ms, critExc=0.784 * volt, critInh=0.67625 * volt):

        tauRiseE = self.p['tauRiseExc']

        # multExc = critExc / self.unitsExc.jE[0]
        # multInh = critInh / self.unitsInh.jE[0]

        vTauExc = self.p['membraneCapacitanceExc'] / self.p['gLeakExc']
        jE = vTauExc * self.p['jEE'] / self.p['nIncExc'] / ms
        multExc = critExc / (jE * 100 * Mohm)
        multInh = critInh / (jE * 100 * Mohm)

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
            on_pre='uExt_post += ' + str(multExc / tauRiseE * ms),
        )
        feedforwardUpExc.connect('i==j')

        self.N.add(Uppers, feedforwardUpExc)

        self.p['duration'] = (np.array(times).max() * second + timeSpacing)
        self.p['iKickRecorded'] = fixed_current_series(0, self.p['duration'], self.p['dt'])

    def prepare_upCrit_experiment2(self, minUnits=170, maxUnits=180, unitSpacing=5, timeSpacing=3000 * ms,
                                   startTime=100 * ms, currentAmp=0.98 * nA):

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

        # TODO: CHECK IF THIS WORKS WITH uE_post += 0.98 * nA
        feedforwardUpExc = Synapses(
            source=Uppers,
            target=self.unitsExc,
            on_pre='uExt_post += 0.98 * nA'
            # on_pre='uE_post += ' + str(currentAmp / nA) + ' * nA'
            #  + str(critExc / (100 * Mohm) / tauRiseE * ms),
        )
        feedforwardUpExc.connect('i==j')

        self.N.add(Uppers, feedforwardUpExc)

        self.p['duration'] = (np.array(times).max() * second + timeSpacing)
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
                 ge * (Ee - v) + gi * (Ei - v)) / Cm: volt (unless refractory)
        dw/dt = (a * (v - El) - w) / tau_w : amp
        dge/dt = -ge / tau_e : siemens
        dgi/dt = -gi / tau_i : siemens
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

    def initialize_units_kickable(self):
        unitModel = '''
            dv/dt = (gl * (El - v) + gl * delta * exp((v - vThresh) / delta) - w +
                     gext * (Ee - v) + ge * (Ee - v) +  gi * (Ei - v)) / Cm: volt (unless refractory)
            dw/dt = (a * (v - El) - w) / tau_w : amp
            dge/dt = -ge / tau_e : siemens
            dgi/dt = -gi / tau_i : siemens
            dgext/dt = -gext / tau_e : siemens
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

    def initialize_units_iExt(self):
        unitModel = '''
        dv/dt = (gl * (El - v) + gl * delta * exp((v - vThresh) / delta) - w +
                 ge * (Ee - v) + gi * (Ei - v) + iAmp * iExt) / Cm: volt (unless refractory)
        dw/dt = (a * (v - El) - w) / tau_w : amp
        dge/dt = -ge / tau_e : siemens
        dgi/dt = -gi / tau_i : siemens
        El : volt
        delta: volt
        a : siemens
        b : amp
        vThresh : volt
        iAmp : amp
        iExt = iExtRecorded(t): 1
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
            dt=self.p['dt']
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

    def initialize_units_NMDA(self):
        # ge_NMDA * (Ee - v) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot +

        # int(v > -55 * mV) * ge_NMDA * (Ee - v) / (1 + Mg2 * exp(-0.2 * (v + 40 * mV) / mV)) * s_NMDA_tot +

        unitModel = '''
        dv/dt = (gl * (El - v) + gl * delta * exp((v - vThresh) / delta) - w +
                 ge * (Ee - v) + gi * (Ei - v) +
                 int(v > vStepSigmoid) * ge_NMDA * (Ee - v) / (1 + Mg2 * exp(-kSigmoid * (v - vMidSigmoid) / mV)) * s_NMDA_tot +
                 iAmp * iExt) / Cm: volt (unless refractory)
        dw/dt = (a * (v - El) - w) / tau_w : amp
        dge/dt = -ge / tau_e : siemens
        dgi/dt = -gi / tau_i : siemens
        El : volt
        delta: volt
        a : siemens
        b : amp
        vThresh : volt
        s_NMDA_tot : 1
        iAmp : amp
        iExt = iExtRecorded(t): 1
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
            dt=self.p['dt']
        )

        self.p['nInh'] = int(self.p['propInh'] * self.p['nUnits'])
        self.p['nExc'] = int(self.p['nUnits'] - self.p['nInh'])
        self.p['nExcSpikemon'] = int(self.p['nExc'] * self.p['propSpikemon'])
        self.p['nInhSpikemon'] = int(self.p['nInh'] * self.p['propSpikemon'])

        unitsExc = units[:self.p['nExc']]
        unitsExc.v = self.p['eLeakExc']
        unitsExc.El = self.p['eLeakExc']
        unitsExc.delta = self.p['deltaVExc']
        unitsExc.a = self.p['aExc']
        unitsExc.b = self.p['bExc']
        unitsExc.vThresh = self.p['vThresh']

        unitsInh = units[self.p['nExc']:]
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

    def initialize_recurrent_synapses(self):

        nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExc'] * self.p['propConnect'])
        nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInh'] * self.p['propConnect'])

        useQExc = self.p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
        useQInh = self.p['qInh'] / nRecurrentInhibitorySynapsesPerUnit

        synapsesExc = Synapses(
            source=self.unitsExc,
            target=self.units,
            on_pre='ge_post += ' + str(useQExc / nS) + ' * nS',
            # on_pre='ge_post += 0 * nS',  # to eliminate recurrent excitation
        )
        synapsesExc.connect('i!=j', p=self.p['propConnect'])

        synapsesInh = Synapses(
            source=self.unitsInh,
            target=self.units,
            on_pre='gi_post += ' + str(useQInh / nS) + ' * nS',
        )
        synapsesInh.connect('i!=j', p=self.p['propConnect'])

        self.synapsesExc = synapsesExc
        self.synapsesInh = synapsesInh
        self.N.add(synapsesExc, synapsesInh)

    def initialize_recurrent_synapses2(self):
        # this turned out not to be necessary....

        nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExc'] * self.p['propConnect'])
        nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInh'] * self.p['propConnect'])

        useQExc = self.p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
        useQInh = self.p['qInh'] / nRecurrentInhibitorySynapsesPerUnit

        synapsesEE = Synapses(
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre='ge_post += ' + str(useQExc / nS) + ' * nS',
            # on_pre='ge_post += 0 * nS',  # to eliminate recurrent excitation
        )
        synapsesIE = Synapses(
            source=self.unitsExc,
            target=self.unitsInh,
            on_pre='ge_post += ' + str(useQExc / nS) + ' * nS',
            # on_pre='ge_post += 0 * nS',  # to eliminate recurrent excitation
        )
        synapsesEI = Synapses(
            source=self.unitsInh,
            target=self.unitsExc,
            on_pre='gi_post += ' + str(useQInh / nS) + ' * nS',
        )
        synapsesII = Synapses(
            source=self.unitsInh,
            target=self.unitsInh,
            on_pre='gi_post += ' + str(useQInh / nS) + ' * nS',
        )
        synapsesEE.connect('i!=j', p=self.p['propConnect'])
        synapsesIE.connect('i!=j', p=self.p['propConnect'])
        synapsesEI.connect('i!=j', p=self.p['propConnect'])
        synapsesII.connect('i!=j', p=self.p['propConnect'])

        self.synapsesEE = synapsesEE
        self.synapsesIE = synapsesIE
        self.synapsesEI = synapsesEI
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesIE, synapsesEI, synapsesII)

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
            preInds, postInds = generate_adjacency_indices_within(self.p['nExc'], self.p['propConnect'],
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
            preInds, postInds = generate_adjacency_indices_between(self.p['nExc'], self.p['nInh'],
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
            preInds, postInds = generate_adjacency_indices_between(self.p['nInh'], self.p['nExc'],
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
            preInds, postInds = generate_adjacency_indices_within(self.p['nInh'], self.p['propConnect'],
                                                                  allowAutapses=False, rng=self.p['rng'])
            synapsesII.connect(i=preInds, j=postInds)

        self.synapsesEE = synapsesEE
        self.synapsesIE = synapsesIE
        self.synapsesEI = synapsesEI
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesIE, synapsesEI, synapsesII)

    def initialize_recurrent_synapses_4bundles_modifiable(self):

        nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExc'] * self.p['propConnect'])
        nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInh'] * self.p['propConnect'])

        useQExc = self.p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
        useQInh = self.p['qInh'] / nRecurrentInhibitorySynapsesPerUnit

        print(self.p['qExc'], nRecurrentExcitatorySynapsesPerUnit, useQExc)
        print(self.p['qInh'], nRecurrentInhibitorySynapsesPerUnit, useQInh)

        weightScales = np.array([1 / nRecurrentExcitatorySynapsesPerUnit,
                                 1 / nRecurrentExcitatorySynapsesPerUnit,
                                 1 / nRecurrentInhibitorySynapsesPerUnit,
                                 1 / nRecurrentInhibitorySynapsesPerUnit])
        weightScales /= weightScales.max()

        self.p['wEEScale'] = weightScales[0]
        self.p['wIEScale'] = weightScales[1]
        self.p['wEIScale'] = weightScales[2]
        self.p['wIIScale'] = weightScales[3]

        # from E to E
        synapsesEE = Synapses(
            model='qEE: siemens',
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre='ge_post += qEE',
        )
        if self.p['propConnect'] == 1:
            synapsesEE.connect('i!=j', p=self.p['propConnect'])
        else:
            preInds, postInds = generate_adjacency_indices_within(self.p['nExc'], self.p['propConnect'],
                                                                  allowAutapses=False, rng=self.p['rng'])
            synapsesEE.connect(i=preInds, j=postInds)
        synapsesEE.qEE = useQExc

        # from E to I
        synapsesIE = Synapses(
            model='qIE: siemens',
            source=self.unitsExc,
            target=self.unitsInh,
            on_pre='ge_post += qIE',
        )
        if self.p['propConnect'] == 1:
            synapsesIE.connect('i!=j', p=self.p['propConnect'])
        else:
            preInds, postInds = generate_adjacency_indices_between(self.p['nExc'], self.p['nInh'],
                                                                   self.p['propConnect'], rng=self.p['rng'])
            synapsesIE.connect(i=preInds, j=postInds)
        synapsesIE.qIE = useQExc

        # from I to E
        synapsesEI = Synapses(
            model='qEI: siemens',
            source=self.unitsInh,
            target=self.unitsExc,
            on_pre='gi_post += qEI',
        )
        if self.p['propConnect'] == 1:
            synapsesEI.connect('i!=j', p=self.p['propConnect'])
        else:
            preInds, postInds = generate_adjacency_indices_between(self.p['nInh'], self.p['nExc'],
                                                                   self.p['propConnect'], rng=self.p['rng'])
            synapsesEI.connect(i=preInds, j=postInds)
        synapsesEI.qEI = useQInh

        # from I to I
        synapsesII = Synapses(
            model='qII: siemens',
            source=self.unitsInh,
            target=self.unitsInh,
            on_pre='gi_post += qII',
        )
        if self.p['propConnect'] == 1:
            synapsesII.connect('i!=j', p=self.p['propConnect'])
        else:
            preInds, postInds = generate_adjacency_indices_within(self.p['nInh'], self.p['propConnect'],
                                                                  allowAutapses=False, rng=self.p['rng'])
            synapsesII.connect(i=preInds, j=postInds)
        synapsesII.qII = useQInh

        self.synapsesEE = synapsesEE
        self.synapsesIE = synapsesIE
        self.synapsesEI = synapsesEI
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesIE, synapsesEI, synapsesII)

    def initialize_recurrent_synapses_4bundles_separate(self):

        # from E to E
        self.p['nIncomingAvgEE'] = int(np.round(self.p['nExc'] * self.p['pEE']))
        print(self.p['qEE'], self.p['nIncomingAvgEE'])
        synapsesEE = Synapses(
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre='ge_post += ' + str(self.p['qEE'] / uS) + ' * uS / ' + str(self.p['nIncomingAvgEE']),
        )
        if self.p['pEE'] == 1:
            synapsesEE.connect('i!=j', p=self.p['pEE'])
        else:
            preInds, postInds = generate_adjacency_indices_within(self.p['nExc'], self.p['pEE'],
                                                                  allowAutapses=False, rng=self.p['rng'])
            synapsesEE.connect(i=preInds, j=postInds)

        # from E to I
        self.p['nIncomingAvgIE'] = int(np.round(self.p['nExc'] * self.p['pIE']))
        print(self.p['qIE'], self.p['nIncomingAvgIE'])
        synapsesIE = Synapses(
            source=self.unitsExc,
            target=self.unitsInh,
            on_pre='ge_post += ' + str(self.p['qIE'] / uS) + ' * uS / ' + str(self.p['nIncomingAvgIE']),
        )
        if self.p['pIE'] == 1:
            synapsesIE.connect('i!=j', p=self.p['pIE'])
        else:
            preInds, postInds = generate_adjacency_indices_between(self.p['nExc'], self.p['nInh'], self.p['pIE'],
                                                                   rng=self.p['rng'])
            synapsesIE.connect(i=preInds, j=postInds)

        # from I to E
        self.p['nIncomingAvgEI'] = int(np.round(self.p['nInh'] * self.p['pEI']))
        print(self.p['qEI'], self.p['nIncomingAvgEI'])
        synapsesEI = Synapses(
            source=self.unitsInh,
            target=self.unitsExc,
            on_pre='gi_post += ' + str(self.p['qEI'] / uS) + ' * uS / ' + str(self.p['nIncomingAvgEI']),
        )
        if self.p['pEI'] == 1:
            synapsesEI.connect('i!=j', p=self.p['pEI'])
        else:
            preInds, postInds = generate_adjacency_indices_between(self.p['nInh'], self.p['nExc'], self.p['pEI'],
                                                                   rng=self.p['rng'])
            synapsesEI.connect(i=preInds, j=postInds)

        # from I to I
        self.p['nIncomingAvgII'] = int(np.round(self.p['nInh'] * self.p['pII']))
        print(self.p['qII'], self.p['nIncomingAvgII'])
        synapsesII = Synapses(
            source=self.unitsInh,
            target=self.unitsInh,
            on_pre='gi_post += ' + str(self.p['qII'] / uS) + ' * uS / ' + str(self.p['nIncomingAvgII']),
        )
        if self.p['pII'] == 1:
            synapsesII.connect('i!=j', p=self.p['pII'])
        else:
            preInds, postInds = generate_adjacency_indices_within(self.p['nInh'], self.p['pII'],
                                                                  allowAutapses=False, rng=self.p['rng'])
            synapsesII.connect(i=preInds, j=postInds)

        self.synapsesEE = synapsesEE
        self.synapsesIE = synapsesIE
        self.synapsesEI = synapsesEI
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesIE, synapsesEI, synapsesII)

    def initialize_recurrent_synapses_4bundles_separate2(self):

        # from E to E
        synapsesEE = Synapses(
            model='q : siemens',
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre='ge_post += q',
        )
        if self.p['pEE'] == 1:
            synapsesEE.connect('i!=j', p=self.p['pEE'])
        else:
            preInds, postInds = generate_adjacency_indices_within(self.p['nExc'], self.p['pEE'],
                                                                  allowAutapses=False, rng=self.p['rng'])
            synapsesEE.connect(i=preInds, j=postInds)
        self.p['nIncomingAvgEE'] = int(np.round(self.p['nExc'] * self.p['pEE']))
        print(self.p['qEE'], self.p['nIncomingAvgEE'])
        synapsesEE.q = str(self.p['qEE'] / uS) + ' * uS / ' + str(self.p['nIncomingAvgEE'])
        print(synapsesEE.q[0])

        # from E to I
        synapsesIE = Synapses(
            model='q : siemens',
            source=self.unitsExc,
            target=self.unitsInh,
            on_pre='ge_post += q',
        )
        if self.p['pIE'] == 1:
            synapsesIE.connect('i!=j', p=self.p['pIE'])
        else:
            preInds, postInds = generate_adjacency_indices_between(self.p['nExc'], self.p['nInh'], self.p['pIE'],
                                                                   rng=self.p['rng'])
            synapsesIE.connect(i=preInds, j=postInds)
        self.p['nIncomingAvgIE'] = int(np.round(self.p['nExc'] * self.p['pIE']))
        print(self.p['qIE'], self.p['nIncomingAvgIE'])
        synapsesIE.q = str(self.p['qIE'] / uS) + ' * uS / ' + str(self.p['nIncomingAvgIE'])
        print(synapsesIE.q[0])

        # from I to E
        synapsesEI = Synapses(
            model='q : siemens',
            source=self.unitsInh,
            target=self.unitsExc,
            on_pre='ge_post += q',
        )
        if self.p['pEI'] == 1:
            synapsesEI.connect('i!=j', p=self.p['pEI'])
        else:
            preInds, postInds = generate_adjacency_indices_between(self.p['nInh'], self.p['nExc'], self.p['pEI'],
                                                                   rng=self.p['rng'])
            synapsesEI.connect(i=preInds, j=postInds)
        self.p['nIncomingAvgEI'] = int(np.round(self.p['nInh'] * self.p['pEI']))
        print(self.p['qEI'], self.p['nIncomingAvgEI'])
        synapsesEI.q = str(self.p['qEI'] / uS) + ' * uS / ' + str(self.p['nIncomingAvgEI'])
        print(synapsesEI.q[0])

        # from I to I
        synapsesII = Synapses(
            model='q : siemens',
            source=self.unitsInh,
            target=self.unitsInh,
            on_pre='ge_post += q',
        )
        if self.p['pII'] == 1:
            synapsesII.connect('i!=j', p=self.p['pII'])
        else:
            preInds, postInds = generate_adjacency_indices_within(self.p['nInh'], self.p['pII'],
                                                                  allowAutapses=False, rng=self.p['rng'])
            synapsesII.connect(i=preInds, j=postInds)
        self.p['nIncomingAvgII'] = int(np.round(self.p['nInh'] * self.p['pII']))
        print(self.p['qII'], self.p['nIncomingAvgII'])
        synapsesII.q = str(self.p['qII'] / uS) + ' * uS / ' + str(self.p['nIncomingAvgII'])
        print(synapsesII.q[0])

        self.synapsesEE = synapsesEE
        self.synapsesIE = synapsesIE
        self.synapsesEI = synapsesEI
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesIE, synapsesEI, synapsesII)

    def initialize_recurrent_synapses_4bundles_distributed(self, normalMean, normalSD):

        self.weightsDistributed = True

        nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExc'] * self.p['propConnect'])
        nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInh'] * self.p['propConnect'])

        useQExc = self.p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
        useQInh = self.p['qInh'] / nRecurrentInhibitorySynapsesPerUnit

        # from E to E
        synapsesEE = Synapses(
            model='wSyn : 1',
            source=self.unitsExc,
            target=self.unitsExc,
            on_pre='ge_post += wSyn * ' + str(useQExc / nS) + ' * nS',
        )
        preInds, postInds = generate_adjacency_indices_within(self.p['nExc'], self.p['propConnect'],
                                                              allowAutapses=False, rng=self.p['rng'])
        synapsesEE.connect(i=preInds, j=postInds)
        weights = normal_positive_weights(preInds.size, normalMean, normalSD, rng=self.p['rng'])
        synapsesEE.wSyn = weights
        self.preInds_EE = preInds
        self.postInds_EE = postInds
        self.weights_EE = weights

        # from E to I
        synapsesIE = Synapses(
            model='wSyn : 1',
            source=self.unitsExc,
            target=self.unitsInh,
            on_pre='ge_post += wSyn * ' + str(useQExc / nS) + ' * nS',
        )
        preInds, postInds = generate_adjacency_indices_between(self.p['nExc'], self.p['nInh'], self.p['propConnect'],
                                                               rng=self.p['rng'])
        synapsesIE.connect(i=preInds, j=postInds)
        weights = normal_positive_weights(preInds.size, normalMean, normalSD, rng=self.p['rng'])
        synapsesIE.wSyn = weights
        self.preInds_IE = preInds
        self.postInds_IE = postInds
        self.weights_IE = weights

        # from I to E
        synapsesEI = Synapses(
            model='wSyn : 1',
            source=self.unitsInh,
            target=self.unitsExc,
            on_pre='gi_post += wSyn * ' + str(useQInh / nS) + ' * nS',
        )
        preInds, postInds = generate_adjacency_indices_between(self.p['nInh'], self.p['nExc'], self.p['propConnect'],
                                                               rng=self.p['rng'])
        synapsesEI.connect(i=preInds, j=postInds)
        weights = normal_positive_weights(preInds.size, normalMean, normalSD, rng=self.p['rng'])
        synapsesEI.wSyn = weights
        self.preInds_EI = preInds
        self.postInds_EI = postInds
        self.weights_EI = weights

        # from I to I
        synapsesII = Synapses(
            model='wSyn : 1',
            source=self.unitsInh,
            target=self.unitsInh,
            on_pre='gi_post += wSyn * ' + str(useQInh / nS) + ' * nS',
        )
        preInds, postInds = generate_adjacency_indices_within(self.p['nInh'], self.p['propConnect'],
                                                              allowAutapses=False, rng=self.p['rng'])
        synapsesII.connect(i=preInds, j=postInds)
        weights = normal_positive_weights(preInds.size, normalMean, normalSD, rng=self.p['rng'])
        synapsesII.wSyn = weights
        self.preInds_II = preInds
        self.postInds_II = postInds
        self.weights_II = weights

        self.synapsesEE = synapsesEE
        self.synapsesIE = synapsesIE
        self.synapsesEI = synapsesEI
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesIE, synapsesEI, synapsesII)

    def initialize_recurrent_excitation_NMDA(self, scaleWeightsByIncSynapses=True):

        eqs_glut = '''
        s_NMDA_tot_post = w_NMDA * s_NMDA : 1 (summed)
        ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha * x * (1 - s_NMDA) : 1 (clock-driven)
        dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
        w_NMDA : 1
        '''

        eqs_pre_glut = '''
        x += 1
        '''

        if scaleWeightsByIncSynapses:
            nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExc'] * self.p['propConnect'])
            useQExc = self.p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
        else:
            useQExc = self.p['qExc']

        synapsesExc = Synapses(
            source=self.unitsExc,
            target=self.units,
            model=eqs_glut,
            on_pre=eqs_pre_glut + 'ge_post += ' + str(useQExc / nS) + ' * nS',
            # on_pre='ge_post += 0 * nS',  # to eliminate recurrent excitation
            method='euler',
        )
        synapsesExc.connect('i!=j', p=self.p['propConnect'])
        synapsesExc.w_NMDA[:] = 1

        self.synapsesExc = synapsesExc
        self.N.add(synapsesExc)

    def initialize_recurrent_inhibition(self, scaleWeightsByIncSynapses=True):

        if scaleWeightsByIncSynapses:
            nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInh'] * self.p['propConnect'])
            useQInh = self.p['qInh'] / nRecurrentInhibitorySynapsesPerUnit
        else:
            useQInh = self.p['qInh']

        synapsesInh = Synapses(
            source=self.unitsInh,
            target=self.units,
            on_pre='gi_post += ' + str(useQInh / nS) + ' * nS',
        )
        synapsesInh.connect('i!=j', p=self.p['propConnect'])

        self.synapsesInh = synapsesInh
        self.N.add(synapsesInh)

    def initialize_external_input(self):

        nFeedforwardSynapsesPerUnit = int(self.p['propConnectFeedforwardProjection'] * self.p['nPoissonInputUnits'] *
                                          (1 - self.p['propInh']))
        useQExcFeedforward = self.p['qExcFeedforward'] / nFeedforwardSynapsesPerUnit

        # set up the external input
        tNumpy = np.arange(int(self.p['duration'] / defaultclock.dt)) * float(defaultclock.dt)
        tRecorded = tNumpy * second
        vExtNumpy = np.zeros(tRecorded.shape)
        useExternalRate = float(self.p['poissonInputRate'])

        if self.p['poissonDriveType'] is 'ramp':
            vExtNumpy[:int(100 * ms / defaultclock.dt)] = np.linspace(0, useExternalRate,
                                                                      int(100 * ms / defaultclock.dt))
            vExtNumpy[int(100 * ms / defaultclock.dt):] = useExternalRate
        elif self.p['poissonDriveType'] is 'constant':
            vExtNumpy[:] = useExternalRate
        elif self.p['poissonDriveType'] is 'fullRamp':
            vExtNumpy = np.linspace(0, useExternalRate, tNumpy.size)

        vExtRecorded = TimedArray(vExtNumpy * Hz, dt=defaultclock.dt)

        if self.p['poissonInputsCorrelated']:
            useRateArray = vExtNumpy
        else:
            useRateArray = vExtNumpy * nFeedforwardSynapsesPerUnit
        # useRateArrayCorr = zeros(tRecorded.shape)
        # useRateArrayCorr[:] = float(correlatedInputRate)

        # TAKES A LONG TIME TO RUN BECAUSE IT GENERATES ALL THE POISSON DISTRIBUTED SPIKES FOR INPUT UNITS
        indices, times = set_spikes_from_time_varying_rate(time_array=tNumpy * 1e3,
                                                           rate_array=useRateArray,
                                                           nPoissonInputUnits=int(self.p['nPoissonInputUnits']),
                                                           rng=self.p['rng'])

        # weak possibly correlated input
        inputGroupWeak = SpikeGeneratorGroup(int(self.p['nPoissonInputUnits']), indices, times)
        feedforwardSynapsesWeak = Synapses(inputGroupWeak, self.units,
                                           on_pre='ge_post += ' + str(useQExcFeedforward / nS) + ' * nS')
        if self.p['poissonInputsCorrelated']:
            feedforwardSynapsesWeak.connect(p=self.p['propConnectFeedforwardProjection'])
        else:
            feedforwardSynapsesWeak.connect('i==j')

        self.N.add(inputGroupWeak, feedforwardSynapsesWeak)

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

    def initialize_prior_external_input_correlated(self, targetSim, loadFolder, targetExc=False):

        # TODO: be able to inherit the exact same feedforward connnectivity as well

        R = Results(targetSim, loadFolder)

        # the total sim duration (and dt?) should be the same for this to make sense
        assert self.p['duration'] == R.p['duration']
        assert self.p['dt'] == R.p['dt']

        # transfered parameters:
        # nPoissonCorrInputUnits, poissonCorrInputRate, indices, times
        # by default we will not monitor since the series is already known

        self.p['nPoissonCorrInputUnits'] = R.p['nPoissonCorrInputUnits']
        self.p['poissonCorrInputRate'] = R.p['poissonCorrInputRate']
        self.monitorInpCorr = 'inherit'
        self.spikeMonInpCorrI = R.spikeMonInpCorrI
        self.spikeMonInpCorrT = R.spikeMonInpCorrT

        # new parameters:
        # qExcFeedforwardCorr, propConnectFeedforwardProjectionCorr

        inputGroupCorrelated = SpikeGeneratorGroup(int(self.p['nPoissonCorrInputUnits']),
                                                   self.spikeMonInpCorrI,
                                                   self.spikeMonInpCorrT * second)

        if targetExc:
            feedforwardSynapsesCorr = Synapses(inputGroupCorrelated, self.unitsExc,
                                               on_pre='ge_post += ' + str(self.p['qExcFeedforwardCorr'] / nS) + ' * nS')
        else:
            feedforwardSynapsesCorr = Synapses(inputGroupCorrelated, self.units,
                                               on_pre='ge_post += ' + str(self.p['qExcFeedforwardCorr'] / nS) + ' * nS')

        feedforwardSynapsesCorr.connect(p=self.p['propConnectFeedforwardProjectionCorr'])
        self.N.add(inputGroupCorrelated, feedforwardSynapsesCorr)

    def initialize_external_input_memory(self, useQExcFeedforward, useExternalRate):

        inputGroupWeak = PoissonGroup(int(self.p['nPoissonInputUnits']), useExternalRate)
        feedforwardSynapsesWeak = Synapses(inputGroupWeak, self.units,
                                           on_pre='ge_post += ' + str(useQExcFeedforward / nS) + ' * nS')
        if self.p['poissonInputsCorrelated']:
            feedforwardSynapsesWeak.connect(p=self.p['propConnectFeedforwardProjection'])
        else:
            feedforwardSynapsesWeak.connect('i==j')
        self.N.add(inputGroupWeak, feedforwardSynapsesWeak)

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

    def create_monitors(self):

        spikeMonExc = SpikeMonitor(self.unitsExc[:self.p['nExcSpikemon']])
        spikeMonInh = SpikeMonitor(self.unitsInh[:self.p['nInhSpikemon']])

        stateMonExc = StateMonitor(self.unitsExc, self.p['recordStateVariables'],
                                   record=self.p['indsRecordStateExc'])
        stateMonInh = StateMonitor(self.unitsInh, self.p['recordStateVariables'],
                                   record=self.p['indsRecordStateInh'])

        self.spikeMonExc = spikeMonExc
        self.spikeMonInh = spikeMonInh
        self.stateMonExc = stateMonExc
        self.stateMonInh = stateMonInh
        self.N.add(spikeMonExc, spikeMonInh, stateMonExc, stateMonInh)

    def build_classic(self):

        self.initialize_network()
        self.initialize_units()
        self.initialize_external_input()
        self.initialize_recurrent_synapses()
        self.create_monitors()

    def run(self):

        # vThresh = self.p['vThresh']
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

    def run_NMDA(self):

        # vThresh = self.p['vThresh']
        Cm = self.p['membraneCapacitance']
        gl = self.p['gLeak']
        tau_w = self.p['adaptTau']
        Ee = self.p['eExcSyn']
        Ei = self.p['eInhSyn']
        tau_e = self.p['tauSynExc']
        tau_i = self.p['tauSynInh']
        Qe = self.p['qExc']
        Qi = self.p['qInh']

        vStepSigmoid = self.p['vStepSigmoid']
        kSigmoid = self.p['kSigmoid']
        vMidSigmoid = self.p['vMidSigmoid']

        tau_NMDA_rise = self.p['tau_NMDA_rise']
        tau_NMDA_decay = self.p['tau_NMDA_decay']
        alpha = self.p['alpha']
        Mg2 = self.p['Mg2']

        ge_NMDA = self.p['qExcNMDA']

        # if needed can define iExtRecorded here

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

        duration = 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)

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

    def determine_fan_in_NMDA(self, minUnits=21, maxUnits=40, unitSpacing=1, timeSpacing=250 * ms):

        eqs_glut = '''
                s_NMDA_tot_post = w_NMDA * s_NMDA : 1 (summed)
                ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha * x * (1 - s_NMDA) : 1 (clock-driven)
                dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
                w_NMDA : 1
                '''

        eqs_pre_glut = '''
                x += 1
                '''

        nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExcFan'] * self.p['propConnect'])
        nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInhFan'] * self.p['propConnect'])

        useQExc = self.p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
        useQInh = self.p['qInh'] / nRecurrentInhibitorySynapsesPerUnit
        useQNMDA = 0.5 * useQExc

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

        ge_NMDA = useQNMDA  # * 800. / self.p['nExc']
        tau_NMDA_rise = 2. * ms
        tau_NMDA_decay = 100. * ms
        alpha = 0.5 / ms
        Mg2 = 1.

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
            model=eqs_glut,
            on_pre=eqs_pre_glut + 'ge_post += ' + str(useQExc / nS) + ' * nS',
            method='euler',
        )
        feedforwardFanInh = Synapses(
            source=Fanners,
            target=self.unitsInh,
            model=eqs_glut,
            on_pre=eqs_pre_glut + 'ge_post += ' + str(useQExc / nS) + ' * nS',
            method='euler',
        )
        feedforwardFanExc.connect(p=1)
        feedforwardFanInh.connect(p=1)

        # add them to the network, set the run duration, create a bogus kick current
        self.N.add(Fanners, feedforwardFanExc, feedforwardFanInh)
        self.p['duration'] = (np.array(times).max() * second + timeSpacing)

        # this must be defined...
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
        dv/dt = (gl * (eLeak - v) - iAdapt + iExt) / Cm: volt (unless refractory)
        diAdapt/dt = -iAdapt / tauAdapt : amp
        
        eLeak : volt
        vReset : volt
        vThresh : volt
        betaAdapt : amp * second
        iExt : amp
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

        # just gonna comment this because it doesn't get used here but will be useful

        # vTauExc = self.p['membraneCapacitanceExc'] / self.p['gLeakExc']
        # vTauInh = self.p['membraneCapacitanceInh'] / self.p['gLeakInh']

        # unitsExc.jE = vTauExc * self.p['jEE'] / self.p['nIncExc'] / ms
        # unitsExc.jI = vTauExc * self.p['jEI'] / self.p['nIncInh'] / ms
        # unitsInh.jE = vTauInh * self.p['jIE'] / self.p['nIncExc'] / ms
        # unitsInh.jI = vTauInh * self.p['jII'] / self.p['nIncInh'] / ms

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

    def initialize_units_synaptic(self):
        unitModel = '''
        dv/dt = (gl * (eLeak - v) - iAdapt + sE) / Cm: volt (unless refractory)
        diAdapt/dt = -iAdapt / tauAdapt : amp
        
        dsE/dt = (-sE + uE) / tauFallE : amp
        duE/dt = -uE / tauRiseE : amp

        eLeak : volt
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

        # just gonna comment this because it doesn't get used here but will be useful

        # vTauExc = self.p['membraneCapacitanceExc'] / self.p['gLeakExc']
        # vTauInh = self.p['membraneCapacitanceInh'] / self.p['gLeakInh']

        # unitsExc.jE = vTauExc * self.p['jEE'] / self.p['nIncExc'] / ms
        # unitsExc.jI = vTauExc * self.p['jEI'] / self.p['nIncInh'] / ms
        # unitsInh.jE = vTauInh * self.p['jIE'] / self.p['nIncExc'] / ms
        # unitsInh.jI = vTauInh * self.p['jII'] / self.p['nIncInh'] / ms

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
        tauAdapt = self.p['adaptTau']

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

# OLD JERCOG WITH TIMEDARRAY

# unitModel = '''
#         dv/dt = (gl * (eLeak - v) - iAdapt +
#                  jE * sE - jI * sI +
#                  kKick * iKick) / Cm +
#                  noiseSigma * (Cm / gl)**-0.5 * xi: volt
#         diAdapt/dt = -iAdapt / tauAdapt : amp
#
#         dsE/dt = (-sE + uE) / tauFallE : 1
#         duE/dt = -uE / tauRiseE : 1
#         dsI/dt = (-sI + uI) / tauFallI : 1
#         duI/dt = -uI / tauRiseI : 1
#
#         eLeak : volt
#         jE : amp
#         jI : amp
#         kKick : amp
#         iKick = iKickRecorded(t) : 1
#         vReset : volt
#         vThresh : volt
#         betaAdapt : amp * second
#         gl : siemens
#         Cm : farad
#         '''
