from brian2 import *
import dill
from datetime import datetime
import os
from generate import set_spikes_from_time_varying_rate


class BaseNetwork(object):
    """ represents descriptive information needed to create network in Brian.
        creates the NeuronGroup, sets variable/randomized self.p of units,
        creates the Synapses, connects them, sets any variable/randomized params of synapses,
        creates Monitor objects,
        eventually saves all created objects in a Network object (Brian), which can be passed
        for various simulation goals
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

        stateMonExc = StateMonitor(unitsExc, self.p['recordStateVariables'], record=list(range(self.p['nRecordStateExc'])))
        stateMonInh = StateMonitor(unitsInh, self.p['recordStateVariables'], record=list(range(self.p['nRecordStateInh'])))

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

    def __init__(self, params):
        self.p = params
        self.p['initTime'] = datetime.now().strftime('%Y-%m-%d-%H-%M')
        saveName = self.p['simName']
        if self.p['saveWithDate']:
            saveName += '_' + self.p['initTime']
        self.saveName = saveName

    def build(self):
        start_scope()

        unitModel = '''
        dv/dt = (eLeak - v - iAdapt +
                 jE * sE - jI * sI +
                 kKick * iKick) / tau +
                 noiseSigma * tau**-0.5 * xi: volt
        diAdapt/dt = -iAdapt / tauAdapt : volt

        dsE/dt = (-sE + uE) / tauFallE : 1
        duE/dt = -uE / tauRiseE : 1
        dsI/dt = (-sI + uI) / tauFallI : 1
        duI/dt = -uI / tauRiseI : 1

        eLeak : volt
        jE : volt
        jI : volt
        kKick : volt
        iKick = iKickRecorded(t) : 1
        tau : second
        vReset : volt
        vThresh : volt
        betaAdapt : volt * second
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

        unitsExc = units[:self.p['nExc']]
        unitsInh = units[self.p['nExc']:]

        unitsExc.jE = self.p['vTauExc'] * self.p['jEE'] / self.p['nExc'] / ms
        unitsExc.jI = self.p['vTauExc'] * self.p['jEI'] / self.p['nInh'] / ms
        unitsInh.jE = self.p['vTauInh'] * self.p['jIE'] / self.p['nExc'] / ms
        unitsInh.jI = self.p['vTauInh'] * self.p['jII'] / self.p['nInh'] / ms

        if self.p['scaleWeightsByPConn']:
            unitsExc.jE /= self.p['propConnect']
            unitsExc.jI /= self.p['propConnect']
            unitsInh.jE /= self.p['propConnect']
            unitsInh.jI /= self.p['propConnect']

        unitsExc.v = (self.p['vResetExc'] +
                      (self.p['vThreshExc'] - self.p['vResetExc']) * rand(self.p['nExc']))
        unitsExc.tau = self.p['vTauExc']
        unitsExc.vReset = self.p['vResetExc']
        unitsExc.vThresh = self.p['vThreshExc']
        unitsExc.betaAdapt = self.p['adaptStrengthExc'] * self.p['vTauExc']
        unitsExc.eLeak = self.p['eLeakExc']
        unitsExcKicked = unitsExc[:int(self.p['nExc'] * self.p['propKicked'])]
        unitsExcKicked.kKick = self.p['kickAmplitudeExc']

        unitsInh.v = (self.p['vResetInh'] +
                      (self.p['vThreshInh'] - self.p['vResetInh']) *
                      rand(self.p['nInh']))
        unitsInh.tau = self.p['vTauInh']
        unitsInh.vReset = self.p['vResetInh']
        unitsInh.vThresh = self.p['vThreshInh']
        unitsInh.betaAdapt = self.p['adaptStrengthInh'] * self.p['vTauInh']
        unitsInh.eLeak = self.p['eLeakInh']
        unitsInhKicked = unitsInh[:int(self.p['nInh'] * self.p['propKicked'])]
        unitsInhKicked.kKick = self.p['kickAmplitudeInh']

        synapsesExc = Synapses(
            source=unitsExc,
            target=units,
            on_pre='uE_post += ' + str(1 / self.p['tauRiseExc'] * ms),
        )
        synapsesInh = Synapses(
            source=unitsInh,
            target=units,
            on_pre='uI_post += ' + str(1 / self.p['tauRiseInh'] * ms),
        )
        synapsesExc.connect(p=self.p['propConnect'])
        synapsesInh.connect(p=self.p['propConnect'])

        spikeMonExc = SpikeMonitor(unitsExc[:self.p['nExcSpikemon']])
        spikeMonInh = SpikeMonitor(unitsInh[:self.p['nInhSpikemon']])

        stateMonExc = StateMonitor(unitsExc, self.p['recordStateVariables'], record=list(range(self.p['nRecordStateExc'])))
        stateMonInh = StateMonitor(unitsInh, self.p['recordStateVariables'], record=list(range(self.p['nRecordStateInh'])))

        N = Network(units, synapsesExc, synapsesInh, spikeMonExc, spikeMonInh, stateMonExc, stateMonInh)

        self.N = N
        self.spikeMonExc = spikeMonExc
        self.spikeMonInh = spikeMonInh
        self.stateMonExc = stateMonExc
        self.stateMonInh = stateMonInh

    def run(self):

        iKickRecorded = self.p['iKickRecorded']

        noiseSigma = self.p['noiseSigma']
        tauRiseE = self.p['tauRiseExc']
        tauFallE = self.p['tauFallExc']
        tauRiseI = self.p['tauRiseInh']
        tauFallI = self.p['tauFallInh']
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


class DestexheNetwork(object):

    def __init__(self, params):
        self.p = params
        self.p['initTime'] = datetime.now().strftime('%Y-%m-%d-%H-%M')
        saveName = self.p['simName']
        if self.p['saveWithDate']:
            saveName += '_' + self.p['initTime']
        self.saveName = saveName

    def build(self):
        start_scope()

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
        '''

        resetCode = '''
        v = El
        w += b
        '''

        threshCode = 'v > vThresh + 5 * delta'
        self.p['vThreshExc'] = self.p['vThresh'] + 5 * self.p['deltaVExc']
        self.p['vThreshInh'] = self.p['vThresh'] + 5 * self.p['deltaVInh']

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

        unitsInh.v = self.p['eLeakInh']
        unitsInh.El = self.p['eLeakInh']
        unitsInh.delta = self.p['deltaVInh']
        unitsInh.a = self.p['aInh']
        unitsInh.b = self.p['bInh']

        nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExc'] * self.p['propConnect'])
        nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInh'] * self.p['propConnect'])
        nFeedforwardSynapsesPerUnit = int(self.p['propConnectFeedforwardProjection'] * self.p['nPoissonInputUnits'] *
                                          (1 - self.p['propInh']))

        useQExc = self.p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
        useQInh = self.p['qInh'] / nRecurrentInhibitorySynapsesPerUnit
        useQExcFeedforward = self.p['qExcFeedforward'] / nFeedforwardSynapsesPerUnit

        # set up the external input
        tNumpy = arange(int(self.p['duration'] / defaultclock.dt)) * float(defaultclock.dt)
        tRecorded = tNumpy * second
        vExtNumpy = zeros(tRecorded.shape)
        useExternalRate = float(self.p['poissonInputRate'])

        if self.p['poissonDriveType'] is 'ramp':
            vExtNumpy[:int(100 * ms / defaultclock.dt)] = linspace(0, useExternalRate, int(100 * ms / defaultclock.dt))
            vExtNumpy[int(100 * ms / defaultclock.dt):] = useExternalRate
        elif self.p['poissonDriveType'] is 'constant':
            vExtNumpy[:] = useExternalRate
        elif self.p['poissonDriveType'] is 'fullRamp':
            vExtNumpy = linspace(0, useExternalRate, tNumpy.size)

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
                                                           nPoissonInputUnits=int(self.p['nPoissonInputUnits']))

        # weak possibly correlated input
        inputGroupWeak = SpikeGeneratorGroup(int(self.p['nPoissonInputUnits']), indices, times)
        feedforwardSynapsesWeak = Synapses(inputGroupWeak, units,
                                           on_pre='ge_post += ' + str(useQExcFeedforward / nS) + ' * nS')
        if self.p['poissonInputsCorrelated']:
            feedforwardSynapsesWeak.connect(p=self.p['propConnectFeedforwardProjection'])
        else:
            feedforwardSynapsesWeak.connect('i==j')

        synapsesExc = Synapses(
            source=unitsExc,
            target=units,
            on_pre='ge_post += ' + str(useQExc / nS) + ' * nS',
        )
        synapsesExc.connect('i!=j', p=self.p['propConnect'])

        synapsesInh = Synapses(
            source=unitsInh,
            target=units,
            on_pre='gi_post += ' + str(useQInh / nS) + ' * nS',
        )
        synapsesInh.connect('i!=j', p=self.p['propConnect'])

        spikeMonExc = SpikeMonitor(unitsExc[:self.p['nExcSpikemon']])
        spikeMonInh = SpikeMonitor(unitsInh[:self.p['nInhSpikemon']])

        stateMonExc = StateMonitor(unitsExc, self.p['recordStateVariables'],
                                   record=list(range(self.p['nRecordStateExc'])))
        stateMonInh = StateMonitor(unitsInh, self.p['recordStateVariables'],
                                   record=list(range(self.p['nRecordStateInh'])))

        # ALL units, spike generators, synapses, and monitors MUST BE INCLUDED HERE
        N = Network(units, synapsesExc, synapsesInh, spikeMonExc, spikeMonInh, stateMonExc, stateMonInh,
                    inputGroupWeak, feedforwardSynapsesWeak)

        self.N = N
        self.spikeMonExc = spikeMonExc
        self.spikeMonInh = spikeMonInh
        self.stateMonExc = stateMonExc
        self.stateMonInh = stateMonInh

    def run(self):

        vThresh = self.p['vThresh']
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


