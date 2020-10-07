from brian2 import *
import dill
from datetime import datetime
import os


class JercogNetwork(object):
    """ represents descriptive information needed to create network in Brian.
        creates the NeuronGroup, sets variable/randomized self.p of units,
        creates the Synapses, connects them, sets any variable/randomized params of synapses,
        creates Monitor objects,
        eventually saves all created objects in a Network object (Brian), which can be passed
        for various simulation goals
    """

    def __init__(self, params):
        self.p = params
        self.initTime = datetime.now().strftime('%Y-%m-%d-%H-%M')

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
        # all results are numpy arrays (ideally)
        # spikeMonExc.t and spikeMonInh.t
        # stateMonExc.v and stateMonInh.v
        # should be saved with numpy savez

        useDType = np.single

        spikeMonExcT = np.array(self.spikeMonExc.t, dtype=useDType)
        spikeMonExcI = np.array(self.spikeMonExc.i, dtype=useDType)
        spikeMonInhT = np.array(self.spikeMonInh.t, dtype=useDType)
        spikeMonInhI = np.array(self.spikeMonInh.i, dtype=useDType)
        stateMonExcV = np.array(self.stateMonExc.v / mV, dtype=useDType)
        stateMonInhV = np.array(self.stateMonInh.v / mV, dtype=useDType)

        savePath = os.path.join(self.p['saveFolder'],
                                self.p['simName'] + '_' + self.initTime + '_results.npz')

        np.savez(savePath, spikeMonExcT=spikeMonExcT, spikeMonExcI=spikeMonExcI, spikeMonInhT=spikeMonInhT,
                 spikeMonInhI=spikeMonInhI, stateMonExcV=stateMonExcV, stateMonInhV=stateMonInhV)

    def save_params(self):
        # save the params dictionary to a file with pickle

        savePath = os.path.join(self.p['saveFolder'],
                                self.p['simName'] + '_' + self.initTime + '_params.pkl')
        with open(savePath, 'wb') as f:
            dill.dump(self.p, f)
