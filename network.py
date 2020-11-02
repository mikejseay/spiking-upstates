from brian2 import *
import dill
from datetime import datetime
import os
from generate import set_spikes_from_time_varying_rate, fixed_current_series
import winsound
import win32api


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

        stateMonExc = StateMonitor(unitsExc, self.p['recordStateVariables'],
                                   record=list(range(self.p['nRecordStateExc'])))
        stateMonInh = StateMonitor(unitsInh, self.p['recordStateVariables'],
                                   record=list(range(self.p['nRecordStateInh'])))

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

    def initialize_network(self):
        start_scope()
        self.N = Network()

    def initialize_units(self):
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

        self.p['nInh'] = int(self.p['propInh'] * self.p['nUnits'])
        self.p['nExc'] = self.p['nUnits'] - self.p['nInh']
        self.p['nExcSpikemon'] = int(self.p['nExc'] * self.p['propSpikemon'])
        self.p['nInhSpikemon'] = int(self.p['nInh'] * self.p['propSpikemon'])

        unitsExc = units[:self.p['nExc']]
        unitsInh = units[self.p['nExc']:]

        unitsExc.jE = self.p['vTauExc'] * self.p['jEE'] / self.p['nIncExc'] / ms
        unitsExc.jI = self.p['vTauExc'] * self.p['jEI'] / self.p['nIncInh'] / ms
        unitsInh.jE = self.p['vTauInh'] * self.p['jIE'] / self.p['nIncExc'] / ms
        unitsInh.jI = self.p['vTauInh'] * self.p['jII'] / self.p['nIncInh'] / ms

        # do more carefully
        # scale by number of incoming connections?
        # in firing rate model wEE = # of incoming synapses * weight
        # if self.p['scaleWeightsByPConn']:
        #     unitsExc.jE /= self.p['propConnect']
        #     unitsExc.jI /= self.p['propConnect']
        #     unitsInh.jE /= self.p['propConnect']
        #     unitsInh.jI /= self.p['propConnect']

        # unitsExc.v = (self.p['vResetExc'] +
        #               (self.p['vThreshExc'] - self.p['vResetExc']) * rand(self.p['nExc']))
        unitsExc.v = self.p['eLeakExc']
        unitsExc.tau = self.p['vTauExc']
        unitsExc.vReset = self.p['vResetExc']
        unitsExc.vThresh = self.p['vThreshExc']
        unitsExc.betaAdapt = self.p['adaptStrengthExc'] * self.p['vTauExc']
        unitsExc.eLeak = self.p['eLeakExc']

        # unitsInh.v = (self.p['vResetInh'] +
        #               (self.p['vThreshInh'] - self.p['vResetInh']) *
        #               rand(self.p['nInh']))
        unitsExc.v = self.p['eLeakInh']
        unitsInh.tau = self.p['vTauInh']
        unitsInh.vReset = self.p['vResetInh']
        unitsInh.vThresh = self.p['vThreshInh']
        unitsInh.betaAdapt = self.p['adaptStrengthInh'] * self.p['vTauInh']
        unitsInh.eLeak = self.p['eLeakInh']

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

    def set_spiked_units(self, onlySpikeExc=True):

        nSpikedUnits = int(self.p['nUnits'] * self.p['propKicked'])

        tauRiseE = self.p['tauRiseExc']

        critExc = 0.784 * volt
        critInh = 0.67625 * volt
        multExc = critExc / self.unitsExc.jE[0]
        multInh = critInh / self.unitsInh.jE[0]

        indices = []
        times = []
        for kickTime in self.p['kickTimes']:
            indices.extend(list(range(nSpikedUnits)))
            times.extend([float(kickTime), ] * nSpikedUnits)

        Spikers = SpikeGeneratorGroup(nSpikedUnits, array(indices), array(times) * second)

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

        synapsesExc.delay = ((rand(synapsesExc.delay.shape[0]) * self.p['delayExc'] /
                              defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesInh.delay = ((rand(synapsesInh.delay.shape[0]) * self.p['delayInh'] /
                              defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self.synapsesExc = synapsesExc
        self.synapsesInh = synapsesInh
        self.N.add(synapsesExc, synapsesInh)

    def initialize_recurrent_synapses2(self):

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
        synapsesEE.connect(p=self.p['propConnect'])
        synapsesEI.connect(p=self.p['propConnect'])
        synapsesIE.connect(p=self.p['propConnect'])
        synapsesII.connect(p=self.p['propConnect'])

        synapsesEE.delay = ((rand(synapsesEE.delay.shape[0]) * self.p['delayExc'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesIE.delay = ((rand(synapsesIE.delay.shape[0]) * self.p['delayExc'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesEI.delay = ((rand(synapsesEI.delay.shape[0]) * self.p['delayInh'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesII.delay = ((rand(synapsesII.delay.shape[0]) * self.p['delayInh'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt

        self.synapsesEE = synapsesEE
        self.synapsesEI = synapsesEI
        self.synapsesIE = synapsesIE
        self.synapsesII = synapsesII
        self.N.add(synapsesEE, synapsesEI, synapsesIE, synapsesII)

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
                print('connecting {} presyn units to {} postsyn units'.format(nPreSynapticUnits, nPostSynapticUnitsTargeted))
                iList = []
                jList = []
                for presynUnitInd in range(nPreSynapticUnits):
                    # jList.append(
                    #     np.remainder(np.arange(presynUnitInd + OFFSET, presynUnitInd + nPostSynapticUnitsTargeted + OFFSET),
                    #                  nPostSynapticUnitsTotal))
                    stepper = int(nPostSynapticUnitsTotal / nPostSynapticUnitsTargeted)
                    jList.append(
                        np.remainder(
                            np.arange(presynUnitInd + OFFSET, presynUnitInd + OFFSET + nPostSynapticUnitsTotal, stepper), nPostSynapticUnitsTotal))
                    iList.append(np.ones((nPostSynapticUnitsTargeted,)) * presynUnitInd)
                iArray = np.concatenate(iList).astype(int)
                jArray = np.concatenate(jList).astype(int)
                synapseBundle.connect(i=iArray, j=jArray)

        synapsesEE.delay = ((rand(synapsesEE.delay.shape[0]) * self.p['delayExc'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesIE.delay = ((rand(synapsesIE.delay.shape[0]) * self.p['delayExc'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesEI.delay = ((rand(synapsesEI.delay.shape[0]) * self.p['delayInh'] /
                             defaultclock.dt).astype(int) + 1) * defaultclock.dt
        synapsesII.delay = ((rand(synapsesII.delay.shape[0]) * self.p['delayInh'] /
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
                                   record=list(range(self.p['nRecordStateExc'])))
        stateMonInh = StateMonitor(self.unitsInh, self.p['recordStateVariables'],
                                   record=list(range(self.p['nRecordStateInh'])))

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

        # print('\a')

        duration = 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)

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
        Fanners = SpikeGeneratorGroup(maxUnits, array(indices), array(times) * second)
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
        self.p['duration'] = (array(times).max() * second + timeSpacing)
        iKickRecorded = fixed_current_series(0, self.p['duration'], self.p['dt'])

        # all that's left is to monitor and run
        self.create_monitors()
        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def prepare_upCrit_experiment(self, minUnits=170, maxUnits=180, unitSpacing=5, timeSpacing=3000 * ms):

        tauRiseE = self.p['tauRiseExc']

        critExc = 0.784 * volt
        critInh = 0.67625 * volt
        multExc = critExc / self.unitsExc.jE[0]
        multInh = critInh / self.unitsInh.jE[0]

        indices = []
        times = []
        dummyInd = 0
        useRange = range(minUnits, maxUnits + 1, unitSpacing)
        for unitInd in useRange:
            dummyInd += 1
            indices.extend(list(range(unitInd)))
            times.extend([float(timeSpacing) * dummyInd, ] * (unitInd))

        Uppers = SpikeGeneratorGroup(maxUnits, array(indices), array(times) * second)

        feedforwardUpExc = Synapses(
            source=Uppers,
            target=self.unitsExc,
            on_pre='uE_post += ' + str(multExc / tauRiseE * ms),
        )
        feedforwardUpExc.connect('i==j')

        self.N.add(Uppers, feedforwardUpExc)

        self.p['duration'] = (array(times).max() * second + timeSpacing)
        self.p['iKickRecorded'] = fixed_current_series(0, self.p['duration'], self.p['dt'])


class DestexheNetwork(object):

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

        unitModel = '''
        dv/dt = (gl * (El - v) + gl * delta * exp((v - vThresh) / delta) - w +
                 ge * (Ee - v) + gi * (Ei - v) +
                 ge_NMDA * (Ee - v) / (1 + Mg2 * exp(-0.2 * (v + 40 * mV) / mV)) * s_NMDA_tot +
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

    def initialize_recurrent_excitation_NMDA(self, useQExc=None):

        eqs_glut = '''
        s_NMDA_tot_post = w_NMDA * s_NMDA : 1 (summed)
        ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha * x * (1 - s_NMDA) : 1 (clock-driven)
        dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
        w_NMDA : 1
        '''

        eqs_pre_glut = '''
        x += 1
        '''

        if not useQExc:
            nRecurrentExcitatorySynapsesPerUnit = int(self.p['nExc'] * self.p['propConnect'])
            useQExc = self.p['qExc'] / nRecurrentExcitatorySynapsesPerUnit

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

    def initialize_recurrent_inhibition(self, useQInh=None):

        if not useQInh:
            nRecurrentInhibitorySynapsesPerUnit = int(self.p['nInh'] * self.p['propConnect'])
            useQInh = self.p['qInh'] / nRecurrentInhibitorySynapsesPerUnit

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

    def initialize_external_input_correlated(self, targetExc=False):

        inputGroupCorrelated = PoissonGroup(int(self.p['nPoissonCorrInputUnits']), self.p['poissonCorrInputRate'])

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

        Spikers = SpikeGeneratorGroup(nSpikedUnits, array(indices), array(times) * second)

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
                                   record=list(range(self.p['nRecordStateExc'])))
        stateMonInh = StateMonitor(self.unitsInh, self.p['recordStateVariables'],
                                   record=list(range(self.p['nRecordStateInh'])))

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

    def run_NMDA(self, iExtRecorded=None, useQNMDA=None):

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

        ge_NMDA = useQNMDA  # * 800. / self.p['nExc']
        tau_NMDA_rise = 1. * ms
        tau_NMDA_decay = 50. * ms
        alpha = 0.5 / ms
        Mg2 = 1.

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
        Fanners = SpikeGeneratorGroup(maxUnits, array(indices), array(times) * second)
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
        self.p['duration'] = (array(times).max() * second + timeSpacing)

        # all that's left is to monitor and run
        self.create_monitors()
        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def determine_fan_in_NMDA(self, minUnits=21, maxUnits=40, unitSpacing=1, timeSpacing=250 * ms):

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
        Fanners = SpikeGeneratorGroup(maxUnits, array(indices), array(times) * second)
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
        self.p['duration'] = (array(times).max() * second + timeSpacing)

        # all that's left is to monitor and run
        self.create_monitors()
        self.N.run(self.p['duration'],
                   report=self.p['reportType'],
                   report_period=self.p['reportPeriod'],
                   profile=self.p['doProfile']
                   )

    def prepare_upCrit_experiment(self, minUnits=170, maxUnits=180, unitSpacing=5, timeSpacing=3000 * ms):

        indices = []
        times = []
        dummyInd = 0
        useRange = range(minUnits, maxUnits + 1, unitSpacing)
        for unitInd in useRange:
            dummyInd += 1
            indices.extend(list(range(unitInd)))
            times.extend([float(timeSpacing) * dummyInd, ] * (unitInd))

        Uppers = SpikeGeneratorGroup(maxUnits, array(indices), array(times) * second)

        feedforwardUpExc = Synapses(
            source=Uppers,
            target=self.unitsExc,
            on_pre='ge_post += 18.5 * nS',
        )
        feedforwardUpExc.connect('i==j')

        self.N.add(Uppers, feedforwardUpExc)

        self.p['duration'] = (array(times).max() * second + timeSpacing)
