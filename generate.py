"""
mathematical type functions that often generate something using RNG
or convert something related to that
"""


import numpy as np
from tqdm import tqdm
from brian2 import ms, second, TimedArray


def generate_poisson_kicks_jercog(lambda_value, duration, minimum_iki, maximum_iki):
    """ generate the times and sizes of the kicks, as used in Jercog et al. (2017) """

    kickTimes = []
    kickSizes = []
    kickInd = -1
    nextTime = 0
    while nextTime < duration:
        nextISI = -lambda_value / np.log(1 - np.random.rand())
        if nextISI < minimum_iki or nextISI > maximum_iki:
            continue
        kickInd += 1
        nextTime += nextISI
        if nextTime < duration:
            kickTimes.append(nextTime)
            kickSizes.append((2 - kickInd % 2) / 2)
    return kickTimes, kickSizes


def generate_poisson(rate, dt, duration, nUnits):
    """ given a fixed rate as well as the dt and duration of a simulation,
    generates nUnits independent Poisson processes and returns them in the way
    that SpikeGeneratorGroup is expecting them """

    timeArray = np.arange(0, float(duration), float(dt))
    randArray = np.random.rand(*timeArray.shape, nUnits)
    spikeBool = randArray < (rate * dt)

    times_lst = []
    indices_lst = []

    for unitInd in range(nUnits):
        tmpTimesArray = timeArray[spikeBool[:, unitInd]]
        times_lst.append(tmpTimesArray)
        indices_lst.append(np.ones_like(tmpTimesArray) * unitInd)

    indices = np.concatenate(indices_lst)
    times = np.concatenate(times_lst) * second

    return indices, times


def convert_indices_times_to_dict(indices, times):
    """ convert two equal-sized arrays representing times and indices of events (spikes, for examples)
    into a dict
        in which each key is an index (e.g. unit #)
        and the value is a list of the times that thing had an event """

    uniqueIndices = np.unique(indices.astype(int))

    spikeDict = {}
    for uniqInd in uniqueIndices:
        matchingBool = indices == uniqInd
        spikeDict[uniqInd] = times[matchingBool]

    return spikeDict


def convert_kicks_to_current_series(kickDur, kickTau, kickTimes, kickSizes, duration, dt):
    """ given the times and sizes of kicks, generate a time series of injected current values """

    tRecorded = np.arange(int(duration / dt)) * dt
    iKickNumpy = np.zeros(tRecorded.shape)
    kickShape = (1 - np.exp(-np.arange(0, kickDur, dt) / kickTau))
    kickDurInd = kickShape.size
    for tKick, sKick in zip(kickTimes, kickSizes):
        startInd = int(tKick / dt)
        iKickNumpy[startInd:startInd + kickDurInd] = kickShape * sKick
    iKickRecorded = TimedArray(iKickNumpy, dt=dt)
    return iKickRecorded


def fixed_current_series(amplitude, duration, dt):
    """ generate a fixed current series of a given amplitude / duration at a given dt """

    tRecorded = np.arange(int(duration / dt)) * dt
    iKickNumpy = amplitude * np.ones_like(tRecorded)
    iKickRecorded = TimedArray(iKickNumpy, dt=dt)
    return iKickRecorded


def set_spikes_from_time_varying_rate(time_array, rate_array, nPoissonInputUnits):
    """
    This function was inherited from some Destexhe code because I was trying to replicate their results.
    # time_array in ms
    # rate_array in Hz
    # nPoissonInputUnits dictates how many distinct external input units should be simulated
    """

    outIndices, outTimes = [], []
    DT = (time_array[1] - time_array[0])

    print('generating Poisson process')
    # for each time step (could have a different rate per time step i.e. inhomogeneous Poisson process)
    for it in tqdm(range(len(time_array))):
        # generate one random U(0, 1) for each unit
        rdm_num = np.random.random(nPoissonInputUnits)
        # for each unit decide whether it spikes on that time step
        for ii in np.arange(nPoissonInputUnits)[rdm_num < DT * rate_array[it] * 1e-3]:  # this samples numbers
            outIndices.append(ii)  # all the indicces
            outTimes.append(time_array[it])  # all the same time !

    return np.array(outIndices), np.array(outTimes) * ms


def generate_adjacency_matrix_within(nUnits, pConn, allowAutapses=False):
    """ creates a square bool array representing synaptic connections within nUnits
        that has a certain probability of connection and may or may not allow autapses.
        pre is row, post is column. """

    bestNumberOfSynapses = int(np.round(pConn * nUnits ** 2))

    if allowAutapses:
        indicesFlat = np.random.choice(nUnits ** 2, bestNumberOfSynapses, replace=False)
        adjMat = np.zeros((nUnits, nUnits), dtype=bool)
        adjMat[np.unravel_index(indicesFlat, (nUnits, nUnits))] = True
    else:
        probabilityArray = np.full((nUnits, nUnits), 1 / (nUnits * (nUnits - 1)))
        probabilityArray[np.diag_indices_from(probabilityArray)] = 0
        indicesFlat = np.random.choice(nUnits ** 2, bestNumberOfSynapses, replace=False, p=probabilityArray.ravel())
        adjMat = np.zeros((nUnits, nUnits), dtype=bool)
        adjMat[np.unravel_index(indicesFlat, (nUnits, nUnits))] = True

    return adjMat


def generate_adjacency_matrix_between(nUnitsPre, nUnitsPost, pConn):
    """ creates a rectangular bool array representing synaptic connections between 2 populations
        with a certain probability of connection
        pre is row, post is column. """

    bestNumberOfSynapses = int(np.round(pConn * nUnitsPre * nUnitsPost))
    indicesFlat = np.random.choice(nUnitsPre * nUnitsPost, bestNumberOfSynapses, replace=False)
    adjMat = np.zeros((nUnitsPre, nUnitsPost), dtype=bool)
    adjMat[np.unravel_index(indicesFlat, (nUnitsPre, nUnitsPost))] = True

    return adjMat