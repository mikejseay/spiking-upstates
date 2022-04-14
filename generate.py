"""
mathematical type functions that often generate something using RNG
or convert something related to that
"""


from brian2 import *


def poisson_single(use_rate, dt, duration, rng=None):
    """ given a fixed rate as well as the dt and duration of a simulation,
    generates 1 Poisson process """

    if not rng:
        rng = np.random.default_rng(None)  # random seed

    timeArray = np.arange(0, float(duration), float(dt))
    randArray = rng.random(*timeArray.shape)
    spikeBool = randArray < (use_rate * dt)

    times_lst = []
    indices_lst = []

    times = timeArray[spikeBool]

    return times


def poisson(use_rate, dt, duration, nUnits, rng=None):
    """ given a fixed rate as well as the dt and duration of a simulation,
    generates nUnits independent Poisson processes and returns them in the way
    that SpikeGeneratorGroup is expecting them """

    if not rng:
        rng = np.random.default_rng(None)  # random seed

    timeArray = np.arange(0, float(duration), float(dt))
    randArray = rng.random(*timeArray.shape, nUnits)
    spikeBool = randArray < (use_rate * dt)

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


def square_bumps(kickTimes, kickDurs, kickSizes, duration, dt):
    """ given the times and sizes of kicks, generate a time series of injected current values """

    tRecorded = np.arange(int(duration / dt)) * dt
    iKickNumpy = np.zeros(tRecorded.shape)
    for tKick, dKick, sKick in zip(kickTimes, kickDurs, kickSizes):
        kickDurInd = int(dKick / dt)
        startInd = int(tKick / dt)
        iKickNumpy[startInd:startInd + kickDurInd] = 1
    iKickRecorded = TimedArray(iKickNumpy, dt=dt)
    return iKickRecorded


def convert_kicks_to_current_series(kickDur, kickTau, kickTimes, kickSizes, duration, dt):
    """ given the times and sizes of kicks, generate a time series of injected current values """

    tRecorded = np.arange(int(duration / dt)) * dt
    iKickNumpy = np.zeros(tRecorded.shape)
    kickShape = (1 - exp(-np.arange(0, kickDur, dt) / kickTau))
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


def adjacency_indices_within(nUnits, pConn, allowAutapses=False, rng=None):
    """ creates indices representing pre- and post-synaptic units within 1 population
        that has a certain probability of connection and may or may not allow autapses.
        preInds are first output, postInds are second. """

    if not rng:
        rng = np.random.default_rng(None)  # random seed

    bestNumberOfSynapses = int(np.round(pConn * nUnits ** 2))

    if allowAutapses:
        indicesFlat = rng.choice(nUnits ** 2, bestNumberOfSynapses, replace=False)
    else:
        probabilityArray = np.full((nUnits, nUnits), 1 / (nUnits * (nUnits - 1)))
        probabilityArray[np.diag_indices_from(probabilityArray)] = 0

        if pConn > (nUnits - 1) / nUnits:
            bestNumberOfSynapses -= int(np.round(nUnits ** 2 * (pConn - (nUnits - 1) / nUnits)))

        indicesFlat = rng.choice(nUnits ** 2, bestNumberOfSynapses, replace=False, p=probabilityArray.ravel())

    preInds, postInds = np.unravel_index(indicesFlat, (nUnits, nUnits))
    return preInds, postInds


def adjacency_indices_between(nUnitsPre, nUnitsPost, pConn, rng=None):
    """ creates indices representing pre- and post-synaptic units between 2 populations
        with a certain probability of connection
        preInds are first output, postInds are second. """

    if not rng:
        rng = np.random.default_rng(None)  # random seed

    bestNumberOfSynapses = int(np.round(pConn * nUnitsPre * nUnitsPost))
    indicesFlat = rng.choice(nUnitsPre * nUnitsPost, bestNumberOfSynapses, replace=False)

    preInds, postInds = np.unravel_index(indicesFlat, (nUnitsPre, nUnitsPost))

    return preInds, postInds


def norm_weights(nConnections, mean=1, sd=0.2, rng=None):
    """ given adjacency indices preInds and postinds, generate weight matrix w
    from a random normal distribution with mean and sd.
    clip negative weights to be 0.
    """

    if not rng:
        rng = np.random.default_rng(None)  # random seed

    weights = rng.normal(mean, sd, nConnections)
    weights[weights < 0] = 0

    return weights


def lognorm_weights(nConnections, mean=0, sd=0.75, rng=None):
    """ given adjacency indices preInds and postinds, generate weight matrix w
    from a random normal distribution with mean and sd.
    clip negative weights to be 0.
    """

    if not rng:
        rng = np.random.default_rng(None)  # random seed

    weights = rng.lognormal(mean, sd, nConnections)
    weights /= weights.mean()  # set the mean to be 1
    weights[weights < 0] = 0

    return weights


def adjacency_matrix_from_flat_inds(nUnitsPre, nUnitsPost, preInds, postInds):
    """ given the total number of pre- and post-synaptic units,
    and 3 flat arrays:
        the indices of the presynaptic units
        the indices of the postsynaptic units
        the weights
    generate a weight matrix that represents the synaptic weights
    """

    # these can be quite big so let's be careful about data type...
    shape = (nUnitsPre, nUnitsPost)
    a = np.zeros(shape, dtype=int)
    a[preInds, postInds] = 1

    return a


def weight_matrix_from_flat_inds_weights(nUnitsPre, nUnitsPost, preInds, postInds, weights):
    """ given the total number of pre- and post-synaptic units,
    and 3 flat arrays:
        the indices of the presynaptic units
        the indices of the postsynaptic units
        the weights
    generate a weight matrix that represents the synaptic weights
    """

    # these can be quite big so let's be careful about data type...
    shape = (nUnitsPre, nUnitsPost)
    # w = np.zeros(shape, dtype=np.float32)
    w = np.full(shape, np.nan, dtype=np.float32)
    w[preInds, postInds] = weights

    return w