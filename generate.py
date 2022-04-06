"""
mathematical type functions that often generate something using RNG
or convert something related to that
"""


from brian2 import *  # for some reason i have to import everything for fixed_current_series to work correctly...


def poisson_single(rate, dt, duration, rng=None):
    """ given a fixed rate as well as the dt and duration of a simulation,
    generates 1 Poisson process """

    if not rng:
        rng = np.random.default_rng(None)  # random seed

    timeArray = np.arange(0, float(duration), float(dt))
    randArray = rng.random(*timeArray.shape)
    spikeBool = randArray < (rate * dt)

    times_lst = []
    indices_lst = []

    times = timeArray[spikeBool]

    return times


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
    w = np.full(shape, np.nan, dtype=np.float32)
    w[preInds, postInds] = weights

    return w
