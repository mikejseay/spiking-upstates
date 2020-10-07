from brian2 import *
from tqdm import tqdm


def generate_poisson_kicks_jercog(lambda_value, duration, minimum_iki, maximum_iki):
    """ generate the times and sizes of the kicks, as used in Jercog et al. (2017) """

    kickTimes = []
    kickSizes = []
    kickInd = -1
    nextTime = 0
    while nextTime < duration:
        nextISI = -lambda_value / log(1 - rand())
        if nextISI < minimum_iki or nextISI > maximum_iki:
            continue
        kickInd += 1
        nextTime += nextISI
        if nextTime < duration:
            kickTimes.append(nextTime)
            kickSizes.append((2 - kickInd % 2) / 2)
    return kickTimes, kickSizes


def convert_kicks_to_current_series(kickDur, kickTau, kickTimes, kickSizes, duration, dt):
    """ given the times and sizes of kicks, generate a time series of injected current values """

    tRecorded = arange(int(duration / dt)) * dt
    iKickNumpy = zeros(tRecorded.shape)
    kickShape = (1 - exp(-arange(0, kickDur, dt) / kickTau))
    kickDurInd = kickShape.size
    for tKick, sKick in zip(kickTimes, kickSizes):
        startInd = int(tKick / dt)
        iKickNumpy[startInd:startInd + kickDurInd] = kickShape * sKick
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

    return array(outIndices), array(outTimes) * ms
