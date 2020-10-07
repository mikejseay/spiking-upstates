from brian2 import *


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

