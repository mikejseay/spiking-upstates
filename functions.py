"""
general functions that can't be categorized easily
"""

import numpy as np


def bins_to_centers(bins):
    return (bins[:-1] + bins[1:]) / 2


def find_upstates(v, dt, v_thresh, dur_thresh=None, extension_thresh=None, last_up_must_end=True):
    # given 1d signal, returns indices of on-sets and off-sets of "upstates"

    # input arguments:
    # v: voltage time series
    # dt: 1 / sampling rate of v
    # v_thresh: voltage threshold above which v must go to be "up"
    # dur_thresh: minimum upstate duration (in same units as dt)
    # extension_thresh: minimum downstate duration (in same units as dt)

    # returns:
    # u_ons: indices of upstate onsets
    # u_off: indices of upstate offsets

    # define logical vectors indicating where signal is at-or-above and below the threshold
    above_bool = v >= v_thresh
    below_bool = v < v_thresh

    # define logical vectors indicating points of upward / downward crossings
    upward_crossings = below_bool[:-1] & above_bool[1:]
    upward_crossings = np.insert(upward_crossings, 0, False)
    downward_crossings = above_bool[:-1] & below_bool[1:]
    downward_crossings = np.insert(downward_crossings, 0, False)

    # find crossing locations: these are the putative up and down transitions
    ups = np.where(upward_crossings)[0]
    downs = np.where(downward_crossings)[0]

    # check if there is one more up than downs. if above_bool until end, call the end a down, if not last_up_must_end
    if ups.size - downs.size == 1:
        if last_up_must_end:
            ups = ups[:-1]
        else:
            downs = np.insert(downs, len(downs), len(v))

    # no upstates? return empty vectors
    if ups.size == 0 or downs.size == 0:
        ups = np.array([])
        downs = np.array([])
        return ups, downs

    # recording could have started anded during different states
    # (e.g. start during upstate & end during downstate or vice versa)
    # in which case one putative up or down transition will not be paired with its buddy
    # we choose the convention that the first putative event should be an up transition
    # and all up transitions should be paired with a subsequent down transition
    if downs[0] < ups[0]:
        downs = downs[1:]

    # check once more if we have no true upstates
    if ups.size == 0 or downs.size == 0:
        ups = np.array([])
        downs = np.array([])
        return ups, downs

    # check if the final up transition has no down transition
    if ups[-1] > downs[-1]:
        if last_up_must_end:
            ups = ups[:-1]
        else:
            downs = np.insert(downs, len(downs), len(v))

    # check once more if we have no true upstates
    if ups.size == 0 or downs.size == 0:
        ups = np.array([])
        downs = np.array([])
        return ups, downs

    # ensure the above worked and we have equal numbers of putative up and down transitions
    assert len(ups) == len(downs)

    if extension_thresh:
        # calculate downstate durations (in points)
        down_durs = ups[1:] - downs[:-1]

        # delete short downstates
        # (this effectively combines upstates that are separated by short downstates)
        keep_downs = down_durs > extension_thresh / dt
        ups = ups[np.concatenate((np.array([True]), keep_downs))]
        downs = downs[np.concatenate((keep_downs, np.array([True])))]

    if dur_thresh:
        # delete short upstates
        up_durs = downs - ups
        long_durs = up_durs > dur_thresh / dt
        ups = ups[long_durs]
        downs = downs[long_durs]

    assert len(ups) == len(downs)

    return ups, downs


# inputFile = 'C:\\Users\\mikejseay\\Documents\\Code\\volo-2019-updown-FR.npy'
#
# with open(inputFile, 'rb') as f:
#     FRExc = np.load(f)
#     FRInh = np.load(f)
#
# dtHist = 0.005  # seconds
# V_THRESH = 0.2  # Hz
# MIN_UPSTATE_DUR = 0.3  # seconds
# MIN_DOWNSTATE_DUR = 0.3  # seconds
#
# myUps, myDowns = find_upstates(FRInh, dtHist, V_THRESH, MIN_UPSTATE_DUR, MIN_DOWNSTATE_DUR)
