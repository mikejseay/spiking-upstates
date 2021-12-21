""" statistical functions.
they support analysis and plotting.
"""

import numpy as np
from sklearn import linear_model
import statsmodels.api as sm


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def outlying(a, sd_thresh=5):
    a_mean = np.nanmean(a)
    a_std = np.nanstd(a)
    a_outlying = np.logical_or(a > a_mean + sd_thresh * a_std, a < a_mean - sd_thresh * a_std)
    return a_outlying


def remove_outliers(a, sd_thresh=5):
    a_mean = np.nanmean(a)
    a_std = np.nanstd(a)
    a_outlying = np.logical_or(a > a_mean + sd_thresh * a_std, a < a_mean - sd_thresh * a_std)
    a_clean = a[~a_outlying]
    return a_clean


def regress_linear(xvals, yvals):
    ''' given equal length vectors, do a linear regression.
    outputs are pred_x, pred_y, b, r2, p'''

    if len(xvals.shape) < 2:
        xvals = xvals.reshape(-1, 1)
    if len(yvals.shape) < 2:
        yvals = yvals.reshape(-1, 1)

    both_notnan = ~np.isnan(xvals) & ~np.isnan(yvals)
    xvals = xvals[both_notnan].reshape(-1, 1)
    yvals = yvals[both_notnan].reshape(-1, 1)

    regr = linear_model.LinearRegression()
    regr.fit(xvals, yvals)

    min_x = xvals.min()
    max_x = xvals.max()
    pred_x = np.array([min_x, max_x]).reshape(-1, 1)
    pred_y = regr.predict(pred_x)

    X2 = sm.add_constant(xvals)
    est = sm.OLS(yvals, X2)
    fii = est.fit()
    p = fii.f_pvalue

    return pred_x, pred_y, regr.coef_[0][0], regr.score(xvals, yvals), p