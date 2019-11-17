import pandas as pd
import numpy as np
import configs
from util_tools import hdf5_to_dataframe, save_to_hdf5
import re

data_dict = {'adj_close': hdf5_to_dataframe(configs.hdf5_path + '/adj_close_gvkey.h5'),
             'adj_open': hdf5_to_dataframe(configs.hdf5_path + '/adj_open_gvkey.h5'),
             'abnormal_price': hdf5_to_dataframe(configs.hdf5_path + '/abnormal_price_gvkey.h5')}

trade_dates = data_dict['adj_close'].index
holidays = [x for x in pd.bdate_range(trade_dates[0], trade_dates[-1]).to_pydatetime() if not x in trade_dates]
CBD = pd.tseries.offsets.CustomBusinessDay(holidays=holidays)

#
def next_period_return(startdate=20090102, enddate=20190628, freq='1M', use_open=True, max_missing_pct=0.2):
    """
    A function to calculate subsequent holding period return (without considering dividends received)
    :param startdate: int, the loaded market data starts from this date. Example format: 20090102
    :param enddate: int, the loaded market data ends at this date. Example format: 20190628
    :param freq: str, the length of holding period. 1M means one month, 8W means eight weeks
    :param use_open: bool, whether to use open price for calculating holding period return. If True, the holding period return is calculated as "last available closing price / first available opening price"; otherwise, it is calculated "last available closing price / first available closing price"  
    :param max_missing_pct: float, between 0.0 and 1.0. The maximal tolerable percent of missing price data in a holding period. Returns for stocks that exeed max_missing_pct in a given holding period will be set to NAN. Missing data can be caused by two situations: 1) the stock is delisted. 2) the prices are "abnormal". Prices from CRSP that satisfy any of the following conditions are defined as abnormal: 1) daily opening price or closing price is missing; 2) closing price is negative, which means no trades at all; 3) daily trading volume is zero or negative
    :return: A dataframe. Values: subsequent (future) holding period return; Columns: crsp permno; Index: date. Below is a sample output (startdate=20090102, enddate=20190628, freq='1M', use_open=True, max_abnormal_pct=0.2):
        permno         10001     10044     10107     10138  ...  93419  93422  93423  93429
        date                                                ...
        2009-01-30  0.079710 -0.089921 -0.051674 -0.163602  ...    NaN    NaN    NaN    NaN
        2009-02-27  0.003681  0.097321  0.151003  0.279823  ...    NaN    NaN    NaN    NaN
        2009-03-31  0.062500 -0.035750  0.111355  0.374242  ...    NaN    NaN    NaN    NaN
        2009-04-30 -0.024166  0.338726  0.034671  0.063992  ...    NaN    NaN    NaN    NaN
        2009-05-29  0.011753 -0.042289  0.131905  0.009692  ...    NaN    NaN    NaN    NaN
    the first element of this dataframe means that at the end of Jan 2009, the security 10001 will have a holding period return in the following month that equals to 0.079710. Thus we can use financial or other types of factors available at 2009-01-30 as features to predict holding period return of the subsequent month.
    """

    freq = freq.upper().strip()
    if not re.search(r'\d*[MWD]', freq):
        raise Exception(f'invalid freq: {freq}')
    step = int(freq[:-1]) if len(freq) > 1 else 1

    startdate = pd.to_datetime(str(startdate))
    enddate = shift_enddate(enddate, freq)
    adjclose = data_dict['adj_close'].loc[startdate: enddate].copy()
    abnormal_price = data_dict['abnormal_price'].copy()
    adjclose[abnormal_price==1] = np.nan
    adjclose.ffill(inplace=True)
    # todo: align_index
    dates = adjclose.index.tolist()

    if use_open:
        adjopen = data_dict['adj_open'].loc[startdate: enddate].copy()
        adjopen[abnormal_price==1] = np.nan
        adjopen.bfill(inplace=True)
    else:
        adjclose2 = data_dict['adj_close'].loc[startdate: enddate].copy()
        adjclose2[abnormal_price == 1] = np.nan
        adjclose2.bfill(inplace=True)

    resample_adjclose = adjclose.resample(freq[-1]).last().dropna(axis=0, how='all')
    resample_normal_count = (abnormal_price==0).astype(float).resample(freq[-1]).sum().reindex(index=resample_adjclose.index).rolling(step).sum()
    resample_total_count = abnormal_price.resample(freq[-1]).count().max(axis=1).reindex(index=resample_adjclose.index).rolling(
        step).sum()
    resample_adjclose = resample_adjclose.iloc[::step]
    resample_normal = (resample_normal_count.div(resample_total_count, axis=0) >= (1 - max_missing_pct)).reindex(index=resample_adjclose.index).shift(-1)

    if use_open:
        resample_adjopen = adjopen.resample(freq[-1]).first().reindex(index=resample_adjclose.index)
        next_period_ret = (resample_adjclose / resample_adjopen - 1).shift(-1)
    else:
        resample_adjclose2 = adjclose2.resample(freq[-1]).first().reindex(index=resample_adjclose.index)
        next_period_ret = (resample_adjclose / resample_adjclose2 - 1).shift(-1)

    next_period_ret[resample_normal == False] = np.nan
    next_period_ret = next_period_ret.iloc[:-1].dropna(axis=1, how='all').dropna(axis=0, how='all')
    next_period_ret = align_datetime(next_period_ret, freq)

    return next_period_ret


def shift_enddate(enddate, freq='M'):
    enddate = pd.to_datetime(str(enddate))
    if 'M' in freq:
        if enddate.day < 18:
            offset = pd.tseries.offsets.MonthEnd()
            return enddate - offset
    else:
        if enddate.isoweekday() < 3:
            offset = pd.tseries.offsets.Week(1, weekday=4)
            return enddate - offset
    return enddate


def align_datetime(df, freq):
    if 'W' in freq:
        return pd.DataFrame(df.values, index=df.index - CBD, columns=df.columns)
    else:
        return pd.DataFrame(df.values, index=df.index + pd.Timedelta(days=1) - CBD, columns=df.columns)



if __name__ == '__main__':
    next_period_return = next_period_return(startdate=20090102, enddate=20190628, freq='1M', use_open=True, max_missing_pct=0.2)
    next_period_return.index = next_period_return.index - pd.Timedelta(days=1) + pd.offsets.MonthEnd(1)
    save_to_hdf5(next_period_return, 'return_1M',  'D:\\Columbia\\2019Fall\\4742DLforFE&OR\\GroupProject\\wrds\\hdf5\\future_return')
