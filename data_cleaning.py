import pandas as pd
import numpy as np
import bottleneck as bn
from util_tools import hdf5_to_dataframe
from configs import hdf5_path
from functools import partial



def factor_outlierlimit(factor, n_extremum=5):
    x_m = factor.values
    median = bn.nanmedian(x_m, axis=1).reshape(-1, 1)
    Dmad = bn.nanmedian(abs(x_m - median), axis=1).reshape(-1, 1)
    upper = (median + n_extremum * Dmad)
    lower = (median - n_extremum * Dmad)
    with np.errstate(invalid='ignore'):
        res = np.clip(x_m, lower, upper)
    return pd.DataFrame(res, factor.index, factor.columns)

def factor_normalize(factor):
    x_m = factor.values
    mean = bn.nanmean(x_m, axis=1).reshape(-1, 1)
    std = bn.nanstd(x_m, axis=1, ddof=1).reshape(-1, 1)
    with np.errstate(invalid='ignore'):
        res = (x_m - mean) / std
    return pd.DataFrame(res, factor.index, factor.columns)

def align_df(df, benchmark):
    cols = sorted(set(df.columns).intersection(set(benchmark.columns)))
    idxs = sorted(set(df.index).intersection(set(benchmark.index)))
    return df.reindex(index=idxs, columns=cols), benchmark.reindex(index=idxs, columns=cols)

# 行业中性化
def neutralize_indu(factor, indu):
    factorvalue = factor.values
    induvalue = indu.values
    inducode = [x for x in np.unique(induvalue) if not np.isnan(x)]
    shape = factorvalue.shape
    neutralized_factor = np.full(shape, np.nan)
    for indcd in inducode:
        loc = induvalue == indcd
        a = np.full(shape, np.nan)
        a[loc] = factorvalue[loc]
        c = a - bn.nanmean(a, axis=1).reshape(-1, 1)
        neutralized_factor[loc] = c[loc]
    mark1 = np.isnan(neutralized_factor)
    mark2 = ~np.isnan(factorvalue)
    neutralized_factor[mark1 & mark2] = 0
    neutralized_factor = pd.DataFrame(neutralized_factor, factor.index, factor.columns)
    return neutralized_factor

def factor_fillna_by_ind_mean(factor, indu):
    pro_x = factor.values.copy()
    loc1 = np.isnan(pro_x)
    pro_indu = indu.values
    inducode = [x for x in np.unique(pro_indu) if not np.isnan(x)]

    for i in range(len(pro_x)):
        temp = pro_x[i]
        temp_group = pro_indu[i]
        if not np.isnan(temp).any():
            continue
        for g in inducode:
            nan_loc = np.argwhere(np.isnan(temp) & (temp_group == g))
            if len(nan_loc):
                mean = bn.nanmean(temp[temp_group == g])
                temp[nan_loc] = mean
    loc = loc1 & np.isnan(indu)
    pro_x[loc] = np.nan
    return pd.DataFrame(pro_x, factor.index, factor.columns)

def factor_fillna_to_max(factor, na_benchmark):
    x_m = factor.values.copy()
    row_max = bn.nanmax(x_m, axis=1).reshape(-1,1)
    max_mat = np.hstack([row_max,] * x_m.shape[1])
    loc = np.isnan(x_m) & ~np.isnan(na_benchmark.values)
    x_m[loc] = max_mat[loc]
    return pd.DataFrame(x_m, columns=factor.columns, index=factor.index)

#div_yield
def factor_fillna_to_min(factor, na_benchmark):
    x_m = factor.values.copy()
    row_min = bn.nanmin(x_m, axis=1).reshape(-1,1)
    min_mat = np.hstack([row_min,] * x_m.shape[1])
    loc = np.isnan(x_m) & ~np.isnan(na_benchmark.values)
    x_m[loc] = min_mat[loc]
    return pd.DataFrame(x_m, columns=factor.columns, index=factor.index)

def factor_fillna_ffill(factor, na_benchmark, period=None):
    loc1 = np.isnan(factor.values)
    x_m = factor.ffill(limit=period).values
    loc = loc1 & np.isnan(na_benchmark.values)
    x_m[loc] = np.nan
    return pd.DataFrame(x_m, columns=factor.columns, index=factor.index)


if __name__ == '__main__':

    procedure_dict = {'at_turn': ['outlier', 'missing_ffill:period=12', 'indu_neutral', 'normalize'],
     'beta': ['outlier', 'missing_ffill:period=6', 'normalize'],
     'curr_ratio': ['outlier', 'missing_max', 'indu_neutral', 'normalize', ],
     'debt_assets': ['outlier', 'missing_ffill:period=12', 'indu_neutral', 'normalize', ],
     'debt_at': ['outlier', 'missing_ffill:period=12', 'indu_neutral', 'normalize'],
     'divyield': ['outlier', 'missing_min', 'indu_neutral', 'normalize'],
     'ep': ['outlier', 'missing_ind', 'indu_neutral', 'normalize'],
     'eps_growth': ['outlier', 'missing_ind', 'indu_neutral', 'normalize'],
     'ps': ['outlier', 'missing_ind', 'indu_neutral', 'normalize'],
     'ptb': ['outlier', 'missing_ind', 'indu_neutral', 'normalize'],
     'quick_ratio': ['outlier', 'missing_max', 'indu_neutral', 'normalize'],
     'revenue_growth': ['outlier', 'missing_ind', 'indu_neutral', 'normalize'],
     'roa': ['outlier', 'missing_ffill:period=12', 'indu_neutral', 'normalize'],
     'roe': ['outlier', 'missing_ffill:period=12', 'indu_neutral', 'normalize'],
     'IBES_buypct': ['missing_ffill:period=1', 'normalize'],
     'IBES_buypct_diff1': ['missing_min', 'normalize'],
     'IBES_holdpct': ['missing_ffill:period=1','normalize'],
     'IBES_holdpct_diff1': ['missing_min', 'normalize'],
     'IBES_meanrec': ['missing_ffill:period=1','normalize'],
     'IBES_meanrec_diff1': ['missing_min', 'normalize'],
     'IBES_numdown': ['missing_ffill:period=1','normalize'],
     'IBES_numdown_diff1': ['missing_min', 'normalize'],
     'IBES_numrec': ['missing_ffill:period=1','normalize'],
     'IBES_numrec_diff1': ['missing_min', 'normalize'],
     'IBES_numup': ['missing_ffill:period=1','normalize'],
     'IBES_numup_diff1': ['missing_min', 'normalize'],
     'IBES_sellpct': ['missing_ffill:period=1','normalize'],
     'IBES_sellpct_diff1': ['missing_min', 'normalize'],
     'IBES_updown_target': ['outlier', 'missing_ffill:period=1', 'normalize'],
    }

    indu = hdf5_to_dataframe(hdf5_path + r'/GIC.h5')

    func_dict = {'missing_min': None,
                'missing_max': None,
    'missing_ind': None,
    'missing_ffill': None,
    'normalize': factor_normalize,
    'outlier': partial(factor_outlierlimit, n_extremum=5),
    'indu_neutral': None,
    }

    factor_df_dict = {}
    for factor, procedure in procedure_dict.items():
        factor_df = hdf5_to_dataframe(hdf5_path + f'/factors/{factor}.h5')
        factor_df, indu_aligned = align_df(factor_df, indu)
        func_dict['missing_min'] = partial(factor_fillna_to_min, na_benchmark=indu_aligned)
        func_dict['missing_max'] = partial(factor_fillna_to_max, na_benchmark=indu_aligned)
        func_dict['missing_ind'] = partial(factor_fillna_by_ind_mean, indu=indu_aligned)
        func_dict['indu_neutral'] = partial(neutralize_indu, indu=indu_aligned)
        for p in procedure:
            if p.startswith('missing_ffill'):
                sp = p.split('period=')
                period = None if len(sp) == 1 else int(sp[-1].strip())
                func_dict['missing_ffill'] = partial(factor_fillna_ffill, na_benchmark=indu_aligned, period=period)
                factor_df = func_dict['missing_ffill'](factor_df)
            else:
                factor_df = func_dict[p](factor_df)
        n0 = (factor_df.isna().sum(axis=1) / factor_df.shape[1]).describe()['50%']
        print("{}, {} {:.3f}".format(factor, factor_df.shape, n0))
        factor_df_dict[factor] = factor_df

    for i, (k, v) in enumerate(factor_df_dict.items()):
        if i == 0:
            identifier_set = set(v.columns)
            date_set = set(v.index)
        else:
            identifier_set.intersection_update(v.columns)
            date_set.intersection_update(v.index)

    for k in factor_df_dict.keys():
        factor_df_dict[k] = factor_df_dict[k].reindex(columns=sorted(identifier_set), index=sorted(date_set))

    future_return = hdf5_to_dataframe(hdf5_path + r'/future_return/return_1M.h5').reindex(columns=sorted(identifier_set), index=sorted(date_set))

    stack_data = np.stack([x.values for x in factor_df_dict.values()] + [future_return.values, ], axis=2)

    mindex = pd.MultiIndex.from_product([sorted(date_set), sorted(identifier_set)], names=['date', 'gvkey'])
    cols = list(factor_df_dict.keys()) + ['return_1M',]

    df_merge = pd.DataFrame(stack_data.reshape(-1, stack_data.shape[-1]), index=mindex, columns=cols)
    df_merge = df_merge.dropna(how='any', axis=0)

    print(df_merge.shape)
    print((df_merge.index.get_level_values(0).unique()[1:] - df_merge.index.get_level_values(0).unique()[:-1]).unique().tolist())

    import matplotlib.pyplot as plt
    df_merge.groupby('date')['return_1M'].count().plot()
    plt.show()


