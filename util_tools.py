import h5py
import pandas as pd

def hdf5_to_dataframe(h5_path, identifier='gvkey'):
    with h5py.File(h5_path, 'r') as h:
        data = h['data'][...]
        identifiers = h[identifier][...]
        date = pd.to_datetime([str(x) for x in h['date'][...]])

    return pd.DataFrame(data, pd.Index(date, name='date'),
                        pd.Index(identifiers, name=identifier))

def save_to_hdf5(dataframe, dataset_name, path='.', identifier='gvkey'):
    df = dataframe.sort_index(axis=0).sort_index(axis=1)
    with h5py.File(f'{path}/{dataset_name}.h5', 'w') as h:
        h.create_dataset('data', dtype='f', data=df.values)
        h.create_dataset(identifier, dtype='i', data=df.columns.get_level_values(identifier).values.astype(int))
        h.create_dataset('date', dtype='i', data=[int(pd.Timestamp(x).strftime('%Y%m%d')) for x in df.index.values])
