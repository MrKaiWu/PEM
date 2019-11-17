# HDF5 Data

### update 20191117

"xxx_gvkey.h5" files in this root folder uses gvkey as identifiers. All other files in the sub folders use gvkey, too.

Files in the factors folder contain feature data as the input to PEM

Files in the future_returns folder are generated using future_return.py.


### Format: 

Values: the variable in interest

Columns: CRSP PERMNO

Index: trading days


### Current data available:

close: closing price

open: openning price

adj_close: adjusted closing price

adj_open: adjusted openning price

abnormal_price: if 1, the corresponding price is abnormal; 0 otherwise.

Prices from CRSP that satisfy any of the following conditions are defined as abnormal: 1) daily opening price or closing price is missing; 2) closing price is negative, which means no trades at all; 3) daily trading volume is zero or negative
