import json
import csv
import pandas as pd 
import sys
import numpy as np
import math


# gvkey and permno relationship
file = pd.read_csv('./universe_of_stock/stock_identifiers.csv')

permno_to_gvkey = {}
permno_to_gvkey["date"] = "date"
for index, row in file.iterrows():
	permno_to_gvkey[row["PERMNO"]] = row["GVKEY"]

# get price
prices = pd.read_csv('./price_permno.csv')
count = 0
new_names = []
for name in prices.columns:
	count += 1
	if(count == 1):
		new_names.append("date")
	else:
		new_names.append(permno_to_gvkey[int(name)])

print(prices.head())
prices.columns = new_names
print(prices.head())
prices.to_csv("./price_gvkey.csv", index=False)
