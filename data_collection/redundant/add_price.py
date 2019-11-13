import json
import csv
import pandas as pd 
import sys
import numpy as np
import math


# gvkey and permno relationship
file = pd.read_csv('./stock_identifiers.csv')
print(file.head())

# build a dictionary
gvkey_to_permno = {}
for index, row in file.iterrows():
	gvkey_to_permno[row["GVKEY"]] = row["PERMNO"]

# get price
prices = pd.read_csv('./price.csv')
print(prices.head())

# open file
output = pd.read_csv('./market_data_3.csv')
price_info = []
count = 0

for index, row in output.iterrows():
	count += 1
	# print(count)
	permno = gvkey_to_permno[row["gvkey"]]
	print(row["gvkey"], permno)
	date = str(row["date"])
	new_date = date[:4]+"-"+date[4:6]+"-"+date[6:]
	result = prices.loc[prices["date"] == new_date][str(permno)].tolist()
	price_info.append(result[0]) if len(result) > 0 else price_info.append(-1)


print(len(price_info))
output["prices"] = price_info
output.to_csv("./final_files/market_data_3.csv", index=False)