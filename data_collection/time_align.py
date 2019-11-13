import json
import csv
import pandas as pd 
import sys
import numpy as np
import math
import datetime


SP_PATH = './s&p.csv'
DIVIDEND_PATH = './ratios_dividend.csv'
PRICE_PATH = './price_gvkey.csv'

def open_file(INPUT_PATH):
	file = pd.read_csv(INPUT_PATH)
	print(file.head(3))
	print(file.tail(3))
	print("\n --- done: reading %s --- \n" % INPUT_PATH)
	return file

def find_common_set(set1, set2):
	comp_list = set1["gvkey"].tolist()
	comp_set = set(comp_list)

	comp_list = set2["gvkey"].tolist()
	comp_set2 = set(comp_list)

	common_set = comp_set2.intersection(comp_set)
	return common_set

def stock_ordered_date_list(sp, ratio_dividend, common_set, IGNORE_SP):
	_dict = {}
	for gvkey in common_set:
		sp_date_set = set(sp.loc[sp['gvkey'] == gvkey]["rdate"].tolist())
		ratio_date_set = set(ratio_dividend.loc[ratio_dividend['gvkey'] == gvkey]["public_date"].tolist())
		date_list = sorted(list(ratio_date_set)) if IGNORE_SP else sorted(list(sp_date_set.union(ratio_date_set)))
		_dict[gvkey] = date_list
	return _dict

# if we want a int, but the list has nothing, we output:0
# if we want a str, but the list has nothing, we output:""
def customize_data(ouput_a_number, input_list):
	if(len(input_list) == 0):
		if(ouput_a_number==1):
			return 0
		else:
			return ""
	else:
		if(ouput_a_number==1 and math.isnan(float(input_list[0])) ):
			return 0
		return input_list[0]

def get_column_names():
	return ["gvkey", 
			"date",
			"ms_rating",
			"ms_sentiment", 
			"sp_rating_flag",
			"st_rating", 
			"lt_rating", 
			"ratios_flag",
			"dividend_yield", 
			"price_to_book", 
			"current_ratio",
			"beta",  
			"price_to_sale" , 
			"earnings_per_share",
			"debt_to_equity",
			"revenue_growth", 
			"price_to_earning",
			"up_down_to_bear_case",
			"up_down_to_price_target",  
			"pe_1y_forward", 
			"pe_lt_forward", 
			"price_to_cashflow", 
			"return_on_asset", 
			"return_on_equity",
			"price",
			"price_next_month",
			"price_change_month",
			"price_next_record",
			"price_change_record"]

def format_date(date, add_month = False):
	date = str(date)
	date_time_obj = datetime.datetime.strptime(date, '%Y%m%d')
	if add_month:
		date_time_obj = date_time_obj + datetime.timedelta(days=+30)
	return date_time_obj.strftime("%Y-%m-%d")

def find_neareset_price(prices, date, gvkey):
	for add_day in range(10):
		price = prices.loc[prices["date"] == date]
		if(len(price[str(gvkey)].tolist())!=0):
			return price
		else:
			date_time_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
			date_time_obj = date_time_obj + datetime.timedelta(days=+1)
			date = date_time_obj.strftime("%Y-%m-%d")
	return prices.loc[prices["date"] == date]


def format_output_data(stock_date_dict, sp, ratio_dividend, prices, common_set, IGNORE_SP, _flag):
	output = []
	no_price_list = []
	count = 0
	for gvkey, date_list in stock_date_dict.items():

		# keep last record
		is_new_rating = False ; last_sp_st_rating = "" ; last_sp_lt_rating = ""

		# appending information in row
		for idx, date in enumerate(date_list):
			new_row = [gvkey, date, 0, 0]



			# s&p_rating
			if IGNORE_SP:
				new_row = [gvkey, date, 0, 0, 0, "", ""]
			else:
				# sp_rating_flag
				record = sp.loc[sp["gvkey"] == gvkey].loc[sp["rdate"] == date]
				if(record.shape[0] == 0):
					new_row.append(0) ; is_new_rating = False
				else:
					new_row.append(1) ; is_new_rating = True

				if(is_new_rating):
					last_sp_st_rating = customize_data(0,record.loc[record["strf"] == 1]["rating"].tolist())
					last_sp_lt_rating = customize_data(0,record.loc[record["ltrf"] == 1]["rating"].tolist())
				new_row.append(last_sp_st_rating) ; new_row.append(last_sp_lt_rating)



			# ratios and dividend
			record = ratio_dividend.loc[ratio_dividend["gvkey"] == gvkey].loc[ratio_dividend["public_date"] == date]
			new_ratio_flag = 0 if record.shape[0] == 0 else 1
			new_row.append(new_ratio_flag)

			if(new_ratio_flag or len(output)==0):
				# dividend_yield
				if(len(record["DIVYIELD"].tolist()) != 0):
					data_point = record["DIVYIELD"].tolist()[0]
					if(type(data_point)==type("")):
						data_point = data_point[:data_point.find("%")]
					temp_list = [data_point]
					new_row.append(customize_data(1, temp_list))
				else:	
					new_row.append(customize_data(1, record["DIVYIELD"].tolist())) 

				new_row.append(customize_data(1, record["ptb"].tolist())) # price_to_book
				new_row.append(customize_data(1, record["curr_ratio"].tolist())) # current_ratio
				new_row.append(0) # new_row.append(customize_data(1, record["DIVYIELD"].tolist())) # beta
				new_row.append(customize_data(1, record["ps"].tolist())) # price_to_sale
				new_row.append(0) # new_row.append(customize_data(1, record["pes"].tolist())) # earnings_per_share
				new_row.append(customize_data(1, record["de_ratio"].tolist())) # debt_to_equity
				new_row.append(0) # new_row.append(customize_data(1, record["DIVYIELD"].tolist())) # revenue_growth
				new_row.append(customize_data(1, record["pe_exi"].tolist())) # price_to_earning
				new_row.append(0) # new_row.append(customize_data(1, record["DIVYIELD"].tolist())) # up_down_to_bear_case
				new_row.append(0) # new_row.append(customize_data(1, record["DIVYIELD"].tolist())) # up_down_to_price_target
				new_row.append(customize_data(1, record["PEG_1yrforward"].tolist())) # pe_1y_forward
				new_row.append(customize_data(1, record["PEG_ltgforward"].tolist())) # pe_lt_forward
				new_row.append(customize_data(1, record["pcf"].tolist())) # price_to_cashflow
				new_row.append(customize_data(1, record["roa"].tolist())) # return_on_asset
				new_row.append(customize_data(1, record["roe"].tolist())) # return_on_equity
			else:
				# fill in previous record
				last_record = output[-1]
				new_row.extend(last_record[8:24])



			# prices
			price = find_neareset_price(prices, format_date(date), gvkey)
			new_row.append(customize_data(1, price[str(gvkey)].tolist())) # "price", "price_change_month", "price_change"
			
			price_next_month = find_neareset_price(prices, format_date(date, add_month = True), gvkey)
			new_row.append(customize_data(1, price_next_month[str(gvkey)].tolist()))
			new_data = 0 if new_row[-2]==0 else (new_row[-1]-new_row[-2])/new_row[-2]
			new_row.append(new_data)
			
			if(idx == len(date_list)-1):
				price_next_record = find_neareset_price(prices, format_date(date, add_month = True), gvkey)
				new_row.append(customize_data(1, price_next_record[str(gvkey)].tolist()))
			else:
				price_next_record = find_neareset_price(prices, format_date(date_list[idx+1]), gvkey)
				new_row.append(customize_data(1, price_next_record[str(gvkey)].tolist()))
			new_data = 0 if new_row[-4]==0 else (new_row[-1]-new_row[-4])/new_row[-4]
			new_row.append(new_data)
			output.append(new_row)
			if(output[-1][-5]==0):
				no_price_list.append(gvkey)
		count += 1
		# break
		# if(count==3):
		# 	break
		if(count%int(len(stock_date_dict.keys())/10)==0):
			print(count/len(stock_date_dict.keys()))
	print(set(no_price_list))
	return output




sp = open_file(SP_PATH)
ratio_dividend = open_file(DIVIDEND_PATH)
prices = open_file(PRICE_PATH)
# exit(0)

IGNORE_SP = False
common_set = find_common_set(ratio_dividend, ratio_dividend) if IGNORE_SP else find_common_set(sp, ratio_dividend)
stock_date_dict = stock_ordered_date_list(sp, ratio_dividend, common_set, IGNORE_SP)
# exit(0)

output = format_output_data(stock_date_dict, sp, ratio_dividend, prices, common_set, IGNORE_SP, False)
column_names = get_column_names()
# exit(0)

output_df = pd.DataFrame(np.array(output),columns = column_names)
output_df.to_csv("./output/market_data_183.csv",index=False)

