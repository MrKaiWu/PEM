## Generate output
Run time_align.py program, output file will be stored in output folder

## output folder
1. market_data_183.csv:<br/> contains 183 stocks that have both s&p rating and ratios / dividend records.<br/>
columns and explaination:<br/>

|       column name       |                                  interpretation                                  |
|:-----------------------:|:--------------------------------------------------------------------------------:|
|          gvkey          |                              unique stock identifier                             |
|           date          | the date new information of the stock arrive (e.g. s&p rating, dividend, ratios) |
|        ms_rating        |                                                                                  |
|       ms_sentiment      |                                                                                  |
|      sp_rating_flag     |                 whether this new information is about s&p rating                 |
|        st_rating        |                          short term rating of the stock                          |
|        lt_rating        |                           long term rating of the stock                          |
|       ratios_flag       |             whether this new information is about ratios and dividend            |
|      dividend_yield     |                                  dividend yield                                  |
|      price_to_book      |                                   price to book                                  |
|      current_ratio      |                                                                                  |
|           beta          |                                                                                  |
|      price_to_sale      |                                   price to sale                                  |
|    earnings_per_share   |                                                                                  |
|      debt_to_equity     |                                  debt to equity                                  |
|      revenue_growth     |                                                                                  |
|     price_to_earning    |                                 price to earning                                 |
|   up_down_to_bear_case  |                                                                                  |
| up_down_to_price_target |                                                                                  |
|      pe_1y_forward      |                1 year forward prediction on price to equity ratio                |
|      pe_lt_forward      |                   long term prediction on price to equity ratio                  |
|    price_to_cashflow    |                                price to cash flow                                |
|     return_on_asset     |                                  return on asset                                 |
|     return_on_equity    |                                 return on equity                                 |
|          price          |                                   current price                                  |
|     price_next_month    |                             stock price after 30 days                            |
|    price_change_month   |                             price changed in 30 days                             |
|    price_next_record    |                     price at next arrival of new information                     |
|   price_change_record   |                     price changed between information arrival                    |


2. no_price_list.txt: <br/>
contains gvkey of the stocks that no long has a price record at the end (probably no longer traded in the market), it is suggested to discard these stocks. <br/>
Also, bear in mind that some stocks does no has a price intially, but has a price at a later point, we can discard those records with a 0 price.

## universe_of_stock folder
define our study scope, mainly high dividend stocks in the market

## other files
1. price_permno.csv, price_gvkey.csv, convert_prices.py:<br/>
convert  permno-based price reference file to gvkey-based reference file
2. s&p.csv, ratios_dividend.csv:<br/>
s&p rating, various ratios data, dividend information