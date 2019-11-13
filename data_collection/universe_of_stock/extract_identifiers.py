import json
import csv
import pandas as pd 
import sys

INPUT_PATH = './stock_identifiers.csv'
GVKEY_PATH = './gvkey.txt'
PERMNO_PATH = './permno.txt'
PERMCO_PATH = './permco.txt'
CUSIP_PATH = './cusip.txt'

def open_input_file():
	global INPUT_PATH
	file = pd.read_csv(INPUT_PATH) 
	return file

def generate_gvkey_file():
	global GVKEY_PATH
	input_file = open_input_file()
	data = input_file['GVKEY'].tolist()
	with open(GVKEY_PATH, 'w') as file:
		for a in data:
			file.write(str(a)+'\n')

def generate_permno_file():
	global PERMNO_PATH
	input_file = open_input_file()
	data = input_file['PERMNO'].tolist()
	with open(PERMNO_PATH, 'w') as file:
		for a in data:
			file.write(str(a)+'\n')

def generate_permco_file():
	global PERMNO_PATH
	input_file = open_input_file()
	data = input_file['PERMCO'].tolist()
	with open(PERMCO_PATH, 'w') as file:
		for a in data:
			file.write(str(a)+'\n')

def generate_cusip_file():
	global CUSIP_PATH
	input_file = open_input_file()
	data = input_file['cusip'].tolist()
	with open(CUSIP_PATH, 'w') as file:
		for a in data:
			file.write(str(a)+'\n')


 

if __name__ == '__main__':
	if sys.argv[1] == "gvkey":
		generate_gvkey_file()
	elif sys.argv[1] == "permno":
		generate_permno_file()
	elif sys.argv[1] == "permco":
		generate_permco_file()
	elif sys.argv[1] == "cusip":
		generate_cusip_file()
	else:
		pass