import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description='data cleaner script')
parser.add_argument("-o", required=True, type=str)
parser.add_argument("-fn", required=True, type=str)
parser.add_argument("-d", required=True, type=str)
args = parser.parse_args()

FILENAME = os.path.join(args.d, args.fn)
FILENAME = open(FILENAME, encoding="utf-8")
dfs = []
for line in FILENAME:
	line = line.strip("\n")
	line = os.path.join(args.d, line)
	try:
		curr_df = pd.read_csv(line, sep="\t")
	except:
		continue
	curr_df.dropna(inplace=True)
	dfs.append(curr_df)
FILENAME.close()

df = pd.concat(dfs, ignore_index=True)
df.drop_duplicates(inplace=True, ignore_index=True)
df.to_csv(args.o, sep="\t", index=False)
