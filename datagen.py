import argparse
import requests
import time
import os
from tqdm import tqdm
from multiprocessing import Pool

#Parsing arguments
parser = argparse.ArgumentParser(description="generate and locally store data from https://seeclickfix.com for text classification")

parser.add_argument("-d", default="SCF_data", type=str, help="directory for downloading stored data")
parser.add_argument("--start_id", default=1, type=int, help="lower value of place id")
parser.add_argument("--end_id", default=20, type=int, help="upper value of place id")
parser.add_argument("-o", default="output", type=str, help="output filename, data will be stored as part files")
args = parser.parse_args()

#Try to create the output directory, if already present the existing one will be used
try:
    os.makedirs(args.d)
except FileExistsError:
    print("Directory " , args.d, " already exists")
   
def collect(x):
    start, end, filename = x 
   
    #Open the file to write in 
    f = open(filename, "w", encoding='utf-8')
    f.write('sentence\tlabel\n')

   
    #Now collect and put the data in the file
    for i in tqdm(range(start, end+1)):
        
        #First fetch the name of the place by place id
        url = "https://seeclickfix.com/api/v2/places/{}".format(i)
        resp = requests.get(url=url)
        if(bool(resp.json())):
            #Get the coordinates
            delay = 3*os.cpu_count()
		    time.sleep(delay)
            try:
                coord = resp.json()['center']['coordinates']
            except KeyError:
                print(f"response code {resp}")
                continue
            url = "https://seeclickfix.com/open311/v2/requests.json?lat={}&long={}".format(coord[1], coord[0])
            current = requests.get(url).json()
            if(bool(current)):
                #Get the data
                for each in current:
                    desc = each['description']
                    issue = each['service_name']
                    if(desc!=None and issue!=None):
                        line = desc.strip()+"\t"+issue.strip()+"\n"
                        f.write(line)
    f.close()

#use as many processes as the number of cpus
n_procs = os.cpu_count()

len_chunk = (args.end_id - args.start_id)//n_procs

job_args = []
for i in range(n_procs):
    if(args.start_id + (i+1)*len_chunk < args.end_id):
        tup = (args.start_id + i*len_chunk, args.start_id + (i+1)*len_chunk, os.path.join(args.d, args.o+f"-part{i}.tsv"))
    else:
        tup = (args.start_id + i*len_chunk, args.end_id, os.path.join(args.d, args.o+f"-part{i}.tsv"))

    job_args.append(tup)

p = Pool(n_procs)
p.map(collect, job_args)
p.close()
