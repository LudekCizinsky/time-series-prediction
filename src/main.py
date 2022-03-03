#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print('[Loading dependencies.]')
import sys
from tqdm import tqdm
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path)
from scripts import query_data, preprocess, get_new_model
from config import HOST, PORT, USERNAME, PASSWORD, DBNAME 
print('> Done!')

print('[Loading environment variables.]')
os.environ['host'] = HOST
os.environ['port'] = str(PORT)
os.environ['username'] = USERNAME
os.environ['password'] = PASSWORD
os.environ['dbname'] = DBNAME
os.environ['path'] = path
print('> Done!')


def main():
  
  # ----------- Get data from DB
  power, weather = query_data()
 
  # ----------- Preprocessing data
  df = preprocess(dfs=[power, weather])

  # ----------- Train the new model
  nm = get_new_model(df)

  # ----------- Compare to old model
  

if __name__ == "__main__":
  main()
