#!/usr/bin/env python3
# -*- coding: utf-8 -*-

LOGCL = 'blue'
from termcolor import colored
import os
os.environ['logcl'] = LOGCL


print(colored('[Loading dependencies.]', LOGCL, attrs=['bold']))
import sys
from tqdm import tqdm
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path)
from scripts import query_data, preprocess, get_new_model, query_future_weather, predict_future
from config import HOST, PORT, USERNAME, PASSWORD, DBNAME 
print('> Done!\n')

print(colored('[Loading environment variables.]', LOGCL, attrs=['bold']))
os.environ['host'] = HOST
os.environ['port'] = str(PORT)
os.environ['username'] = USERNAME
os.environ['password'] = PASSWORD
os.environ['dbname'] = DBNAME
os.environ['path'] = path
print('> Done!\n')


def main():
  
  # ----------- Get data from DB
  power, weather = query_data()
 
  # ----------- Preprocessing data
  df = preprocess(dfs=[power, weather])

  # ----------- Train the new model and get test data
  nm, X_test, y_test = get_new_model(df)

  # ----------- Compare to old model and predict the future
  predict_future(nm, X_test, y_test)

if __name__ == "__main__":
  main()

