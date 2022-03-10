#!/usr/bin/env python3
# -*- coding: utf-8 -*-

LOGCL = 'blue'
from termcolor import colored
import os
import datetime
os.environ['logcl'] = LOGCL


print(colored('[Loading dependencies.]', 'green', attrs=['bold']))
import sys
from tqdm import tqdm
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path)
from scripts import query_data, preprocess, get_new_model, query_future_weather, predict_future, eda
from config import HOST, PORT, USERNAME, PASSWORD, DBNAME 
print('> Done!\n')

print(colored('[Loading environment variables.]', 'green', attrs=['bold']))
os.environ['host'] = HOST
os.environ['port'] = str(PORT)
os.environ['username'] = USERNAME
os.environ['password'] = PASSWORD
os.environ['dbname'] = DBNAME
os.environ['path'] = path
print('> Done!\n')


def main():
  
  se = "-"*10
  sect = "-"*5

  now = datetime.datetime.now() 
  print(colored(f"{se} Log start - {now.strftime('%Y-%m-%d %H:%M:%S')}\n", 'red', attrs=['bold']))
 
  print(colored(f"{sect} Get data from DB", 'magenta', attrs=['bold']))
  power, weather = query_data()
 
  print(colored(f"{sect} Preprocessing data", 'magenta', attrs=['bold']))
  df = preprocess(dfs=[power, weather])

  print(colored(f"{sect} EDA", 'magenta', attrs=['bold']))
  eda(df)

  print(colored(f"{sect} Train the new model and get test data", 'magenta', attrs=['bold']))
  nm, X_test, y_test = get_new_model(df)

  print(colored(f"{sect} Compare to old model and predict the future", 'magenta', attrs=['bold']))
  predict_future(nm, X_test, y_test)

  now = datetime.datetime.now()
  print(colored(f"{se} Log end - {now.strftime('%Y-%m-%d %H:%M:%S')}\n", 'red', attrs=['bold']))
  
if __name__ == "__main__":
  main()

