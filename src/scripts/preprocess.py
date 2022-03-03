import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from termcolor import colored
import os
LOGCL = os.environ['logcl']


def aggregate_power(df, fr, to, intv):

  # Sample data
  print(colored(f"[Aggregating power data in the interval of {intv} minutes]", LOGCL, attrs=['bold']))
  start, end = fr, fr + timedelta(minutes=intv) 
  tm = []
  avg_power = []
  while start <= to:
     
    mask = df["time"].between(start, end, inclusive="left")
    s = df[mask]
    tm.append(start)
    avg_power.append(s["Total"].mean())
    start += timedelta(minutes=intv)
    end += timedelta(minutes=intv)
  
  # create a new df
  df = pd.DataFrame.from_dict({"time": tm, "total_mean": avg_power})
  n1 = df.shape[0]
  df.dropna(inplace=True)
  n2 = df.shape[0]
  print(f"> Successfully aggregated the power dataset into {df.shape[0]} rows. More details:")
  print(f">> Dropped {n1 - n2} rows with missing values.")
  print(">> This is because in the given intervals there were no power records.\n")
   
  return df


def pp_power(df):

  print(colored("[Preprocessing power data]", LOGCL, attrs=['bold']))

  print("## Drop rows with missing values")
  n1 = df.shape[0]
  df.dropna(inplace=True)
  n2 = df.shape[0]
  print(f"> Dropped {n1 - n2} rows with missing values.\n")

  print("## Drop irrelevant columns")
  df.drop(labels=["ANM","Non-ANM"], inplace=True, axis=1)
  print("> Dropped successfully ANM and Non-ANM columns.\n")  

  return df

def dir2deg(s):
  """Copied from:
  Https://codegolf.stackexchange.com/questions/54755/convert-a-point-of-the-compass-to-degree://codegolf.stackexchange.com/questions/54755/convert-a-point-of-the-compass-to-degrees
  """

  if 'W' in s:
      s = s.replace('N','n')
  a=(len(s)-2)/8
  if 'b' in s:
      a = 1/8 if len(s)==3 else 1/4
      return (1-a)*f(s[:-2])+a*dir2deg(s[-1])
  else:
      if len(s)==1:
          return 'NESWn'.find(s)*90
      else:
          return (dir2deg(s[0])+dir2deg(s[1:]))/2


def pp_weather(df):

  print(colored("[Preprocessing weather data]", LOGCL, attrs=['bold']))

  print("## Drop rows with missing values")
  n1 = df.shape[0]
  df.dropna()
  n2 = df.shape[0]
  print(f"> Dropped {n1 - n2} rows with missing values.\n")

  print("## Drop irrelevant columns")
  df.drop(labels=["Lead_hours", "Source_time"], inplace=True, axis=1)
  print("> Dropped successfully Lead hours and Source time columns.\n")  
 
  print("## Transforming direction to a vector feature")
  print("> Getting the needed info: dir2deg, deg2rad")
  wd = df.pop("Direction")
  wd_deg = wd.apply(dir2deg)
  wv = df["Speed"] 
  wd_rad = wd_deg*np.pi / 180
  print("> Calculating the wind x and y components.")
  df['Wx'] = wv*np.cos(wd_rad)
  df['Wy'] = wv*np.sin(wd_rad)
  print(f"> Done!\n")

  return df
  
 
def preprocess(dfs):

  # --- Preprocessing of given datasets
  power, weather = pp_power(dfs[0]), pp_weather(dfs[1])

  # ---- Aggregate power dataset
  fr, to = weather["time"].iloc[0], weather["time"].iloc[-1]
  power = aggregate_power(power, fr, to, intv=180)
 
  # --- Merging
  print(colored("[Merging power and dataset together using inner join]", LOGCL, attrs=['bold']))
  df = weather.merge(power, on="time", how="inner")
  n1, m1 = weather.shape
  n2, m2 = power.shape
  n3, m3 = df.shape
  print(
  f"""> Merged done successfully. Here is useful info:
  >> Weather df shape: {n1} x {m1}
  >> Power df shape:   {n2} x {m2} 
  >> Merged df shape:  {n3} x {m3}
  """
  )
  cols = '\n>> '.join(df.columns)
  print(f"> Here are columns in the merged data frame:\n>> {cols}")
  print()

  return df

