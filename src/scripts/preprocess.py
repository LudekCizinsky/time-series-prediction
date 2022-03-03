import pandas as pd
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
 
  print(colored("[One hot encoding relevant columns]", LOGCL, attrs=['bold']))
  df = pd.get_dummies(df, columns=["Direction"])
  print(f"> Successfully one hot encoded direction column.\n")

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

