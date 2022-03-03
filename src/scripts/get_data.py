from influxdb import InfluxDBClient
import pandas as pd
import datetime
import os
from termcolor import colored
LOGCL = os.environ['logcl']


def get_df(results):

    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns)
    df["time"] = pd.to_datetime(df["time"])
    return df

def get_client():

  client = InfluxDBClient(host=os.environ['host'],
        port=int(os.environ['port']),
        username=os.environ['username'],
        password=os.environ['password']
    )
  client.switch_database(os.environ['dbname'])

  return client


def query_data():
  
  now = datetime.datetime.now()
  print(colored("[Log start - " + now.strftime("%Y-%m-%d %H:%M:%S") + "]\n", 'red', attrs=['bold']))

  print(colored("[Initializing connection]", LOGCL, attrs=['bold']))
  try:
    client = get_client() 
  except Exception:
    print("> Failed to connect to the remote DB. Using the latest saved data.\n")
    path = os.environ['path']
    power, weather = pd.read_csv(f"{path}/data/old/power.csv"), pd.read_csv(f"{path}/data/old/weather.csv")
    return power, weather
  print("> Done!\n")

  print(colored("[Getting the last 90 days of power generation data]", LOGCL, attrs=['bold']))
  generation = client.query(
      "SELECT * FROM Generation where time > now()-90d"
      )
  print("> Done!\n")

  print(colored("[Getting the last 90 days of weather forecasts with the shortest lead time]",LOGCL, attrs=['bold']))
  wind  = client.query(
      "SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'"
      )
  print("> Done!\n")
  
  print(colored("[Saving the fetched data]", LOGCL, attrs=['bold']))
  power, weather = get_df(generation), get_df(wind)
  path = os.environ['path']
  power.to_csv(f"{path}/data/old/power.csv", index=False)
  weather.to_csv(f"{path}/data/old/weather.csv", index=False)
  print("> Done!\n")

  return power, weather

def query_future_weather():

  print(colored("[Initializing DB connection]", LOGCL, attrs=['bold']))
  try:
    client = get_client() 
  except Exception:
    print("> Failed to connect to the remote DB. Using the latest saved data.\n")
    path = os.environ['path']
    weather = pd.read_csv(f"{path}/data/future/weather.csv")
    return weather
  print("> Done!\n")
  
  print(colored("[Fetching the latest weather forecast.]", LOGCL, attrs=['bold']))
  forecasts  = client.query(
      "SELECT * FROM MetForecasts where time > now()"
      ) 
  for_df = get_df(forecasts)

  # Limit to only the newest source time
  newest_source_time = for_df["Source_time"].max()
  weather = for_df.loc[for_df["Source_time"] == newest_source_time].copy()
  print("> Done!\n")
  
  print(colored("[Saving the fetched data]", LOGCL, attrs=['bold']))
  path = os.environ['path']
  weather.to_csv(f"{path}/data/future/weather.csv", index=False)
  print("> Done!\n")

  return weather

