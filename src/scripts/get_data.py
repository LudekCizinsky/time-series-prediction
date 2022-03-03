from influxdb import InfluxDBClient
import pandas as pd
import datetime
import os


def get_df(results):

    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns)
    df["time"] = pd.to_datetime(df["time"])
    return df


def query_data():

  now = datetime.datetime.now()
  print ("[Log start - " + now.strftime("%Y-%m-%d %H:%M:%S") + "]\n")
  
  print("[Initializing connection]")
  try:
    client = InfluxDBClient(host=os.environ['host'],
        port=int(os.environ['port']),
        username=os.environ['username'],
        password=os.environ['password']
    )
    client.switch_database(os.environ['dbname'])
  except Exception:
    print("> Failed to connect to the remote DB. Using the latest saved data.")
    path = os.environ['path']
    power, weather = pd.read_csv(f"{path}/data/old/power.csv"), pd.read_csv(f"{path}/data/old/weather.csv")
    return power, weather
  print("> Done!\n")

  print("[Getting the last 90 days of power generation data]")
  generation = client.query(
      "SELECT * FROM Generation where time > now()-90d"
      )
  print("> Done!\n")

  print("[Getting the last 90 days of weather forecasts with the shortest lead time]")
  wind  = client.query(
      "SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'"
      )
  print("> Done!\n")
  
  print("[Saving the fetched data]")
  power, weather = get_df(generation), get_df(wind)
  path = os.environ['path']
  power.to_csv(f"{path}/data/old/power.csv", index=False)
  weather.to_csv(f"{path}/data/old/weather.csv", index=False)
  print("> Done!\n")

  return power, weather

