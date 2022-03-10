import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
LOGCL = os.environ['logcl']
from termcolor import colored


def time_vs_power_past(df):

  print(colored('[EDA: time vs power history]', LOGCL, attrs=['bold'])) 
  back = 350
  d = df.iloc[-back:]
  years = mdates.YearLocator()
  months =  mdates.MonthLocator()
  days = mdates.DayLocator()
  years_fmt = mdates.DateFormatter('%Y-%m')

  fig, ax = plt.subplots(figsize=(25,7))
  sns.lineplot(x="time", y="Total",
      ax=ax, data=d, marker = 'o',
      label='Power in MegaWatts',
      estimator='mean',
      ci='sd')
  ax.xaxis.set_major_locator(months)
  ax.xaxis.set_major_formatter(years_fmt)
  ax.xaxis.set_minor_locator(days)
  t = datetime.now().strftime("%Y-%m-%d")
  ax.set_title(f"Power production from past ~2 months\n[{t}]")
  ax.set_xlabel("Time [3 hours interval]")
  ax.set_ylabel("Mean of power produced in 3 hours [MW]")
 
  path = f"{os.environ['path']}/figures/history/timevsproduction_{t}.jpg"
  fig.savefig(path)
  print(f"> Figure saved to {path}\n")


def explore_rel(df):

  print(colored('[EDA: power vs dependent vars]', LOGCL, attrs=['bold'])) 

  fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,7))

  sns.scatterplot(x="Speed", y="Total", ax=ax[0], data=df, alpha=0.3)
  sns.scatterplot(x="Wx", y="Total", ax=ax[1], data=df, alpha=0.3)
  sns.scatterplot(x="Wy", y="Total", ax=ax[2], data=df, alpha=0.3)
 
  t = datetime.now().strftime("%Y-%m-%d")
  path = f"{os.environ['path']}/figures/history/productionvsdepvars_{t}.jpg"
  fig.savefig(path)
  print(f"> Figure saved to {path}\n")

def eda(df):

  time_vs_power_past(df)
  explore_rel(df)

