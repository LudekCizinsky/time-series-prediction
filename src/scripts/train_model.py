from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
from termcolor import colored
import os
LOGCL = os.environ['logcl']


FEATURES = ['Speed', 'Wx', 'Wy']

def get_future_weather_X(df):
  X = df.loc[:, df.columns.isin(FEATURES)].to_numpy()
  return X

def train_lr(X_train, y_train):

  print(colored("[Training Linear regression model with polynomial features]", LOGCL, attrs=['bold']))
  pipe = make_pipeline(
      StandardScaler(),
      PolynomialFeatures(degree=3, include_bias=False),
      LinearRegression()
      )
  pipe.fit(X_train, y_train)
  print("> Training successfully done.\n")

  return pipe

def train_dtr(X_train, y_train):

  print(colored("[Training Decision tree model.]", LOGCL, attrs=['bold']))
  pipe = make_pipeline(
      StandardScaler(),
      DecisionTreeRegressor(random_state=42)
      )
  pipe.fit(X_train, y_train)
  print("> Training successfully done.\n")

  return pipe

def compare_models(mds, X_dev, y_dev):

  print(colored("[Comparing models on dev set.]\n", LOGCL, attrs=['bold']))
  res = []
  for name, md in mds.items():
    print(f"> Name: {name}")
    y_hat_dev = md.predict(X_dev)
    mse = mean_squared_error(y_dev, y_hat_dev)
    r2 = r2_score(y_dev, y_hat_dev)
    print(f"> MSE: {mse}")
    print(f"> R2: {r2}\n")
    res.append((name, mse,))
  name, _ = sorted(res, key=lambda x: x[1])[0]
  
  return name, mds[name]


def do_train_test_split(df):

  print(colored("[Splitting data into train, dev, test sets]", LOGCL, attrs=['bold']))
  target = "total_mean"
  y = df[target].to_numpy()
  X = df.loc[:, df.columns.isin(FEATURES)].to_numpy()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
  X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
  print("> Done!\n")

  return X_train, X_dev, X_test, y_train, y_dev, y_test
   

def get_new_model(df):
  
  # --------- Train, dev, test split
  X_train, X_dev, X_test, y_train, y_dev, y_test = do_train_test_split(df)

  # --------- Training of models
  mds = dict()
  mds["Linear regression"] = train_lr(X_train, y_train)
  mds["Decision tree"] = train_dtr(X_train, y_train)

  # --------- Comparison and evaluation of models
  bestmdname, bestmd = compare_models(mds, X_dev, y_dev)

  return bestmd, X_test, y_test


