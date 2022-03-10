from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
from termcolor import colored
import os
import warnings
warnings.filterwarnings('ignore')
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
  
  params = {
      "polynomialfeatures__degree": [i for i in range(1, 12, 2)]
  }
  grid = GridSearchCV(pipe, param_grid=params, scoring='r2',
      cv=TimeSeriesSplit(5), verbose=-1)

  grid.fit(X_train, y_train)
  print("> Training successfully done.\n")

  return grid

def train_dtr(X_train, y_train):

  print(colored("[Training Decision tree model.]", LOGCL, attrs=['bold']))
  pipe = make_pipeline(
      StandardScaler(),
      DecisionTreeRegressor(random_state=42)
      )
  
  params = {
      "decisiontreeregressor__splitter": ["best", "random"],
      "decisiontreeregressor__max_depth": [i for i in range(4, 20, 2)],
      "decisiontreeregressor__max_features": ["auto", "sqrt", "log2"]

  }
  grid = GridSearchCV(pipe, param_grid=params, scoring='r2',
      cv=TimeSeriesSplit(5), verbose=-1)

  grid.fit(X_train, y_train)
  print("> Training successfully done.\n")

  return grid

def train_nn(X_train, y_train):

  print(colored("[Training NN model.]", LOGCL, attrs=['bold']))
  pipe = make_pipeline(
      StandardScaler(),
      MLPRegressor(random_state=42, shuffle=False)
      )
  
  params = {
      "mlpregressor__hidden_layer_sizes": [[25, 30], [10, 15]],
      "mlpregressor__activation": ['relu', 'tanh'],
      "mlpregressor__learning_rate_init": [.01, .001, .0001, 0.00001],
      "mlpregressor__max_iter": [400, 600]
  }
  grid = GridSearchCV(pipe, param_grid=params, scoring='r2',
      cv=TimeSeriesSplit(5), verbose=-1)

  grid.fit(X_train, y_train)
  print("> Training successfully done.\n")

  return grid

def compare_models(mds):

  print(colored("[Comparing models' best cross validated scores]\n", LOGCL, attrs=['bold']))
  res = []
  for name, md in mds.items():
    print(f"> Name: {name}")
    sc = md.best_score_
    print(f"> R2: {sc}\n")
    res.append((name, sc,))
  name, _ = sorted(res, key=lambda x: x[1])[0]
  
  return name, mds[name]


def do_train_test_split(df):

  print(colored("[Splitting data into train, dev, test sets]", LOGCL, attrs=['bold']))
  target = "Total"
  y = df[target].to_numpy()
  X = df.loc[:, df.columns.isin(FEATURES)].to_numpy()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
  print("> Done!\n")

  return X_train, X_test, y_train, y_test
   

def get_new_model(df):
  
  # --------- Train, dev, test split
  X_train, X_test, y_train, y_test = do_train_test_split(df)

  # --------- Training of models
  mds = dict()
  mds["Linear regression"] = train_lr(X_train, y_train)
  mds["Decision tree"] = train_dtr(X_train, y_train)
  mds["Neural network"] = train_nn(X_train, y_train)

  # --------- Comparison and evaluation of models
  bestmdname, bestmd = compare_models(mds)

  return bestmd, X_test, y_test


