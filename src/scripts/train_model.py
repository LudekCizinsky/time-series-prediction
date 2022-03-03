from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict

def train_lr(X_train, y_train):

  print("[Training Linear regression model with polynomial features]")
  pipe = make_pipeline(
      StandardScaler(),
      PolynomialFeatures(degree=3, include_bias=False),
      LinearRegression()
      )
  pipe.fit(X_train, y_train)
  print("> Training successfully done.\n")

  return pipe

def train_dtr(X_train, y_train):

  print("[Training Decision tree model.]")
  pipe = make_pipeline(
      StandardScaler(),
      DecisionTreeRegressor(random_state=42)
      )
  pipe.fit(X_train, y_train)
  print("> Training successfully done.\n")

  return pipe

def compare_models(mds, X_dev, y_dev):

  print("[Comparing models on dev set.]\n")
  res = []
  for name, md in mds.items():
    print(f"> Name: {name}")
    y_hat_dev = md.predict(X_dev)
    mse = mean_squared_error(y_dev, y_hat_dev)
    print(f"> MSE: {mse}\n")
    res.append((name, mse,))
  name, _ = sorted(res, key=lambda x: x[1])[0]
  
  return name, mds[name]


def do_train_test_split(df):

  print("[Splitting data into train, dev, test sets]")
  target = "total_mean"
  y = df[target].to_numpy()
  X = df.loc[:, df.columns.isin(["Speed"])].to_numpy()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
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
  print(f"[Performance of the best model ({bestmdname}) on test set]")
  y_hat_test = bestmd.predict(X_test)
  mse = mean_squared_error(y_test, y_hat_test)
  r2 = r2_score(y_test, y_hat_test)
  print(f"> MSE: {mse}")
  print(f"> R2: {r2}\n")

  return bestmd 


