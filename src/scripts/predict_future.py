import os
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from . import query_future_weather
from . import pp_weather
from . import get_future_weather_X
from datetime import datetime
from termcolor import colored
LOGCL = os.environ['logcl']


def predict_future(nm, X_test, y_test):

  print(colored("[Loading old model from the disk]", LOGCL, attrs=['bold']))
  modelpath = f"{os.environ['path']}/models/latest.model"
  try:
    om = pickle.load(open(modelpath, 'rb'))
  except FileNotFoundError:
    print("> No previous model. Therefore will use the new model.\n")
    om = None

  if om is not None:
    print("> Done!\n")
    print(colored(f"[Performance of the latest old model on test set]", LOGCL, attrs=['bold']))
    y_hat_om = om.predict(X_test)
    mseom = mean_squared_error(y_test, y_hat_om)
    r2om = r2_score(y_test, y_hat_om)
    print(f"> MSE: {mseom}")
    print(f"> R2: {r2om}\n")


  print(colored(f"[Performance of the new model on test set]", LOGCL, attrs=['bold']))
  y_hat_nm = nm.predict(X_test)
  msenm = mean_squared_error(y_test, y_hat_nm)
  r2nm = r2_score(y_test, y_hat_nm)
  print(f"> MSE: {msenm}")
  print(f"> R2: {r2nm}\n")

  
  if om is not None:
    bm = om if r2om >= r2nm else nm
  else:
    bm = nm

  X = get_future_weather_X(pp_weather(query_future_weather(), False))

  print(colored(f"[Using the better model to make future prediction]", LOGCL, attrs=['bold']))
  y_hat = ','.join([str(v) for v in bm.predict(X)])
  filename = f"{os.environ['path']}/predictions/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
  with open(filename, "w") as out:
    out.write(y_hat)
  print("> Done!\n")
  
  print(colored("[Saving the best model.]", LOGCL, attrs=['bold']))
  filename1 = f"{os.environ['path']}/models/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".model"
  filename2 = f"{os.environ['path']}/models/latest.model"
  pickle.dump(bm, open(filename1, 'wb'))
  pickle.dump(bm, open(filename2, 'wb'))
  print("> Done!\n")

