## Intro
My project can be divided into the following stages:

1. Getting the data from DB 
2. Preprocessing
3. Model training
4. Choosing the best model and predicting future power production

Each stage is dependent on the previous one, i.e., each stage depends on the
output of the previous stage. Last but not the least, after the preprocessing
step, there is also an `EDA`, however, this acts as an independent unit.
Each of these stages will be now discussed more in detail in the below section.

## Detailed look at each stage

### Getting the data from DB 
In this stage, I used the code given in `template.py` to get the following two
datasets: 

1. Weather data
2. Power production data

and store them as `pandas dataframe`. In a similar fashion, I also retrieved the
future weather data.

### Preprocessing
In this stage, I identified the following challenges to be addressed for both
datasets:

1. Checking in both datasets for missing values
2. Getting rid of columns that are not used

The first point is solved by simply dropping rows with missing values. I decided
to do as there has never been any missing values. However, possibly I could
implement an alert which would send me for example an email notification when
the number of dropped rows would exceed a certain threshold. The second point is
self - explanatory as it does not make sense to keep columns which are not used
in memory.

In addition to the above steps, for weather dataset, I had to address the
following:

1. Sth

## Conclusion

