## System requirements
### Mandatory
- [X] Reads the latest data from the InfluxDB
- [X] Prepares the data for model training, including
    - Aligning the timestamps of the two data sources (e.g. through resampling or an inner join)
– [X] Handling missing data
– [X] Altering the wind direction to be a usable feature (by mapping to radians, encoding as categorical, or other)
– [X] Scaling the data to be within a set range
- [X] Trains a regression model of your choice, (e.g. LinearRegression)
- [X] Compares the newly trained model with the currently saved model, and picks the best performing model.
- [X] Saves the best performing model to disk
- [X] Uses the model to forecast wind power production

### Possible improvements
- [X] Actually use the wind direction as a feature using the described
  transformation to the vector
- [X] I could try to use a neural net or more complex model, also possibly play around with the degree number in pol. features
- [X] Think whether there can be any improvement made for the aligning power and weather data
- [X] I could include visualization of features for EDA part

## Hand in
You should hand in a report describing your solution, including which design choices and trade-offs you made. We value concise and well formulated argument with supplementary code examples. Here are some questions for inspirations:

- [X] Which steps does your (preprocessing, retraining, evaluation) pipeline include?
- [X] What is the format of the data once it reaches the model?
- [X] How did you align the data from the two data sources?
- [X] How did you decide on the type of model and hyperparameters?
- [X] How do you compare the newly trained model with the stored version?
- [X] How could the pipeline/system be improved?
- [X] How would you determine if the wind direction is a useful feature for the model?
- [X] Currently the system fetches data for the previous 90 days worth of data:
  – How would you determine if this is a good interval?
  – What are the trade-offs when deciding on the interval?
  – Would accuracy necessarily increase by including more data?

The maximum length for the report handed in is five pages.
