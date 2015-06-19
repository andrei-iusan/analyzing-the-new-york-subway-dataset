#! /usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dtime
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats import t
def predictions(dataframe, cost_history_filename=None, plots=False):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.
    '''
    # Select Features (try different features!)
    dummy_units = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    # days = map(lambda date_string: \
    #                 dtime.datetime.strptime(date_string,"%m-%d-%y").weekday(), \
    #                 dataframe.loc[:,'DATEn'])
    dummy_days = pd.get_dummies(dataframe['day_week'], prefix='day')
    dummy_hours = pd.get_dummies(dataframe['hour'], prefix='hour')
    dummy_conds = pd.get_dummies(dataframe['conds'], prefix='conds')
    features = dataframe[['precipi','fog','wspdi','pressurei']]
    features, mu, sigma = normalize_features(features)
    features['pressure2']=features['pressurei']*features['pressurei']
    features = features[['precipi','fog','wspdi','pressure2']]
    features = features.join(dummy_days).join(dummy_hours).join(dummy_units).join(dummy_conds)
    # print "Not normalized features"
    # print features.head()
    # print "Normalized features"
    # print features.head()
    values = dataframe[['ENTRIESn_hourly']]
    m = len(values)
    if plots:
        for feat in features.columns:
            plt.figure()
            # plt.scatter( values, features['precipi'])
            plt.scatter(features[feat],values)
            plt.savefig('scatterplots/'+feat)
            plt.close()
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values).flatten()

    return features_array, values_array

def normalize_features(array):
   """
   Normalize the features in the data set.
   """
   array_normalized = (array-array.mean())/array.std()
   mu = array.mean()
   sigma = array.std()

   return array_normalized, mu, sigma

def compute_cost(features, values, theta):
    """
    Compute the cost function given a set of features / values, 
    and the values for our thetas.
    
    This can be the same code as the compute_cost function in the lesson #3 exercises,
    but feel free to implement your own.
    """
    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)
    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    
    This can be the same gradient descent code as in the lesson #3 exercises,
    but feel free to implement your own.
    """
    
    m = len(values)
    cost_history = []

    X = np.matrix(features)
    Y = np.matrix(values).transpose()
    for i in range(num_iterations):
        TH = np.matrix(theta).transpose()
        XTY = X*TH-Y
        cost = compute_cost(features, values, theta)
        cost_history.append(cost)
        TH = TH - alpha/m * (X.transpose()*XTY)
        theta = np.array(TH.transpose())[0]
    return theta, pd.Series(cost_history)

def plot_cost_history(alpha, cost_history, filename):
   """This function is for viewing the plot of your cost history."""
   cost_df = pd.DataFrame({
      'Cost_History': cost_history,
      'Iteration': range(len(cost_history))
   })
   plt.figure()
   plt.plot(cost_df['Iteration'], cost_df['Cost_History'])
   plt.xlabel("Iteration")
   plt.ylabel("Cost")
   plt.savefig(filename)

def compute_r_squared(data, predictions):
    SST = ((data-np.mean(data))**2).sum()
    SSReg = ((predictions-np.mean(data))**2).sum()
    r_squared = SSReg / SST

    return r_squared

def plot_hourly_entries(turnstile_weather, filename):
    '''
    You are passed in a dataframe called turnstile_weather. 
    Use turnstile_weather along with ggplot to make a data visualization
    focused on the MTA and weather data we used in assignment #3.  
    You should feel free to implement something that we discussed in class 
    (e.g., scatterplots, line plots, or histograms) or attempt to implement
    something more advanced if you'd like.  

    Here are some suggestions for things to investigate and illustrate:
     * Ridership by time of day or day of week
     * How ridership varies based on Subway station
     * Which stations have more exits or entries at different times of day

    If you'd like to learn more about ggplot and its capabilities, take
    a look at the documentation at:
    https://pypi.python.org/pypi/ggplot/
     
    You can check out:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
     
    To see all the columns and data points included in the turnstile_weather 
    dataframe. 
     
    However, due to the limitation of our Amazon EC2 server, we are giving you about 1/3
    of the actual data in the turnstile_weather dataframe
    '''
    df = turnstile_weather[['DATEn','hour','rain','ENTRIESn_hourly']]
    # summing over all units
    agg = df.groupby(['DATEn','hour','rain'], as_index= False).aggregate(np.sum)
    # averaging over all days
    agg = df.groupby(['hour', 'rain'], as_index = False).aggregate(np.mean)
    rain = agg[agg.rain==1][['ENTRIESn_hourly','hour']]
    clear = agg[agg.rain==0][['ENTRIESn_hourly','hour']]
    plt.subplots_adjust(hspace=.4)
    plt.subplot(211)
    plt.title('Average hourly entries per unit')
    w = .35
    intervals = ['00-04','04-08','08-12','12-16','16-20','20-00']
    hourly_rain = np.array(rain['ENTRIESn_hourly'])
    hourly_rain = np.concatenate([hourly_rain[1:],[hourly_rain[0]]])
    hourly_clear = np.array(clear['ENTRIESn_hourly'])
    hourly_clear = np.concatenate([hourly_clear[1:],[hourly_clear[0]]])
    
    plt.bar(np.array(range(len(intervals))) + w/2,
        hourly_rain,
        color='b', label='rainy', width=w)
    plt.bar(np.array(range(len(intervals))) + w+w/2,
        hourly_clear,
        color='y', label='clear', width=w)
    # print rain[['ENTRIESn_hourly','hour']]
    # print hourly_rain
    plt.xticks(np.array(range(len(intervals)))+w+w/2,intervals)
    # plt.plot(clear['hour'], clear['ENTRIESn_hourly'], color='y', label='clear')
    plt.xlabel('hour interval')
    plt.ylabel('Average Entries')
    plt.legend(loc=0)
    plt.subplot(212)
    plt.title('Average daily entries')
    df = turnstile_weather[['DATEn','day_week','ENTRIESn_hourly']]
    agg = df.groupby(['DATEn','day_week'], as_index=False).aggregate(np.sum)
    agg2 = agg.groupby(['day_week'], as_index=False).aggregate(np.mean)
    # rain = agg[agg.rain==1][['ENTRIESn_hourly','day_week']]
    # clear = agg[agg.rain==0][['ENTRIESn_hourly','day_week']]
    data = agg2[['ENTRIESn_hourly','day_week']]
    day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.bar(np.array(data['day_week'])+w/2, data['ENTRIESn_hourly'], width=w, color='b')
    # plt.bar(np.array(clear['day_week'])+w, clear['ENTRIESn_hourly'], width=w, color='y', label='clear')
    plt.xticks(np.array(range(7))+w,day)
    plt.ylabel("Entries")
    #clear['day_week'], clear['ENTRIESn_hourly']\
    plt.savefig(filename)
    plt.close()
    plt.figure()
    plt.title("Total Number of Entries by Day")
    plt.xlabel("Date")
    plt.ylabel("Total entries")
    plt.bar(range(len(agg)), agg['ENTRIESn_hourly'])
    plt.xticks(np.array(range(len(agg)))+0.5, agg['DATEn'], rotation='vertical')
    plt.subplots_adjust(bottom=0.2, left = 0.15)
    (x1,x2,y1,y2) = plt.axis()
    plt.axis((0,31,y1,y2))
    plt.savefig("daily entries.png")

    plt.close()
    #agg['hour'], agg['ENTRIESn_hourly']
def plot_entries_histogram(clear_w,rainy_w, n_bins):
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(211)
    plt.hist((clear_w, rainy_w), bins = n_bins, label=('clear','rainy'), color=('y','b'))
    plt.title("Histogram of Hourly Entries (a)")
    plt.xlabel("Entries")
    plt.ylabel("count")
    plt.legend()
    plt.subplot(212)
    normed_clear = np.ones_like(clear_w)/len(clear_w)
    normed_rainy = np.ones_like(rainy_w)/len(rainy_w)
    plt.hist((clear_w, rainy_w), weights=(normed_clear, normed_rainy), bins = n_bins, label=('clear','rainy'), color=('y','b'))
    plt.title("Histogram of Hourly Entries (b)")
    plt.xlabel("Entries")
    plt.ylabel("Percent")
    plt.legend()
    plt.savefig("histogram of hourly entries")
    plt.close()

def CV_experiment(features, values, log):
     indeces = np.array(range(len(features)))
     np.random.seed(1)
     np.random.shuffle(indeces)
     k_cv = 10
     test_set_len = len(features)/k_cv
     R_sq_array = []
     for k in range(k_cv):
          train_i = indeces[range(0,k*test_set_len)+range((k+1)*test_set_len,len(features))]
          test_i = indeces[range(k*test_set_len,(k+1)*test_set_len)]
          model = sm.OLS(values[train_i], features[train_i])
          results = model.fit()
          predicted_values = results.predict(features[test_i])
          r_sq = compute_r_squared(values[test_i], predicted_values)
          R_sq_array.append(r_sq)
          log.write(str(r_sq)+'\n')
     Rm = np.mean(R_sq_array)
     Rsig = np.std(R_sq_array)
     conf_interval = t.interval(.95,len(R_sq_array)-1,loc=Rm, scale=Rsig)
     log.write("\nAverage R squared\n")
     log.write(str(Rm))
     log.write("\nR squared STD\n")
     log.write(str(Rsig))
     log.write("\nR squared 95% confidence interval:\n")
     log.write(str(conf_interval))
    

def main():
    log = open("NYC stats.txt", 'w')
    log.write("NYC statistics\n\n")

    turnstile_weather = pd.read_csv("turnstile_weather_v2.csv")
    turnstile_weather['datetime'] = turnstile_weather['DATEn'] + ' ' + turnstile_weather['TIMEn']
    clear_w = turnstile_weather[turnstile_weather.rain==0]['ENTRIESn_hourly']
    rainy_w = turnstile_weather[turnstile_weather.rain==1]['ENTRIESn_hourly']
    n_bins=40
    plot_entries_histogram(clear_w, rainy_w,n_bins)
    log.write("saved histogram of hourly entries, with "+str(n_bins)+" bins\n=============\n\n")
    u,p = st.mannwhitneyu(clear_w, rainy_w)
    log.write("Mann Whitney test (two tailed test):\np="+ str(2*p)+ "\nU="+ str(u)+"\n\n")
    log.write("mean_clear="+ str(np.mean(clear_w)) + "\n")
    log.write("mean_rainy="+ str(np.mean(rainy_w)) + "\n\n")

    turnstile_weather['datetime'] = turnstile_weather['DATEn'] + ' ' + turnstile_weather['TIMEn']
    plot_hourly_entries(turnstile_weather, "hourly entries.png")

    features, values = predictions(turnstile_weather, plots=True)
    model = sm.OLS(values, features)
    results = model.fit()
    predicted_values = results.predict(features)
    r_squared = compute_r_squared(turnstile_weather['ENTRIESn_hourly'], predicted_values) 
    log.write ("R2 value: "+str(r_squared)+'\n\n')
    # Theta values
    theta = results.params
    #'precipi','fog','wspdi','pressure2'
    log.write("Theta values:\n")
    log.write('precipi: '+str(theta[0])+'\n')
    log.write('fog: '+str(theta[1])+'\n')
    log.write('wspdi: '+str(theta[2])+'\n')
    log.write('pressure2: '+str(theta[3])+'\n\n')
    # k fold cross validation
    log.write("cross validation experiment\n\nR squared:\n")
    CV_experiment(features, values, log)
    log.close()

if __name__ == '__main__':
    main()
