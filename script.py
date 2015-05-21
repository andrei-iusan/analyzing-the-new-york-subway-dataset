#! /usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dtime
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats import t
def predictions(dataframe, cost_history_filename=None):
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
    features = dataframe[['precipi', 'meantempi','fog']]
    features = features.join(dummy_days).join(dummy_hours).join(dummy_units)
    # print "Not normalized features"
    # print features.head()
    features, mu, sigma = normalize_features(features)
    # print "Normalized features"
    # print features.head()
    values = dataframe[['ENTRIESn_hourly']]
    m = len(values)

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
    #summing over all units
    agg = df.groupby(['DATEn','hour','rain'], as_index= False).aggregate(np.sum)
    agg = df.groupby(['hour', 'rain'], as_index = False).aggregate(np.mean)
    #average over days
    #print agg
    # # agg = agg[['hour', 'rain','ENTRIESn_hourly']]
    # l = agg.groupby(['hour','rain'], as_index = False).aggregate(len)
    # agg = agg.groupby(['hour','rain'], as_index = False).aggregate(np.sum)
    # av = agg['ENTRIESn_hourly']/l['ENTRIESn_hourly']
    # print agg,av
    #plot = ggplot(agg, aes(x='hour',y='ENTRIESn_hourly', color='rain')) + geom_point() + geom_line() + ggtitle("Average entries by hour") + ylab("Entries")
    rain = agg[agg.rain==1][['ENTRIESn_hourly','hour']]
    clear = agg[agg.rain==0][['ENTRIESn_hourly','hour']]
    plt.subplots_adjust(hspace=.4)
    plt.subplot(211)
    plt.title('Average hourly entries per unit')
    plt.plot(rain['hour'], rain['ENTRIESn_hourly'], color='b', label='rainy')
    plt.plot(clear['hour'], clear['ENTRIESn_hourly'], color='y', label='clear')
    plt.xlabel('hour')
    plt.ylabel('Average Entries')
    plt.legend()
    plt.subplot(212)
    plt.title('Average daily entries')
    df = turnstile_weather[['DATEn','day_week','ENTRIESn_hourly']]
    agg = df.groupby(['DATEn','day_week'], as_index=False).aggregate(np.sum)
    agg2 = agg.groupby(['day_week'], as_index=False).aggregate(np.mean)
    # rain = agg[agg.rain==1][['ENTRIESn_hourly','day_week']]
    # clear = agg[agg.rain==0][['ENTRIESn_hourly','day_week']]
    data = agg2[['ENTRIESn_hourly','day_week']]
    day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    w = .35
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

def main():
	log = open("NYC stats.txt", 'w')
	log.write("NYC statistics\n\n")

	turnstile_weather = pd.read_csv("turnstile_weather_v2.csv")
	turnstile_weather['datetime'] = turnstile_weather['DATEn'] + ' ' + turnstile_weather['TIMEn']
	clear_w = turnstile_weather[turnstile_weather.rain==0]['ENTRIESn_hourly']
	rainy_w = turnstile_weather[turnstile_weather.rain==1]['ENTRIESn_hourly']
	n_bins = 40
	plt.subplots_adjust(hspace=0.4)
	plt.subplot(211)
	plt.hist((clear_w, rainy_w), bins = n_bins, label=('clear','rainy'), color=('y','b'))
	plt.title("Histogram of Hourly Entries (a)")
	plt.xlabel("Entries")
	plt.ylabel("count")
	plt.legend()
	plt.subplot(212)
	plt.hist((clear_w, rainy_w), bins = n_bins, label=('clear','rainy'), color=('y','b'), normed=True)
	plt.title("Histogram of Hourly Entries (b)")
	plt.xlabel("Entries")
	plt.ylabel("frequency")
	plt.legend()
	plt.savefig("histogram of hourly entries")
	plt.close()
	log.write("saved histogram of hourly entries, with "+str(n_bins)+" bins\n=============\n\n")
	u,p = st.mannwhitneyu(clear_w, rainy_w)
	log.write("Mann Whitney test:\np="+ str(p)+ "\nU="+ str(u)+"\n\n")
	log.write("mean_clear="+ str(np.mean(clear_w)) + "\n")
	log.write("mean_rainy="+ str(np.mean(rainy_w)) + "\n\n")

	turnstile_weather['datetime'] = turnstile_weather['DATEn'] + ' ' + turnstile_weather['TIMEn']
	plot_hourly_entries(turnstile_weather, "hourly entries.png")

	features, values = predictions(turnstile_weather)
	model = sm.OLS(values, features)
	results = model.fit()
	predicted_values = results.predict(features)
	r_squared = compute_r_squared(turnstile_weather['ENTRIESn_hourly'], predicted_values) 
	log.write ("R2 value: "+str(r_squared)+'\n\n')
	# Theta values
	theta = results.params
	log.write("Theta values:\n")
	log.write('precipi: '+str(theta[0])+'\n')
	log.write('meantempi: '+str(theta[1])+'\n')
	log.write('fog: '+str(theta[2])+'\n\n')
	# k fold cross validation
	log.write("cross validation experiment\n\nR squared:\n")
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
		r_sq = compute_r_squared(turnstile_weather['ENTRIESn_hourly'][test_i], predicted_values)
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
	log.close()

if __name__ == '__main__':
	main()
