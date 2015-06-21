
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from pandas import DataFrame, Series


# In[24]:

path = "/Users/xiaoyiliu/Dropbox/udacity_intro_to_data_science/Berkeley.csv"
data = pd.read_csv (path, sep = ",")
# data


# In[9]:

numbers = [1, 2, 3, 4, 5]
print np.mean (numbers)
print np.median (numbers)
print np.std (numbers)


# In[16]:

array = np.array([1, 4, 5, 8, 9], float)
print array
print array [1]
print array [2:]
print array [:2]

two_D_array = np.array([[1, 2, 3], [4, 5, 6]], float)


# In[18]:

array_1 = np.array([[1, 2], [3, 4]], float)
array_2 = np.array([[5, 6], [7, 8]], float)

print array_1 * array_2
print np.dot (array_1, array_2)


# In[12]:

series = pd.Series(['Dave', 'Cheng-Han', 'Udacity', 42, -1789710578])
print series
print ""
series = pd.Series(['Dave', 'Cheng-Han', 359, 9001],index=['Instructor', 'Curriculum Manager', 'Course Number', 'Power Level'])
print series
print ""
print series['Instructor']
print ""
print series[['Instructor', 'Curriculum Manager', 'Course Number']]


# In[15]:

data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
        'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions',
                     'Lions', 'Lions'],
        'wins': [11, 8, 10, 15, 11, 6, 10, 4],
        'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data)
print football
print ""
print football.dtypes
print ""
print football.describe()
print ""
print football.head()
print ""
print football.tail()


# In[33]:

countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
                 'Netherlands', 'Germany', 'Switzerland', 'Belarus',
                 'Austria', 'France', 'Poland', 'China', 'Korea', 
                 'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
                 'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
                 'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']
gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]

# python dictionary
medal = {'countries': Series(countries), 'gold': Series(gold), 'silver': Series(silver), 'bronze': Series(bronze)};
# can add index = {'1', '2'.....}
olympic_medal_counts_df = DataFrame (medal);
print olympic_medal_counts_df
print np.mean(olympic_medal_counts_df['gold'])
print olympic_medal_counts_df.applymap (lambda x: x>=2)
print olympic_medal_counts_df ['gold'].map (lambda x: x>=2)

# find mean in different dimension
print np.mean (olympic_medal_counts_df, axis = 1)


# # write to data
# data = pd.read_csv () 
# 
# 
# data.to_csv ('file_name.csv')

# # SQL in Python
# import pandas
# 
# import pandasql
# 
# def select_first_50(filename):
#     # Read in our aadhaar_data csv to a pandas dataframe.  Afterwards, we rename the columns
#     # by replacing spaces with underscores and setting all characters to lowercase, so the
#     # column names more closely resemble columns names one might find in a table.
#     aadhaar_data = pandas.read_csv(filename)
#     aadhaar_data.rename(columns = lambda x: x.replace(' ', '_').lower(), inplace=True)
# 
#     # Select out the first 50 values for "registrar" and "enrolment_agency"
#     # in the aadhaar_data table using SQL syntax. 
#     #
#     # Note that "enrolment_agency" is spelled with one l. Also, the order
#     # of the select does matter. Make sure you select registrar then enrolment agency
#     # in your query.
#     q = """
#     select registrar, enrolment_agency from aadhaar_data limit 50
#     """
# 
#     #Execute your SQL command against the pandas frame
#     aadhaar_solution = pandasql.sqldf(q.lower(), locals())
#     return aadhaar_solution    

# # get json in python
# import json
# 
# import requests
# 
# def api_get_request(url):
#     # In this exercise, you want to call the last.fm API to get a list of the
#     # top artists in Spain.
#     #
#     # Once you've done this, return the name of the number 1 top artist in Spain.
#     data = requests.get (url).text
#     data = json.loads (data)
#     
#     
#     return data ['topartists']['artist'][0]['name'] # return the top artist in Spain
# 

# # pandas: fillna (value)
# from pandas import *
# 
# import numpy
# 
# def imputation(filename):
#     # Pandas dataframes have a method called 'fillna(value)', such that you can
#     # pass in a single value to replace any NAs in a dataframe or series. You
#     # can call it like this: 
#     #     dataframe['column'] = dataframe['column'].fillna(value)
#     #
#     # Using the numpy.mean function, which calculates the mean of a numpy
#     # array, impute any missing values in our Lahman baseball
#     # data sets 'weight' column by setting them equal to the average weight.
#     # 
#     # You can access the 'weight' colum in the baseball data frame by
#     # calling baseball['weight']
# 
#     baseball = pandas.read_csv(filename)
#     baseball ['weight'] = baseball ['weight'].fillna (numpy.mean (baseball ['weight']))
#     
#     #YOUR CODE GOES HERE
# 
#     return baseball

# # cast into integers
# 
# import pandas
# 
# import pandasql
# 
# 
# def num_rainy_days(filename):
#     '''
#     This function should run a SQL query on a dataframe of
#     weather data.  The SQL query should return one column and
#     one row - a count of the number of days in the dataframe where
#     the rain column is equal to 1 (i.e., the number of days it
#     rained).  The dataframe will be titled 'weather_data'. You'll
#     need to provide the SQL query.  You might find SQL's count function
#     useful for this exercise.  You can read more about it here:
#     
#     https://dev.mysql.com/doc/refman/5.1/en/counting-rows.html
#     
#     You might also find that interpreting numbers as integers or floats may not
#     work initially.  In order to get around this issue, it may be useful to cast
#     these numbers as integers.  This can be done by writing cast(column as integer).
#     So for example, if we wanted to cast the maxtempi column as an integer, we would actually
#     write something like where cast(maxtempi as integer) = 76, as opposed to simply 
#     where maxtempi = 76.
#     
#     You can see the weather data that we are passing in below:
#     https://www.dropbox.com/s/7sf0yqc9ykpq3w8/weather_underground.csv
#     '''
#     weather_data = pandas.read_csv(filename)
# 
#     q = """
#     select count (*)
#     from weather_data
#     where cast(weather_data.rain as integer) = 1
#     """
#     
#     #Execute your SQL command against the pandas frame
#     rainy_days = pandasql.sqldf(q.lower(), locals())
#     return rainy_days
# 

# # sql: decide a day is weekend day or not
# 
# import pandas
# 
# import pandasql
# 
# def avg_weekend_temperature(filename):
#     '''
#     This function should run a SQL query on a dataframe of
#     weather data.  The SQL query should return one column and
#     one row - the average meantempi on days that are a Saturday
#     or Sunday (i.e., the the average mean temperature on weekends).
#     The dataframe will be titled 'weather_data' and you can access
#     the date in the dataframe via the 'date' column.
#     
#     You'll need to provide  the SQL query.
#     
#     You might also find that interpreting numbers as integers or floats may not
#     work initially.  In order to get around this issue, it may be useful to cast
#     these numbers as integers.  This can be done by writing cast(column as integer).
#     So for example, if we wanted to cast the maxtempi column as an integer, we would actually
#     write something like where cast(maxtempi as integer) = 76, as opposed to simply 
#     where maxtempi = 76.
#     
#     Also, you can convert dates to days of the week via the 'strftime' keyword in SQL.
#     For example, cast (strftime('%w', date) as integer) will return 0 if the date
#     is a Sunday or 6 if the date is a Saturday.
#     
#     You can see the weather data that we are passing in below:
#     https://www.dropbox.com/s/7sf0yqc9ykpq3w8/weather_underground.csv
#     '''
#     weather_data = pandas.read_csv(filename)
# 
#     q = """
#     select avg (weather_data.meantempi)
#     from weather_data
#     where cast (strftime('%w', weather_data.date) as integer) = 0 or 
#     cast (strftime('%w', weather_data.date) as integer) = 6
#     """
#     
#     #Execute your SQL command against the pandas frame
#     mean_temp_weekends = pandasql.sqldf(q.lower(), locals())
#     return mean_temp_weekends

# # import csv to deal with text file
# 
# import csv
# 
# def fix_turnstile_data(filenames):
#     '''
#     Filenames is a list of MTA Subway turnstile text files. A link to an example
#     MTA Subway turnstile text file can be seen at the URL below:
#     http://web.mta.info/developers/data/nyct/turnstile/turnstile_110507.txt
#     
#     As you can see, there are numerous data points included in each row of the
#     a MTA Subway turnstile text file. 
# 
#     You want to write a function that will update each row in the text
#     file so there is only one entry per row. A few examples below:
#     A002,R051,02-00-00,05-28-11,00:00:00,REGULAR,003178521,001100739
#     A002,R051,02-00-00,05-28-11,04:00:00,REGULAR,003178541,001100746
#     A002,R051,02-00-00,05-28-11,08:00:00,REGULAR,003178559,001100775
#     
#     Write the updates to a different text file in the format of "updated_" + filename.
#     For example:
#         1) if you read in a text file called "turnstile_110521.txt"
#         2) you should write the updated data to "updated_turnstile_110521.txt"
# 
#     The order of the fields should be preserved. Remember to read through the 
#     Instructor Notes below for more details on the task. 
#     
#     In addition, here is a CSV reader/writer introductory tutorial:
#     http://goo.gl/HBbvyy
#     
#     You can see a sample of the turnstile text file that's passed into this function
#     and the the corresponding updated file in the links below:
#     
#     Sample input file:
#     https://www.dropbox.com/s/mpin5zv4hgrx244/turnstile_110528.txt
#     Sample updated file:
#     https://www.dropbox.com/s/074xbgio4c39b7h/solution_turnstile_110528.txt
#     '''
#     for name in filenames:
#         # your code here
#         updated_name = 'updated_' + name
#         file_in = open (name, 'r')
#         
#         read_file = csv.reader (file_in, delimiter = ',')
#         file_out = open (updated_name, 'w')
#         write_file = csv.writer (file_out, delimiter = ',')
#         
#         for line in read_file:
#             line_length = len (line)
#             for loop in range ((line_length - 3)/5):
#                 row = [line [0], line [1], line [2], line [loop * 5 + 3], line [loop * 5 + 4], line [loop * 5 + 5], line [loop * 5 + 6], line [loop * 5 + 7] ]
#                 write_file.writerow (row)
#                 
#         file_in.close ()
#         file_out.close ()
#         
#         
# def create_master_turnstile_file(filenames, output_file):
#     '''
#     Write a function that takes the files in the list filenames, which all have the 
#     columns 'C/A, UNIT, SCP, DATEn, TIMEn, DESCn, ENTRIESn, EXITSn', and consolidates
#     them into one file located at output_file.  There should be ONE row with the column
#     headers, located at the top of the file. The input files do not have column header
#     rows of their own.
#     
#     For example, if file_1 has:
#     'C/A, UNIT, SCP, DATEn, TIMEn, DESCn, ENTRIESn, EXITSn'
#     line 1 ...
#     line 2 ...
#     
#     and another file, file_2 has:
#     'C/A, UNIT, SCP, DATEn, TIMEn, DESCn, ENTRIESn, EXITSn'
#     line 3 ...
#     line 4 ...
#     line 5 ...
#     
#     We need to combine file_1 and file_2 into a master_file like below:
#      'C/A, UNIT, SCP, DATEn, TIMEn, DESCn, ENTRIESn, EXITSn'
#     line 1 ...
#     line 2 ...
#     line 3 ...
#     line 4 ...
#     line 5 ...
#     '''
#     with open(output_file, 'w') as master_file:
#        master_file.write('C/A,UNIT,SCP,DATEn,TIMEn,DESCn,ENTRIESn,EXITSn\n')
#         
#        for file_name in filenames:
#            file_in = open (file_name, 'r')
#             
#            for line in file_in:
#                master_file.write (line)
#      
#        file_in.close ()
#        master_file.close ()
#         
# 
#                 # your code here

# # pandas: shift function
# 
# import pandas
# 
# def get_hourly_entries(df):
#     '''
#     The data in the MTA Subway Turnstile data reports on the cumulative
#     number of entries and exits per row.  Assume that you have a dataframe
#     called df that contains only the rows for a particular turnstile machine
#     (i.e., unique SCP, C/A, and UNIT).  This function should change
#     these cumulative entry numbers to a count of entries since the last reading
#     (i.e., entries since the last row in the dataframe).
#     
#     More specifically, you want to do two things:
#        1) Create a new column called ENTRIESn_hourly
#        2) Assign to the column the difference between ENTRIESn of the current row 
#           and the previous row. If there is any NaN, fill/replace it with 1.
#     
#     You may find the pandas functions shift() and fillna() to be helpful in this exercise.
#     
#     Examples of what your dataframe should look like at the end of this exercise:
#     
#            C/A  UNIT       SCP     DATEn     TIMEn    DESCn  ENTRIESn    EXITSn  ENTRIESn_hourly
#     0     A002  R051  02-00-00  05-01-11  00:00:00  REGULAR   3144312   1088151                1
#     1     A002  R051  02-00-00  05-01-11  04:00:00  REGULAR   3144335   1088159               23
#     2     A002  R051  02-00-00  05-01-11  08:00:00  REGULAR   3144353   1088177               18
#     3     A002  R051  02-00-00  05-01-11  12:00:00  REGULAR   3144424   1088231               71
#     4     A002  R051  02-00-00  05-01-11  16:00:00  REGULAR   3144594   1088275              170
#     5     A002  R051  02-00-00  05-01-11  20:00:00  REGULAR   3144808   1088317              214
#     6     A002  R051  02-00-00  05-02-11  00:00:00  REGULAR   3144895   1088328               87
#     7     A002  R051  02-00-00  05-02-11  04:00:00  REGULAR   3144905   1088331               10
#     8     A002  R051  02-00-00  05-02-11  08:00:00  REGULAR   3144941   1088420               36
#     9     A002  R051  02-00-00  05-02-11  12:00:00  REGULAR   3145094   1088753              153
#     10    A002  R051  02-00-00  05-02-11  16:00:00  REGULAR   3145337   1088823              243
#     ...
#     ...
# 
#     '''
#     df ['ENTRIESn_hourly'] = (df ['ENTRIESn'] - df ['ENTRIESn'].shift (1)).fillna (1)
#         
#     #your code here
#     return df

# # datetime: strptime strftime
# 
# import datetime
# 
# def reformat_subway_dates(date):
#     '''
#     The dates in our subway data are formatted in the format month-day-year.
#     The dates in our weather underground data are formatted year-month-day.
#     
#     In order to join these two data sets together, we'll want the dates formatted
#     the same way.  Write a function that takes as its input a date in the MTA Subway
#     data format, and returns a date in the weather underground format.
#     
#     Hint: 
#     There are a couple of useful functions in the datetime library that will
#     help on this assignment, called strptime and strftime. 
#     More info can be seen here and further in the documentation section:
#     http://docs.python.org/2/library/datetime.html#datetime.datetime.strptime
#     '''
# 
#     date_formatted = # your code here
#     return date_formatted
# 

# # Welch's t-test in python
# import scipy.stats
# 
# import numpy
# import scipy.stats
# import pandas
# 
# def compare_averages(filename):
#     """
#     Performs a t-test on two sets of baseball data (left-handed and right-handed hitters).
# 
#     You will be given a csv file that has three columns.  A player's
#     name, handedness (L for lefthanded or R for righthanded) and their
#     career batting average (called 'avg'). You can look at the csv
#     file by downloading the baseball_stats file from Downloadables below. 
#     
#     Write a function that will read that the csv file into a pandas data frame,
#     and run Welch's t-test on the two cohorts defined by handedness.
#     
#     One cohort should be a data frame of right-handed batters. And the other
#     cohort should be a data frame of left-handed batters.
#     
#     We have included the scipy.stats library to help you write
#     or implement Welch's t-test:
#     http://docs.scipy.org/doc/scipy/reference/stats.html
#     
#     With a significance level of 95%, if there is no difference
#     between the two cohorts, return a tuple consisting of
#     True, and then the tuple returned by scipy.stats.ttest.  
#     
#     If there is a difference, return a tuple consisting of
#     False, and then the tuple returned by scipy.stats.ttest.
#     
#     For example, the tuple that you return may look like:
#     (True, (9.93570222, 0.000023))
#     """
#     
#     data = pandas.read_csv (filename)
#     left = data [data ['handedness'] == 'L']
#     left_avg = left ['avg']
#     right = data [data ['handedness'] == 'R']
#     right_avg = right ['avg']
#     result = scipy.stats.ttest_ind (left_avg, right_avg, equal_var = False)
#     if result [1] < 0.05:
#         return (False, result)
#     else:
#         return (True, result)

# # plot hisgram
# 
# import numpy as np
# 
# import pandas
# 
# import matplotlib.pyplot as plt
# 
# def entries_histogram (turnstile_weather):
#     '''
#     Before we perform any analysis, it might be useful to take a
#     look at the data we're hoping to analyze. More specifically, let's 
#     examine the hourly entries in our NYC subway data and determine what
#     distribution the data follows. This data is stored in a dataframe
#     called turnstile_weather under the ['ENTRIESn_hourly'] column.
#     
#     Let's plot two histograms on the same axes to show hourly
#     entries when raining vs. when not raining. Here's an example on how
#     to plot histograms with pandas and matplotlib:
#     turnstile_weather['column_to_graph'].hist()
#     
#     Your histograph may look similar to bar graph in the instructor notes below.
#     
#     You can read a bit about using matplotlib and pandas to plot histograms here:
#     http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms
#     
#     You can see the information contained within the turnstile weather data here:
#     https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
#     '''
#     
#     plt.figure()
#     turnstile_weather [turnstile_weather ['rain'] == 1] ['ENTRIESn_hourly'].hist () # your code here to plot a historgram for hourly entries when it is raining
#     turnstile_weather [turnstile_weather ['rain'] == 0] ['ENTRIESn_hourly'].hist () # your code here to plot a historgram for hourly entries when it is not raining
#     return plt
# 

# import numpy as np
# import pandas
# from ggplot import *
# 
# """
# In this question, you need to:
# 1) implement the compute_cost() and gradient_descent() procedures
# 2) Select features (in the predictions procedure) and make predictions.
# 
# """
# 
# def normalize_features(df):
#     """
#     Normalize the features in the data set.
#     """
#     mu = df.mean()
#     sigma = df.std()
#     
#     if (sigma == 0).any():
#         raise Exception("One or more features had the same value for all samples, and thus could " + \
#                          "not be normalized. Please do not include features with only a single value " + \
#                          "in your model.")
#     df_normalized = (df - df.mean()) / df.std()
# 
#     return df_normalized, mu, sigma
# 
# def compute_cost(features, values, theta):
#     """
#     Compute the cost function given a set of features / values, 
#     and the values for our thetas.
#     
#     This can be the same code as the compute_cost function in the lesson #3 exercises,
#     but feel free to implement your own.
#     """
#     
#     # your code here
#     cost = 1/(2 * len (values)) * np.sum (np.square (np.dot (features, theta) - values))
# 
#     return cost
# 
# def gradient_descent(features, values, theta, alpha, num_iterations):
#     """
#     Perform gradient descent given a data set with an arbitrary number of features.
#     
#     This can be the same gradient descent code as in the lesson #3 exercises,
#     but feel free to implement your own.
#     """
#     
#     cost_history = []
#     m = len (values)
#     n = len (theta)
#     
#     for loop_0 in range (num_iterations):
#         cost_history.append (compute_cost(features, values, theta))
#         theta = theta - (alpha/m) * np.dot((np.dot(features,theta) - values),features)
# 
#     return theta, pandas.Series(cost_history)
# 
# def predictions(dataframe):
#     '''
#     The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
#     Using the information stored in the dataframe, let's predict the ridership of
#     the NYC subway using linear regression with gradient descent.
#     
#     You can download the complete turnstile weather dataframe here:
#     https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv    
#     
#     Your prediction should have a R^2 value of 0.40 or better.
#     You need to experiment using various input features contained in the dataframe. 
#     We recommend that you don't use the EXITSn_hourly feature as an input to the 
#     linear model because we cannot use it as a predictor: we cannot use exits 
#     counts as a way to predict entry counts. 
#     
#     Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
#     give you a random subet (~15%) of the data contained in 
#     turnstile_data_master_with_weather.csv. You are encouraged to experiment with 
#     this computer on your own computer, locally. 
#     
#     
#     If you'd like to view a plot of your cost history, uncomment the call to 
#     plot_cost_history below. The slowdown from plotting is significant, so if you 
#     are timing out, the first thing to do is to comment out the plot command again.
#     
#     If you receive a "server has encountered an error" message, that means you are 
#     hitting the 30-second limit that's placed on running your program. Try using a 
#     smaller number for num_iterations if that's the case.
#     
#     If you are using your own algorithm/models, see if you can optimize your code so 
#     that it runs faster.
#     '''
#     # Select Features (try different features!)
#     # features = dataframe[['rain', 'precipi', 'Hour', 'meantempi']]
#     features = dataframe[['rain', 'precipi', 'Hour']]
#     
#     # Add UNIT to features using dummy variables
#     dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
#     features = features.join(dummy_units)
#     
#     # Values
#     values = dataframe['ENTRIESn_hourly']
#     m = len(values)
# 
#     features, mu, sigma = normalize_features(features)
#     features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
#     
#     # Convert features and values to numpy arrays
#     features_array = np.array(features)
#     values_array = np.array(values)
# 
#     # Set values for alpha, number of iterations.
#     alpha = 0.1 # please feel free to change this value
#     num_iterations = 75 # please feel free to change this value
# 
#     # Initialize theta, perform gradient descent
#     theta_gradient_descent = np.zeros(len(features.columns))
#     theta_gradient_descent, cost_history = gradient_descent(features_array, 
#                                                             values_array, 
#                                                             theta_gradient_descent, 
#                                                             alpha, 
#                                                             num_iterations)
#     
#     plot = None
#     # -------------------------------------------------
#     # Uncomment the next line to see your cost history
#     # -------------------------------------------------
#     # plot = plot_cost_history(alpha, cost_history)
#     # 
#     # Please note, there is a possibility that plotting
#     # this in addition to your calculation will exceed 
#     # the 30 second limit on the compute servers.
#     
#     predictions = np.dot(features_array, theta_gradient_descent)
#     return predictions, plot
# 
# 
# def plot_cost_history(alpha, cost_history):
#    """This function is for viewing the plot of your cost history.
#    You can run it by uncommenting this
# 
#        plot_cost_history(alpha, cost_history) 
# 
#    call in predictions.
#    
#    If you want to run this locally, you should print the return value
#    from this function.
#    """
#    cost_df = pandas.DataFrame({
#       'Cost_History': cost_history,
#       'Iteration': range(len(cost_history))
#    })
#    return ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
#       geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )
# 
# 
# 

# In[3]:

# plt figure
import numpy as np

import scipy

import matplotlib.pyplot as plt

def plot_residuals(turnstile_weather, predictions):
    '''
    Using the same methods that we used to plot a histogram of entries
    per hour for our data, why don't you make a histogram of the residuals
    (that is, the difference between the original hourly entry data and the predicted values).
    Try different binwidths for your histogram.

    Based on this residual histogram, do you have any insight into how our model
    performed?  Reading a bit on this webpage might be useful:

    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''
    
    plt.figure()
    (turnstile_weather['ENTRIESn_hourly'] - predictions).hist()
    return plt


# # use linear regression package
# 
# # -*- coding: utf-8 -*-
# 
# import numpy as np
# 
# import pandas
# 
# import scipy
# 
# import statsmodels.api as sm
# 
# """
# In this optional exercise, you should complete the function called 
# predictions(turnstile_weather). This function takes in our pandas 
# turnstile weather dataframe, and returns a set of predicted ridership values,
# based on the other information in the dataframe.  
# 
# In exercise 3.5 we used Gradient Descent in order to compute the coefficients
# theta used for the ridership prediction. Here you should attempt to implement 
# another way of computing the coeffcients theta. You may also try using a reference implementation such as: 
# http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.OLS.html
# 
# One of the advantages of the statsmodels implementation is that it gives you
# easy access to the values of the coefficients theta. This can help you infer relationships 
# between variables in the dataset.
# 
# You may also experiment with polynomial terms as part of the input variables.  
# 
# The following links might be useful: 
# http://en.wikipedia.org/wiki/Ordinary_least_squares
# http://en.wikipedia.org/w/index.php?title=Linear_least_squares_(mathematics)
# http://en.wikipedia.org/wiki/Polynomial_regression
# 
# This is your playground. Go wild!
# 
# How does your choice of linear regression compare to linear regression
# with gradient descent computed in Exercise 3.5?
# 
# You can look at the information contained in the turnstile_weather dataframe below:
# https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
# 
# Note: due to the memory and CPU limitation of our amazon EC2 instance, we will
# give you a random subset (~10%) of the data contained in turnstile_data_master_with_weather.csv
# 
# If you receive a "server has encountered an error" message, that means you are hitting 
# the 30 second limit that's placed on running your program. See if you can optimize your code so it
# runs faster.
# """
# 
# def predictions(weather_turnstile):
#     #
#     # Your implementation goes here. Feel free to write additional
#     # helper functions
#     # 
#     
#     dummy_units = pandas.get_dummies (weather_turnstile['UNIT'], prefix = 'unit') 
#     features = weather_turnstile [['rain', 'precipi', 'Hour', 'meantempi']].join(dummy_units) 
#     values = weather_turnstile [['ENTRIESn_hourly']]
#     m = len(values)
#     features['ones'] = np.ones(m)
#     features_array = np.array (features)
#     values_array = np.array(values).flatten()
#     X = features_array
#     Y = values_array
#     lm = sm.OLS(Y,X).fit()
#     prediction = lm.predict (X)
#     
#     return prediction
# 

# # example of ggplot
# 
# from pandas import *
# 
# from ggplot import *
# 
# import pandas
# 
# def lineplot (hr_year_csv):
#     # A csv file will be passed in as an argument which
#     # contains two columns -- 'HR' (the number of homerun hits)
#     # and 'yearID' (the year in which the homeruns were hit).
#     #
#     # Fill out the body of this function, lineplot, to use the
#     # passed-in csv file, hr_year.csv, and create a
#     # chart with points connected by lines, both colored 'red',
#     # showing the number of HR by year.
#     #
#     # You will want to first load the csv file into a pandas dataframe
#     # and use the pandas dataframe along with ggplot to create your visualization
#     #
#     # You can check out the data in the csv file at the link below:
#     # https://www.dropbox.com/s/awgdal71hc1u06d/hr_year.csv
#     #
#     # You can read more about ggplot at the following link:
#     # https://github.com/yhat/ggplot/
#     
#     data = pandas.read_csv (hr_year_csv)
#     gg = ggplot (data, aes ('yearID', 'HR')) + geom_point (color = 'red') + geom_line (color = 'red') + ggtitle ('the number of HR by year') + xlab ('year') + ylab ('HR')
#     #YOUR CODE GOES HERE
#     return gg
# 

# # ggplot with categorical data
# 
# 
# import pandas
# 
# from ggplot import *
# 
# 
# def lineplot_compare (hr_by_team_year_sf_la_csv):
#     # Write a function, lineplot_compare, that will read a csv file
#     # called hr_by_team_year_sf_la.csv and plot it using pandas and ggplot.
#     #
#     # This csv file has three columns: yearID, HR, and teamID. The data in the
#     # file gives the total number of home runs hit each year by the SF Giants 
#     # (teamID == 'SFN') and the LA Dodgers (teamID == "LAN"). Produce a 
#     # visualization comparing the total home runs by year of the two teams. 
#     # 
#     # You can see the data in hr_by_team_year_sf_la_csv
#     # at the link below:
#     # https://www.dropbox.com/s/wn43cngo2wdle2b/hr_by_team_year_sf_la.csv
#     #
#     # Note that to differentiate between multiple categories on the 
#     # same plot in ggplot, we can pass color in with the other arguments
#     # to aes, rather than in our geometry functions. For example, 
#     # ggplot(data, aes(xvar, yvar, color=category_var)). This should help you 
#     # in this exercise.
#     
#     data = pandas.read_csv (hr_by_team_year_sf_la_csv)
#     
#     gg = ggplot (data, aes ('yearID', 'HR', color = 'teamID')) + geom_point () + geom_line ()
#     return gg
# 

# from pandas import *
# from ggplot import *
# 
# def plot_weather_data(turnstile_weather):
#     ''' 
#     plot_weather_data is passed a dataframe called turnstile_weather. 
#     Use turnstile_weather along with ggplot to make another data visualization
#     focused on the MTA and weather data we used in Project 3.
#     
#     Make a type of visualization different than what you did in the previous exercise.
#     Try to use the data in a different way (e.g., if you made a lineplot concerning 
#     ridership and time of day in exercise #1, maybe look at weather and try to make a 
#     histogram in this exercise). Or try to use multiple encodings in your graph if 
#     you didn't in the previous exercise.
#     
#     You should feel free to implement something that we discussed in class 
#     (e.g., scatterplots, line plots, or histograms) or attempt to implement
#     something more advanced if you'd like.
# 
#     Here are some suggestions for things to investigate and illustrate:
#      * Ridership by time-of-day or day-of-week
#      * How ridership varies by subway station (UNIT)
#      * Which stations have more exits or entries at different times of day
#        (You can use UNIT as a proxy for subway station.)
# 
#     If you'd like to learn more about ggplot and its capabilities, take
#     a look at the documentation at:
#     https://pypi.python.org/pypi/ggplot/
#      
#     You can check out the link 
#     https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
#     to see all the columns and data points included in the turnstile_weather 
#     dataframe.
#      
#    However, due to the limitation of our Amazon EC2 server, we are giving you a random
#     subset, about 1/3 of the actual data in the turnstile_weather dataframe.
#     '''
# 
#     plot_df = turnstile_weather[['DATEn', 'ENTRIESn_hourly']]
#     plot_df.is_copy = False
#     plot_df['week_day'] = turnstile_weather['DATEn'].map(lambda x: datetime.strptime(x, '%Y-%m-%d').weekday())
#     plot = ggplot(plot_df, aes('week_day', 'ENTRIESn_hourly')) + geom_bar(stat = "bar")
#     return plot
# 

# # word counter
# 
# import logging
# 
# import sys
# 
# import string
# 
# from util import logfile
# 
# logging.basicConfig(filename=logfile, format='%(message)s',
#                    level=logging.INFO, filemode='w')
# 
# 
# def word_count():
#     # For this exercise, write a program that serially counts the number of occurrences
#     # of each word in the book Alice in Wonderland.
#     #
#     # The text of Alice in Wonderland will be fed into your program line-by-line.
#     # Your program needs to take each line and do the following:
#     # 1) Tokenize the line into string tokens by whitespace
#     #    Example: "Hello, World!" should be converted into "Hello," and "World!"
#     #    (This part has been done for you.)
#     #
#     # 2) Remove all punctuation
#     #    Example: "Hello," and "World!" should be converted into "Hello" and "World"
#     #
#     # 3) Make all letters lowercase
#     #    Example: "Hello" and "World" should be converted to "hello" and "world"
#     #
#     # Store the the number of times that a word appears in Alice in Wonderland
#     # in the word_counts dictionary, and then *print* (don't return) that dictionary
#     #
#     # In this exercise, print statements will be considered your final output. Because
#     # of this, printing a debug statement will cause the grader to break. Instead, 
#     # you can use the logging module which we've configured for you.
#     #
#     # For example:
#     # logging.info("My debugging message")
#     #
#     # The logging module can be used to give you more control over your
#     # debugging or other messages than you can get by printing them. Messages 
#     # logged via the logger we configured will be saved to a
#     # file. If you click "Test Run", then you will see the contents of that file
#     # once your program has finished running.
#     # 
#     # The logging module also has other capabilities; see 
#     # https://docs.python.org/2/library/logging.html
#     # for more information.
# 
#     word_counts = {}
# 
#     for line in sys.stdin:
#         data = line.strip ().split (" ")
# 
#         for key in data:
#             key = key.translate (string.maketrans ("",""), string.punctuation).lower ()
#             if key in word_counts.keys ():
#                 word_counts [key] = word_counts [key] + 1
#             else:
#                 word_counts [key] = 1
#         
# 
#     print word_counts
# 
# word_count()
# 

# # MapReduce Code
# 
# import sys
# 
# import string
# 
# import logging
# 
# from util import mapper_logfile
# logging.basicConfig(filename=mapper_logfile, format='%(message)s',
#                     level=logging.INFO, filemode='w')
# 
# def mapper():
# 
#     #Also make sure to fill out the reducer code before clicking "Test Run" or "Submit".
# 
#     #Each line will be a comma-separated list of values. The
#     #header row WILL be included. Tokenize each row using the 
#     #commas, and emit (i.e. print) a key-value pair containing the 
#     #district (not state) and Aadhaar generated, separated by a tab. 
#     #Skip rows without the correct number of tokens and also skip 
#     #the header row.
# 
#     #You can see a copy of the the input Aadhaar data
#     #in the link below:
#     #https://www.dropbox.com/s/vn8t4uulbsfmalo/aadhaar_data.csv
# 
#     #Since you are printing the output of your program, printing a debug 
#     #statement will interfere with the operation of the grader. Instead, 
#     #use the logging module, which we've configured to log to a file printed 
#     #when you click "Test Run". For example:
#     #logging.info("My debugging message")
#     #
#     #Note that, unlike print, logging.info will take only a single argument.
#     #So logging.info("my message") will work, but logging.info("my","message") will not.
# 
#     for line_num, line in enumerate(sys.stdin):
#         #your code here
#         if line_num:
#             data = line.strip().split(',')
#             if len (data) == 12:
#                 print '{0}\t{1}'.format (data [3],  data [8])
#         
# mapper()
# 
# import sys
# 
# import logging
# 
# from util import reducer_logfile
# logging.basicConfig(filename=reducer_logfile, format='%(message)s',
#                     level=logging.INFO, filemode='w')
# 
# def reducer():
#     
#     aadhaar_generated = 0
#     old_key = None
#     
#     #Also make sure to fill out the mapper code before clicking "Test Run" or "Submit".
# 
#     #Each line will be a key-value pair separated by a tab character.
#     #Print out each key once, along with the total number of Aadhaar 
#     #generated, separated by a tab. Make sure each key-value pair is 
#     #formatted correctly! Here's a sample final key-value pair: 'Gujarat\t5.0'
# 
#     #Since you are printing the output of your program, printing a debug 
#     #statement will interfere with the operation of the grader. Instead, 
#     #use the logging module, which we've configured to log to a file printed 
#     #when you click "Test Run". For example:
#     #logging.info("My debugging message")
#     #Note that, unlike print, logging.info will take only a single argument.
#     #So logging.info("my message") will work, but logging.info("my","message") will not.
#         
#     for line_num, line in enumerate(sys.stdin):
#         # your code here
#         if line_num:
#             data = line.strip().split ('\t')
#             if len (data) == 2:
#                 this_key, count = data
#                 if old_key and old_key != this_key:
#                     print "{0}\t{1}".format(old_key, aadhaar_generated)
#                     aadhaar_generated = 0
#                 
#                 old_key = this_key
#                 aadhaar_generated += float(count)
#                 
#     if old_key != None:
#         print "{0}\t{1}".format(old_key, aadhaar_generated)
#                 
# 
# reducer()
# 
# 

# # debugg information in python
# 
# from util import mapper_logfile
# 
# logging.basicConfig(filename=mapper_logfile, format='%(message)s',
#                     level=logging.INFO, filemode='w')
# 

# # mapreduce-> calculate average
# 
# 
# import sys
# 
# import logging
# 
# from util import reducer_logfile
# 
# logging.basicConfig(filename=reducer_logfile, format='%(message)s',
#                     level=logging.INFO, filemode='w')
# 
# def reducer():
#     '''
#     Given the output of the mapper for this assignment, the reducer should
#     print one row per weather type, along with the average value of
#     ENTRIESn_hourly for that weather type, separated by a tab. You can assume
#     that the input to the reducer will be sorted by weather type, such that all
#     entries corresponding to a given weather type will be grouped together.
# 
#     In order to compute the average value of ENTRIESn_hourly, you'll need to
#     keep track of both the total riders per weather type and the number of
#     hours with that weather type. That's why we've initialized the variable 
#     riders and num_hours below. Feel free to use a different data structure in 
#     your solution, though.
# 
#     An example output row might look like this:
#     'fog-norain\t1105.32467557'
# 
#     Since you are printing the output of your program, printing a debug 
#     statement will interfere with the operation of the grader. Instead, 
#     use the logging module, which we've configured to log to a file printed 
#     when you click "Test Run". For example:
#     logging.info("My debugging message")
#     Note that, unlike print, logging.info will take only a single argument.
#     So logging.info("my message") will work, but logging.info("my","message") will not.
#     '''
# 
#     riders = 0      # The number of total riders for this key
#     num_hours = 0   # The number of hours with this key
#     key = None
# 
#     for line in sys.stdin:
#         # your code here
#         data = line.strip ().split ('\t')
#         if len (data) == 2:
#             this_key, this_num_hours = data
#             if key and key != this_key:
#                 print '{0}\t{1}'.format (key, num_hours/riders)
#                 riders = 0
#                 num_hours = 0
#                 
#             key = this_key
#             riders += 1
#             num_hours += float (this_num_hours)
#             
#     if key != None:
#         print '{0}\t{1}'.format (key, num_hours/riders) 
# 
# reducer()
# 

# In[ ]:



