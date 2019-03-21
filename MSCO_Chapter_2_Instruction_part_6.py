import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# This ingests the housing starts data
df = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='FRED_Graph', header=10, index_col=None, usecols=[0,1])

# This creates a totals column which shows the total for that year
df['year_total'] = df['HOUST1FQ'].groupby(df['observation_date'].dt.year).transform('sum')

# This creates a column which divides the housing starts for the given quarter by the total for that year
df['percent_of_year'] = df['HOUST1FQ']/df['year_total']

# These rows map a season value based on the month
season_map = {1:'Winter', 4:'Spring',7:'Summer',10:'Fall'}
df['season'] = df['observation_date'].dt.month.map(season_map)

# # These rows create the line plot for the last 20 years (last 80 periods)
    # last_20_years_df = df[-80:]
    # plt.plot(last_20_years_df['observation_date'],last_20_years_df['HOUST1FQ'])
    # plt.xlabel('Year')
    # plt.ylabel('Housing Starts')
    # plt.title('Housing starts by quarter, last 20 years, ending Fall 2018')
    # plt.show()

# # These rows create a bar plot colored by season for the last 20 full years
    # last_20_full_years_df = df[-83:-3]
    # mask1 = df['season'] == 'Winter'
    # mask2 = df['season'] == 'Spring'
    # mask3 = df['season'] == 'Summer'
    # mask4 = df['season'] == 'Fall'
    # plt.bar(last_20_full_years_df['observation_date'][mask1],last_20_full_years_df['percent_of_year'][mask1], width=np.timedelta64(75, 'D'), color='blue', label='Winter')
    # plt.bar(last_20_full_years_df['observation_date'][mask2],last_20_full_years_df['percent_of_year'][mask2], width=np.timedelta64(75, 'D'), color='green', label='Spring')
    # plt.bar(last_20_full_years_df['observation_date'][mask3],last_20_full_years_df['percent_of_year'][mask3], width=np.timedelta64(75, 'D'), color='red', label='Summer')
    # plt.bar(last_20_full_years_df['observation_date'][mask4],last_20_full_years_df['percent_of_year'][mask4], width=np.timedelta64(75, 'D'), color='orange', label='Fall')
    # plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=3,ncol=4, mode="expand", borderaxespad=0.)
    # plt.xlabel('Year')
    # plt.ylabel('Housing Starts')
    # plt.title('Housing starts by quarter, last 20 full years (1997-2017)', y=1.07)
    # plt.show()

# # This creates a line plot for the last nine full years
    # last_9_years_df = df[-39:-3]
    # print(last_9_years_df)
    # plt.plot(last_9_years_df['observation_date'],last_9_years_df['HOUST1FQ'])
    # plt.xlabel('Year')
    # plt.ylabel('Housing Starts')
    # plt.title('Housing starts by quarter, last 9 full years (2009-2017)')
    # plt.show()

# This creates a period column for the last nine full years
last_9_years_df = df[-39:-3]
last_9_years_df = last_9_years_df.reset_index()
last_9_years_df['Period'] = last_9_years_df.index + 1

# This calculates the OLS regression for the last nine full years
import statsmodels.api as sm
X = sm.add_constant(last_9_years_df['Period'])
y = last_9_years_df['HOUST1FQ']
results = sm.OLS(y,X).fit()

# This creates a column in the dataframe using the results of the OLS regression
c, m = results.params
last_9_years_df['prediction'] = c+m*last_9_years_df['Period']

# # This creates a line plot for the last nine full years PLUS the regression prediction
# plt.plot(last_9_years_df['observation_date'],last_9_years_df['HOUST1FQ'], label='Actual Demand')
# plt.plot(last_9_years_df['observation_date'],last_9_years_df['prediction'], color='red', label='OLS Regression')
# plt.xlabel('Year')
# plt.ylabel('Housing Starts')
# plt.title('Housing starts by quarter, last 9 full years (2009-2017)')
# plt.legend()
# plt.show()

# # This calculates the seasonal factor for each season and puts it in a nice table
# actual_demand = last_9_years_df['HOUST1FQ'].groupby(last_9_years_df['season']).sum()
# predicted_demand = last_9_years_df['prediction'].groupby(last_9_years_df['season']).sum()
# seasonal_factor = actual_demand/predicted_demand
# print(seasonal_factor)

# This calculates the seasonal factor for each season, adjusts the prediction by that amount, and tacks that on as a new column in the dataframe
actual_demand = last_9_years_df['HOUST1FQ'].groupby(last_9_years_df['season']).transform('sum')
predicted_demand = last_9_years_df['prediction'].groupby(last_9_years_df['season']).transform('sum')
last_9_years_df['adjusted_prediction'] = last_9_years_df['prediction']*(actual_demand/predicted_demand)

# # This plots the last nine years demand, regression prediction, as well as the adjusted prediction
# plt.plot(last_9_years_df['observation_date'],last_9_years_df['HOUST1FQ'], label='Actual Demand')
# plt.plot(last_9_years_df['observation_date'],last_9_years_df['prediction'], color='red', label='OLS Regression')
# plt.plot(last_9_years_df['observation_date'],last_9_years_df['adjusted_prediction'], color='green', label='Adjusted Prediction')
# plt.xlabel('Year')
# plt.ylabel('Housing Starts')
# plt.title('Housing starts by quarter, last 9 full years (2009-2017)')
# plt.legend()
# plt.show()

# This calculates and compares the mean squared error for each forecast method
from sklearn.metrics import mean_squared_error
mse_linear_regression = mean_squared_error(last_9_years_df['HOUST1FQ'], last_9_years_df['prediction'])
mse_seasonally_adjusted = mean_squared_error(last_9_years_df['HOUST1FQ'], last_9_years_df['adjusted_prediction'])
print('\nLinear regression MSE: \n', round(mse_linear_regression,2), '\n\nAdjusted prediction MSE:\n', round(mse_seasonally_adjusted,2))