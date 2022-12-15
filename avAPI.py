# Importing api requests
import requests
import json

# Importing Necessary Libraries for Predictive Modeling
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# Set base url for API
baseUrl = "https://www.alphavantage.co/query"
# Core Stock API Functions
def timeSeriesIntraday(symbol, interval):
    # Set base url for API
    baseUrl = "https://www.alphavantage.co/query"
    
    # Make API Request
    response = requests.get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": "OX9K2D1Q89P22TMN"
    })
    
    # Print Response
    print(response.text)
    
def timeSeriesIntradayExtendo(symbol, interval, yearMonth):
    # Set base url for API
    baseUrl = "https://www.alphavantage.co/query"
    
    # Make API Request
    response = requests.get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_INTRADAY_EXTENDED",
        "symbol": symbol,
        "interval": interval,
        "slice": yearMonth,
        "apikey": "OX9K2D1Q89P22TMN"
    })
    
    # Print Response
    print(response.text)

def timeSeriesDailyAdj(symbol):
    # Set base url for API
    baseUrl = "https://www.alphavantage.co/query"
    
    # Make API Request
    response = requests.get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": "OX9K2D1Q89P22TMN"
    })
    
    # Print Response
    print(response.text)
    
def timeSeriesWeekly(symbol):
    # Set base url for API
    baseUrl = "https://www.alphavantage.co/query"
    
    # Make the API Request
    response = requests.get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_WEEKLY",
        "symbol": symbol,
        "apikey": "OX9K2D1Q89P22TMN"
    })

    # Print Response
    print(response.text)
    
def timeSeriesWeeklyAdj(symbol):
    # Set base url for API
    baseUrl = "https://www.alphavantage.co/query"
    
    # Make the API Request
    response = requests.get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_WEEKLY_ADJUSTED",
        "symbol": symbol,
        "datatype": "JSON",
        "apikey": "OX9K2D1Q89P22TMN"
    })

    # Print Response
    return(response.text)
    
def timeSeriesMonthly(symbol):
    # Set base url for API
    baseUrl = "https://www.alphavantage.co/query"
    
    # Make the API Request
    response = requests.get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_MONTHLY",
        "symbol": symbol,
        "apikey": "OX9K2D1Q89P22TMN"
    })

    # Print Response
    print(response.text)
    
def timeSeriesMonthlyAdj(symbol):
    # Set base url for API
    baseUrl = "https://www.alphavantage.co/query"
    
    # Make the API Request
    response = requests.get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_MONTHLY_ADJUSTED",
        "symbol": symbol,
        "apikey": "OX9K2D1Q89P22TMN"
    })

    # Print Response
    print(response.text)



# Testing functions
# timeSeriesIntraday("AAPL", "60min")
# timeSeriesIntradayExtendo("AAPL", "60min", "year1month1")
#timeSeriesDailyAdj("RIOT")
#timeSeriesWeekly("IBM")
# timeSeriesWeeklyAdj("COIN")
#timeSeriesMonthly("AMD")
#timeSeriesMonthlyAdj("CAT")

# Preparing Data
# Weekly Data Taken From Alpha Vantage API
# Problem: Predicting future price of a stock based on its historical data
jsonFile = timeSeriesWeeklyAdj("COIN")

#Creating a Pandas DF and Cleaning the Data

data = json.loads(jsonFile)
data = data['Weekly Adjusted Time Series']
index = []
columns=[]
for key in data:
    index.append(key)
#print(index)
values = list(data.values())
#print(values)
innerDict = values[0]
for key in innerDict:
    columns.append(key)
#print(columns)
#print(data)
df = pd.DataFrame(values, columns=columns, index=index)
df['1. open'] = df['1. open'].astype(float)
df['2. high'] = df['2. high'].astype(float)
df['3. low'] = df['3. low'].astype(float)
df['4. close'] = df['4. close'].astype(float)
df['5. adjusted close'] = df['5. adjusted close'].astype(float)
df['6. volume'] = df['6. volume'].astype(float)
df['7. dividend amount'] = df['7. dividend amount'].astype(float)
df.head()

# Perform feature engineering to create new features that may be useful for prediction

df["avg_price"] = (df["2. high"] + df["3. low"]) / 2
df.head()

# Select a model: use linear regression
model = LinearRegression()

# Split the data into training and test sets
X = df[["avg_price"]]
y = df["4. close"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Use the trained model to make predictions on new data
new_data = pd.DataFrame({"avg_price": [100, 150, 200]})
predictions = model.predict(new_data)
print("Predictions:", predictions)

model = DecisionTreeRegressor()

X = df[["avg_price"]]
y = df["4. close"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Use the trained model to make predictions on new data
new_data = pd.DataFrame({"avg_price": [100, 150, 200]})
predictions = model.predict(new_data)
print("Predictions:", predictions)
