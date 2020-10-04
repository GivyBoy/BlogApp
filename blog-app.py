
import pandas as pd
from pandas_datareader import data as web
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
plt.style.use('seaborn-whitegrid')
st.set_option('deprecation.showPyplotGlobalUse', False)


st.write("""# Portfolio Optimization""")

if st.checkbox('Tap/Click to see examples of stock tickers'):
  examples = pd.read_csv('stocks.csv')
  st.write(examples)

investment = st.slider('Investment Amount', 0, 10000, step=100)
st.write(f'The sum of money being invested is: ${investment} USD')

data= st.text_input('Enter the stock tickers here: ').upper()
if st.button('Submit'):
  for datum in data:
    if datum == ',':
        data = data.replace(datum, '')
  st.success(f'Your portfolio consists of: {data}; the value is: ${investment} USD ')

stocks = data.split()

stock_data = pd.DataFrame()

start_date = '2015-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
for stock in stocks:
    try:
        stock_data[stock] = web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)['Adj Close']
    except:
        st.write('Enter the tickers(for example, AAPL for Apple) for the stocks.')
        break
   
if len(stock_data.columns) > 0:  
  weights = np.array([1/len(stock_data.columns) for i in range(1, len(stock_data.columns)+1)])
else:
  weights = []

  
for i in stock_data.columns.values:
   plt.plot(stock_data[i])
   plt.title(f"{i}'s Stock Price ($USD)")
   plt.xlabel('Date')
   plt.ylabel('Stock Price ($USD)')
   plt.legend([i])
   st.pyplot()

if len(stock_data.columns) > 0:    
  dsr = stock_data.pct_change()
  dsr = round(dsr, 4)*100
  dsrUpdated = dsr.copy()

  plt.figure(figsize=(10, 6))
  plt.plot(dsr)
  plt.title('Volatility of the Individual Stocks')
  plt.xlabel('Date')
  plt.ylabel('Volatility (%)')
  plt.legend(dsr, bbox_to_anchor=(1,1))
  st.pyplot()
  
  DailyReturns = pd.DataFrame()
  DailyReturns = dsrUpdated.sum(axis=1)
  dsrUpdated['Daily Returns'] = DailyReturns
  
  port = [122]
  port = pd.DataFrame(port)

  dsrUpdated['Portfolio Value'] = port
  dsrUpdated.iloc[0, -1,] = investment

  for i in range(1, len(stock_data.index)):
    dsrUpdated['Portfolio Value'][i] = dsrUpdated['Portfolio Value'][i-1] + dsrUpdated['Daily Returns'][i]

  plt.figure(figsize=(10, 6))
  plt.plot(dsrUpdated['Portfolio Value'])
  plt.title('Portfolio Value Over Time')
  plt.xlabel('Date')
  plt.ylabel('Portfolio Value ($USD)')
  st.pyplot()
  
  # generate a w random weight of length of number of stocks
  def generate():
      number_of_portfolios = 2000
      RF = 0
      portfolio_returns = []
      portfolio_risk = []
      sharpe_ratio_port = []
      portfolio_weights = []

      for portfolio in range(number_of_portfolios):
          # generate a w random weight of length of number of stocks
          weights = np.random.random_sample(len(stock_data.columns))

          weights = weights / np.sum(weights)
          annualize_return = np.sum((dsr.mean() * weights) * 252)
          portfolio_returns.append(annualize_return)
          # variance
          matrix_covariance_portfolio = (dsr.cov()) * 252
          portfolio_variance = np.dot(weights.T, np.dot(matrix_covariance_portfolio, weights))
          portfolio_standard_deviation = np.sqrt(portfolio_variance)
          portfolio_risk.append(portfolio_standard_deviation)
          # sharpe_ratio
          sharpe_ratio = ((annualize_return - RF) / portfolio_standard_deviation)
          sharpe_ratio_port.append(sharpe_ratio)

          portfolio_weights.append(weights)

      portfolio_risk = np.array(portfolio_risk)
      portfolio_returns = np.array(portfolio_returns)
      sharpe_ratio_port = np.array(sharpe_ratio_port)

      porfolio_metrics = [portfolio_returns, portfolio_risk, sharpe_ratio_port, portfolio_weights]

      portfolio_dfs = pd.DataFrame(porfolio_metrics)
      portfolio_dfs = portfolio_dfs.T
      portfolio_dfs.columns = ['Port Returns', 'Port Risk', 'Sharpe Ratio', 'Portfolio Weights']

      # convert from object to float the first three columns.
      for col in ['Port Returns', 'Port Risk', 'Sharpe Ratio']:
          portfolio_dfs[col] = portfolio_dfs[col].astype(float)

      # portfolio with the highest Sharpe Ratio
      Highest_sharpe_port = portfolio_dfs.iloc[portfolio_dfs['Sharpe Ratio'].idxmax()]
      # portfolio with the minimum risk
      min_risk = portfolio_dfs.iloc[portfolio_dfs['Port Risk'].idxmin()]

      def plot_data_with_indicators():
          plt.figure(figsize=(10, 5))
          plt.scatter(portfolio_risk, portfolio_returns, c=portfolio_returns/portfolio_risk,cmap='YlGnBu', alpha=0.9)
          plt.xlabel('Volatility (%)')
          plt.ylabel('Returns (%)')
          plt.colorbar(label='Sharpe ratio')
          plt.scatter(Highest_sharpe_port['Port Risk'],Highest_sharpe_port['Port Returns'], marker=(3,1,0),color='g',s=100, label='Maximum Sharpe ratio' )
          plt.scatter(min_risk['Port Risk'], min_risk['Port Returns'], marker=(3,1,0), color='r',s=100, label='Minimum volatility')
          plt.legend(labelspacing=0.8)
          plt.title('Portfolio Performance')
          st.pyplot()

      def plot_data():
          plt.figure(figsize=(10, 5))
          plt.scatter(portfolio_risk, portfolio_returns, c=portfolio_returns/portfolio_risk,cmap='YlGnBu', alpha=0.9)
          plt.xlabel('Volatility (%)')
          plt.ylabel('Returns (%)')
          plt.title('Portfolio Performance')
          plt.colorbar(label='Sharpe ratio')
          st.pyplot()

      def data():
          st.write('-----------------------------------------------------------------------')
          st.write(f"To maximize returns, your Portfolio should consist of: ")
          for i in range(len(Highest_sharpe_port['Portfolio Weights'])):
            st.write(f"{round((Highest_sharpe_port['Portfolio Weights'][i]*100), 2)}% of {stocks[i]}")
          st.write(f"Your Portfolio Risk is: {round(Highest_sharpe_port['Port Risk'], 2)}%")
          st.write(f"Your Sharpe Ratio is: {round(Highest_sharpe_port['Sharpe Ratio'], 2)}")
          st.write('-----------------------------------------------------------------------')
          st.write(f"To minimize risk, your Portfolio should consist of: ")
          for i in range(len(min_risk['Portfolio Weights'])):
            st.write(f"{round((min_risk['Portfolio Weights'][i]*100), 2)}% of {stocks[i]}")
          st.write(f"Your Portfolio Risk is: {round(min_risk['Port Risk'], 2)}%")
          st.write(f"Your Sharpe Ratio is: {round(min_risk['Sharpe Ratio'], 2)}")
          st.write('-----------------------------------------------------------------------')

      st.write("""## Efficient Markets Frontier""")
      st.write(f'The number of portfolios being used is: {number_of_portfolios}')             

      plot_data()
      plot_data_with_indicators()
      data()

  generate()               
                   
else:
    st.write()
    
