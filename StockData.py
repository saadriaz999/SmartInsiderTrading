import yfinance as yf

def get_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a specific ticker symbol.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Dataframe with stock market data.
    """

    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    data.reset_index(inplace=True)
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return data

if __name__ == "__main__":
    ticker = "CVS"
    start_date = "2006-01-01"
    end_date = "2024-09-30"
    stock_data = get_stock_data(ticker, start_date, end_date)
    stock_data.to_csv(f"{ticker}_stock_data.csv", index=False)
    