import os
import yfinance as yf

def get_stock_data(ticker, start_date, end_date, path):
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

    data.to_csv(os.path.join(path, "stock_data.csv"), index=False)


if __name__ == "__main__":
    ticker = "BRK-B"
    start_date = "2006-01-01"
    end_date = "2024-09-30"
    # stock_data = get_stock_data(ticker, start_date, end_date)
    # stock_data.to_csv(f"{ticker}_stock_data.csv", index=False)

    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    data.reset_index(inplace=True)
    data = data[['Date', 'Open', 'High', 'Low', 'Close']]

    data.to_csv("stock_data.csv", index=False)
    