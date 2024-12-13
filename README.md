# Stock Prediction using Insider Data

Data Sources (the processed data for companies can be found in the 'data/companies' folder in the repo):
- yfinance: https://pypi.org/project/yfinance/ This python package was used to download the daily market data for each company
- US Security and Exchange Commission: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets This site was used to get quarterly data on inside trading

The project currently has the following python scripts:
- InsiderData.py: Cleans and processes insider data for a company
- StockData.py: Downloads stock data for a company
- DownloadAllData.py: Uses the above two files to form a pipeline to prepare data for each company
- Models: PyTorch model architectures of the two approaches used
- PreprocessingDataUtils: Utility functions to train data using models defined
- TrainModels: Trains the two models for each company whose data has been preprocessed
