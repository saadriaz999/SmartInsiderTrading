import os
import pathlib
import Constants
from InsiderData import combine_and_save_results
from StockData import get_stock_data

project_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent

companies_base_dir = project_path / "data" / "companies"
insider_base_dir = project_path / "data" / "insider_data"


# Process each company in the Constants.COMPANIES list
for code in Constants.COMPANIES:
    # Create a directory for each company
    company_dir = companies_base_dir / code
    os.makedirs(company_dir, exist_ok=True)

    # Combine and save results for the company
    combine_and_save_results(
        base_dir=insider_base_dir,
        directories=Constants.QUARTER_DIRECTORIES,
        path=company_dir,
        company_code=code
    )

    # Fetch stock data for each company and save it
    start_date = "2006-01-01"
    end_date = "2024-09-30"

    # Fetch stock data
    stock_data = get_stock_data(code, start_date, end_date, company_dir)
