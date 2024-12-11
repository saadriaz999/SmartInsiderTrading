import os
import pathlib
import pandas as pd

from PreprocessingDataUtils import preprocess_data, train_and_evaluate

results = []
project_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent
companies_base_dir = project_path / "data" / "companies"
print(companies_base_dir)

for company in os.listdir(companies_base_dir):
    insider_path = os.path.join(companies_base_dir, company, "insider_data.csv")
    stock_path = os.path.join(companies_base_dir, company, "stock_data.csv")
    stock_data, insider_data, merged_data = preprocess_data(insider_path, stock_path)
    metrics = train_and_evaluate(stock_data, insider_data, merged_data)
    results.append({"company": company, **metrics})

# Print the results
results_df = pd.DataFrame(results)
results_df.to_csv('results.csv')
print(results_df)
