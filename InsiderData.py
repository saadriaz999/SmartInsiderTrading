import os
import pandas as pd

import Constants


def process_directory(directory: str, base_dir: str) -> pd.DataFrame:
    """
    Process all the relevant files in a given directory, filter required columns, 
    and return a DataFrame.
    
    Args:
        directory (str): The directory name (e.g., "2006q1_form345").
        base_dir (str): The base directory path where the data is stored.
    
    Returns:
        pd.DataFrame: Processed data for the directory.
    """
    # Define file paths for the current directory
    file_paths = {
        "DERIV_HOLDING": os.path.join(base_dir, directory, "DERIV_HOLDING.tsv"),
        "NONDERIV_HOLDING": os.path.join(base_dir, directory, "NONDERIV_HOLDING.tsv"),
        "NONDERIV_TRANS": os.path.join(base_dir, directory, "NONDERIV_TRANS.tsv"),
        "OWNER_SIGNATURE": os.path.join(base_dir, directory, "OWNER_SIGNATURE.tsv"),
        "REPORTINGOWNER": os.path.join(base_dir, directory, "REPORTINGOWNER.tsv"),
        "SUBMISSION": os.path.join(base_dir, directory, "SUBMISSION.tsv")
    }
    
    # Start by reading the first file
    final_df = pd.read_csv(file_paths["DERIV_HOLDING"], sep="\t")
    
    # Loop through the rest of the files and merge them
    for file_name, file_path in file_paths.items():
        if file_name != "DERIV_HOLDING" and os.path.exists(file_path):  # Only merge if the file exists
            df = pd.read_csv(file_path, sep="\t")
            final_df = pd.merge(final_df, df, on='ACCESSION_NUMBER', how='outer')

    # Select required columns for filtering
    columns = ['FILING_DATE', 'ISSUERNAME', 'ISSUERTRADINGSYMBOL', 'RPTOWNERNAME', 'RPTOWNER_TITLE', 
               'TRANS_CODE', 'TRANS_PRICEPERSHARE', 'TRANS_SHARES', 'SHRS_OWND_FOLWNG_TRANS']
    filtered_df = final_df[columns]
    
    return filtered_df


def process_company_data(filtered_df: pd.DataFrame, company_code: str = None) -> pd.DataFrame:
    """
    Process the company data by filtering by company code (if provided), 
    calculating ownership changes, and formatting the DataFrame.
    
    Args:
        filtered_df (pd.DataFrame): The filtered DataFrame with necessary columns.
        company_code (str, optional): Company ticker code to filter by. Defaults to None.
    
    Returns:
        pd.DataFrame: Processed company data with additional columns and proper formatting.
    """
    # If company code is provided, filter by the ticker symbol
    if company_code:
        filtered_df = filtered_df[filtered_df['ISSUERTRADINGSYMBOL'] == company_code]
    
    # Convert 'FILING_DATE' to datetime format
    filtered_df['FILING_DATE'] = pd.to_datetime(filtered_df['FILING_DATE'], format='%d-%b-%Y', errors='coerce')
    
    # Sort values by filing date in descending order
    filtered_df = filtered_df.sort_values(by='FILING_DATE', ascending=False)
    
    # Adjust the transaction share for 'S' code to be negative
    filtered_df.loc[filtered_df['TRANS_CODE'] == 'S', 'TRANS_SHARES'] *= -1
    
    # Filter for transactions of type 'S' and 'P'
    filtered_df = filtered_df[filtered_df['TRANS_CODE'].isin(['S', 'P'])]
    
    # Calculate the percentage change in ownership
    filtered_df['PERCENTAGE_CHANGED_OWNED'] = (filtered_df['TRANS_SHARES'] / filtered_df['SHRS_OWND_FOLWNG_TRANS']) * 100
    
    # Calculate the total value of shares traded
    filtered_df['TOTAL_VALUE_CHANGED'] = filtered_df['TRANS_SHARES'] * filtered_df['TRANS_PRICEPERSHARE']
    
    # Reset the index
    filtered_df.reset_index(drop=True, inplace=True)
    
    # Rename the columns as required
    filtered_df.rename(columns={
        'FILING_DATE': 'filing_date',
        'ISSUERNAME': 'company_name',
        'ISSUERTRADINGSYMBOL': 'ticker',
        'RPTOWNERNAME': 'insider_name',
        'RPTOWNER_TITLE': 'job_title',
        'TRANS_CODE': 'trade_type',
        'TRANS_PRICEPERSHARE': 'price',
        'TRANS_SHARES': 'quantity_traded',
        'SHRS_OWND_FOLWNG_TRANS': 'final_shares_owned',
        'PERCENTAGE_CHANGED_OWNED': 'change_in_shares_owned',
        'TOTAL_VALUE_CHANGED': 'value_of_shares_traded'
    }, inplace=True)
    
    return filtered_df


def combine_and_save_results(base_dir: str, directories: list, company_code: str = None) -> None:
    """
    Combine data from all directories, process company data (if company_code is provided),
    and save the final results to a CSV file.
    
    Args:
        base_dir (str): The base directory path where the data is stored.
        directories (list): List of directories (e.g., ["2006q1_form345", "2006q2_form345", ...]).
        company_code (str, optional): Company ticker code to filter by. Defaults to None.
    """
    final_df_all = pd.DataFrame()
    
    # Iterate over each directory and process
    for directory in directories:
        # Process the directory data
        filtered_df = process_directory(directory, base_dir)
        
        # Process company data if company code is provided
        company_df = process_company_data(filtered_df, company_code)
        
        # Append to the final combined DataFrame
        final_df_all = pd.concat([final_df_all, company_df], ignore_index=True)
    
    # Sort the final DataFrame by filing date in increasing order
    final_df_all = final_df_all.sort_values(by='filing_date', ascending=True)
    
    # Save the final DataFrame to CSV
    final_df_all.to_csv(f"{company_code}_insider_data.csv", index=False)


if __name__ == "__main__":
    # Define the company code you want to filter by (optional)
    company_code = "CSCO"
    
    # Call the function to process and save results
    combine_and_save_results(
        base_dir=Constants.BASE_DIR,
        directories=Constants.QUARTER_DIRECTORIES,
        company_code=company_code
    )