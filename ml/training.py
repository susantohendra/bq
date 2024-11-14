from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# Initialize BigQuery client
client = bigquery.Client()

# Define the expiration dates for ES Futures contracts
expiration_dates = {
    'ESH9': '2019-03-15',
    'ESM9': '2019-06-21',
    'ESU9': '2019-09-20',
    'ESZ9': '2019-12-20',
    'ESH0': '2020-03-20',
    'ESM0': '2020-06-19',
    'ESU0': '2020-09-18',
    'ESZ0': '2020-12-18',
    'ESH1': '2021-03-19',
    'ESM1': '2021-06-18',
    'ESU1': '2021-09-17',
    'ESZ1': '2021-12-17',
    'ESH2': '2022-03-18',
    'ESM2': '2022-06-17',
    'ESU2': '2022-09-16',
    'ESZ2': '2022-12-16',
    'ESH3': '2023-03-17',
    'ESM3': '2023-06-16',
    'ESU3': '2023-09-15',
    'ESZ3': '2023-12-15',
    'ESH4': '2024-03-15',
    'ESM4': '2024-06-21',
    'ESU4': '2024-09-20',
    'ESZ4': '2024-12-20'
}

# Function to determine the front month contract based on the current date
def get_front_month_contract(current_date):
    for contract, exp_date in expiration_dates.items():
        exp_date = datetime.strptime(exp_date, '%Y-%m-%d')
        if current_date <= exp_date - timedelta(days=10):
            return contract
    return 'ESZ4'  # Default to the last contract if no match found

# Define the begin_date and end_date variables
begin_date = '2019-01-01'
end_date = '2024-11-13'

# Get the current date
current_date = datetime.strptime(begin_date, '%Y-%m-%d')

# Initialize an empty DataFrame to store the minutebar data
all_minute_bars = pd.DataFrame()

# Loop through each day from the beginning of the year to the end date
with tqdm(total=(datetime.strptime(end_date, '%Y-%m-%d') - current_date).days + 1) as pbar:
    while current_date <= datetime.strptime(end_date, '%Y-%m-%d'):
        # Determine the front month contract for the current date
        # Determine the front month contract for the current date
        front_month_contract = get_front_month_contract(current_date)
        
        
        # Determine the expiration date for the front month contract
        front_month_expiration = expiration_dates[front_month_contract]
        front_month_expiration = datetime.strptime(front_month_expiration, '%Y-%m-%d')
        
        # Determine the start date and end date for the quarter
        start_date = current_date.strftime('%Y-%m-%d')
        end_date_qtr = (front_month_expiration - timedelta(days=10)).strftime('%Y-%m-%d')
        
        # Query minutebar data for the quarter
        query_template = """
        SELECT 
            ts_event,
            open,
            high,
            low,
            close,
            volume,
            symbol
        FROM `peppy-arcadia-437101-r3.bar_data.databento_minutebar`
        WHERE symbol = '{symbol}'
            AND DATE(ts_event) >= '{start_date}'
            AND DATE(ts_event) <= '{end_date}'
        ORDER BY ts_event
        """
        
        query = query_template.format(symbol=front_month_contract, start_date=start_date, end_date=end_date_qtr)
        query_job = client.query(query)
        minute_bars = query_job.to_dataframe()
        
        # Append the minutebar data to the all_minute_bars DataFrame
        all_minute_bars = pd.concat([all_minute_bars, minute_bars])
        
        # Move to the next day
        tqdm.write(f"Processed data for {current_date.strftime('%Y-%m-%d')}")
        pbar.update((datetime.strptime(end_date_qtr, '%Y-%m-%d') - current_date).days + 1)
        current_date = datetime.strptime(end_date_qtr, '%Y-%m-%d') + timedelta(days=1)
        # print(current_date)
        # print(end_date)

# Your processing code goes here
# For example, you can save the data to a CSV file
print(all_minute_bars)
all_minute_bars.to_csv('es_minutebar_data.csv', index=False)