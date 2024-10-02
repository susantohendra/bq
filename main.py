from google.cloud import bigquery
import pandas as pd

import os
os.environ["GCLOUD_PROJECT"] = "peppy-arcadia-437101-r3"

# Initialize BigQuery client
client = bigquery.Client()
# Define the begin_date and end_date variables
begin_date = '2024-09-23'
end_date = '2024-09-27'

# Query daily bars data from BigQuery
query_template = """
DECLARE symbolInput STRING DEFAULT 'ESZ4';

SELECT 
    ts_event,
    open,
    high,
    low,
    close,
    symbol
FROM `peppy-arcadia-437101-r3.bar_data.databento_minutebar`
WHERE symbol = symbolInput
    AND DATE(ts_event) = '{date}'
ORDER BY ts_event
"""

# Process the daily bars data
for date in pd.date_range(begin_date, end_date):
    query = query_template.format(date=date.strftime('%Y-%m-%d'))
    query_job = client.query(query)
    daily_bars = query_job.to_dataframe()

    # Your processing code goes here
    minute_bars = daily_bars.copy()
    for i in range(1, len(minute_bars)):
        if i == 1:
            minute_bars.at[i, 'ha_open'] = minute_bars.at[i, 'open']
            minute_bars.at[i, 'ha_high'] = minute_bars.at[i, 'high']
            minute_bars.at[i, 'ha_low'] = minute_bars.at[i, 'low']
            minute_bars.at[i, 'ha_close'] = minute_bars.at[i, 'close']
        else:
            minute_bars.at[i, 'ha_open'] = (minute_bars.at[i-1, 'ha_open'] + minute_bars.at[i-1, 'ha_close']) / 2.0
            minute_bars.at[i, 'ha_high'] = max(minute_bars.at[i, 'high'], minute_bars.at[i, 'ha_open'])
            minute_bars.at[i, 'ha_low'] = min(minute_bars.at[i, 'low'], minute_bars.at[i, 'ha_open'])
            minute_bars.at[i, 'ha_close'] = (minute_bars.at[i, 'open'] + minute_bars.at[i, 'high']+ minute_bars.at[i, 'low']+ minute_bars.at[i, 'close']) / 4.0
    # Select relevant columns
    heikin_ashi_bars = minute_bars[['ts_event', 'ha_open', 'ha_high', 'ha_low', 'ha_close', 'symbol']]
    # Store the results back to BigQuery
    table_id = 'peppy-arcadia-437101-r3.bar_data.ha_minutebar'
    job = client.load_table_from_dataframe(heikin_ashi_bars, table_id)
    job.result()  # Wait for the job to complete
    print(f"Heikin Ashi bars for {date.strftime('%Y-%m-%d')} have been successfully stored in BigQuery.")
