import pandas as pd
import json

from sleep import get_monitoring_stats


# Load the data
df = pd.read_csv("[ARUBA]-activities_fixed_interval_data.csv")

monitoring_stats = {}
activity = "Sleeping"

# Monitor sleeping activity
monitoring_stats[activity] = get_monitoring_stats(df, activity)

# Save the MONITORING data as a JSON file
with open("sleep_monitoring_stats.json", "w") as json_file:
    json.dump(monitoring_stats, json_file, indent=4)
