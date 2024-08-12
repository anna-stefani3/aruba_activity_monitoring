import pandas as pd
import numpy as np


def get_monitoring_stats(df, activity_to_monitor):
    monitoring_stat = {}

    # Ensure the 'Time' column is in datetime format
    df["Time"] = pd.to_datetime(df["Time"])

    # Initialize variables for storing data
    sleep_register = {}
    current_day = None
    sleep_duration = 0
    sleep_interruptions_count = 0
    daily_counts = []
    last_activity = None

    # Iterate over each row to simulate live data feed
    for _, row in df.iterrows():
        day = row["Time"].date()

        # If the day has changed, store the previous day's data and reset
        if current_day != day:
            if current_day is not None:
                # Store duration and interruptions for the previous day
                sleep_register[str(current_day)] = {
                    "sleep_duration_count": sleep_duration,
                    "sleep_interruptions_count": sleep_interruptions_count,
                }
                daily_counts.append(sleep_duration)

                # Abnormal Sleep ALERT Trigger
                abnormal_sleep_duration_alert(str(current_day), sleep_duration, sleep_interruptions_count)

            # Reset for the new day
            current_day = day
            sleep_duration = 0
            sleep_interruptions_count = 0

        # Check for sleep_interruptions_count (activity stopped and started again)
        if row["activity"] == activity_to_monitor:
            if last_activity is not None and last_activity != activity_to_monitor:
                sleep_interruptions_count += 1
            sleep_duration += 1

        last_activity = row["activity"]

    # Store the data for the last day
    if current_day is not None:
        sleep_register[str(current_day)] = {
            "sleep_duration_count": sleep_duration,
            "sleep_interruptions_count": sleep_interruptions_count,
        }
        daily_counts.append(sleep_duration)

    # Calculate final metrics
    avg_sleep = np.mean(daily_counts)
    std_sleep = np.std(daily_counts)

    # Update the global MONITORING dictionary
    monitoring_stat = {
        "per_day_stat": sleep_register,
        "average_sleep_duration": int(avg_sleep),
        "sleep_duration_standard_deviation": int(std_sleep),
    }
    return monitoring_stat


def under_sleep_alert(date: str, sleep_duration: int, sleep_interruptions_count):
    if sleep_duration < 1 * 60:
        print(
            date,
            f": EXTREME UNDERSLEEP DETECTED [Duration : {sleep_duration}, Interruption : {sleep_interruptions_count}]",
        )
        return True
    if sleep_duration < 4 * 60:
        print(date, f": UNDERSLEEP DETECTED [Duration : {sleep_duration}, Interruption : {sleep_interruptions_count}]")
        return True

    return False


def over_sleep_alert(date: str, sleep_duration: int, sleep_interruptions_count):
    if sleep_duration > 16 * 60:
        print(
            date,
            f": EXTREME OVERSLEEP DETECTED [Duration : {sleep_duration}, Interruption : {sleep_interruptions_count}]",
        )
        return True
    if sleep_duration > 11 * 60:
        print(date, f": OVERSLEEP DETECTED [Duration : {sleep_duration}, Interruption : {sleep_interruptions_count}]")
        return True
    return False


def abnormal_sleep_duration_alert(date: str, sleep_duration: int, sleep_interruptions_count: int):
    over_sleep_alert(date, sleep_duration, sleep_interruptions_count)
    under_sleep_alert(date, sleep_duration, sleep_interruptions_count)
