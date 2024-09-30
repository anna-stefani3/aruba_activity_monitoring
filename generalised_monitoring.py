class GeneralisedMonitoring:
    def __init__(self):
        """
        Initialize the monitor with optional custom thresholds for each activity.
        Default thresholds are defined if not provided.
        """
        self.thresholds = {
            "sleep_min": 240,  # 4 hours in minutes
            "sleep_max": 720,  # 12 hours in minutes
            "sleep_disturbances_max": 3,
            "eating_min": 1,
            "eating_max": 6,
            "meal_prep_min": 1,
            "wake_up_min": 5,  # 5 AM
            "wake_up_max": 10,  # 10 AM
            "sleep_start_min": 21,  # 9 PM
            "sleep_start_max": 2,  # 2 AM
        }

    def check_sleep_duration(self, sleep_minutes):
        """
        Check if sleep duration is within normal human range (4-12 hours).
        """
        if sleep_minutes < self.thresholds["sleep_min"]:
            return "Sleep Deprivation (Under 4 Hours)"
        elif sleep_minutes > self.thresholds["sleep_max"]:
            return "Oversleeping (More than 12 Hours)"
        return "Normal Sleep Range"

    def check_sleep_disturbances(self, sleep_disturbances):
        """
        Check if sleep disturbances are higher than expected (e.g., more than 3 disturbances).
        """
        if sleep_disturbances > self.thresholds["sleep_disturbances_max"]:
            return "Poor Sleep Quality (More than 3 disturbances)"
        return "Normal Sleep Quality"

    def check_eating_events(self, eating_count):
        """
        Check if the number of eating events is within the normal range.
        """
        if eating_count < self.thresholds["eating_min"]:
            return "No Eating Events Detected (Possible missed meals)"
        elif eating_count > self.thresholds["eating_max"]:
            return "Excessive Eating Events (More than 6)"
        return "Normal Eating Pattern"

    def check_meal_preparation(self, meal_preparation_count):
        """
        Check if meal preparation events are happening regularly.
        """
        if meal_preparation_count < self.thresholds["meal_prep_min"]:
            return "No Meal Preparation Detected"
        return "Normal Meal Preparation"

    def check_wake_up_time(self, wake_up_time):
        """
        Check if the wake-up time is within a reasonable range (e.g., 5 AM - 10 AM).
        """
        if wake_up_time is not None:
            if wake_up_time < self.thresholds["wake_up_min"] or wake_up_time > self.thresholds["wake_up_max"]:
                return "Abnormal Wake-up Time (Outside 5 AM to 10 AM)"
            return "Normal Wake-up Time"
        return "Wake-up Time Not Available"

    def check_sleep_start_time(self, sleep_start_time):
        """
        Check if the sleep start time is within a reasonable range (e.g., 9 PM - 2 AM).
        """
        if sleep_start_time is not None:
            if sleep_start_time < self.thresholds["sleep_start_min"] and sleep_start_time >= 0:
                return "Abnormal Early Sleep Time (Before 9 PM)"
            elif sleep_start_time > self.thresholds["sleep_start_max"]:
                return "Abnormal Late Sleep Time (After 2 AM)"
            return "Normal Sleep Start Time"
        return "Sleep Start Time Not Available"

    def highlight_risks(self, day_data):
        """
        Apply rule-based checks to a single day's data and return only those with risks.
        """
        day_result = {}

        # Apply each rule-based check
        sleep_check = self.check_sleep_duration(day_data["sleep_count"])
        if "Deprivation" in sleep_check or "Oversleeping" in sleep_check:
            day_result["Sleep Check"] = sleep_check

        sleep_disturbance_check = self.check_sleep_disturbances(day_data["sleep_disturbances"])
        if "Poor Sleep Quality" in sleep_disturbance_check:
            day_result["Sleep Disturbance Check"] = sleep_disturbance_check

        eating_check = self.check_eating_events(day_data["eating_count"])
        if "No Eating Events" in eating_check or "Excessive Eating" in eating_check:
            day_result["Eating Check"] = eating_check

        meal_preparation_check = self.check_meal_preparation(day_data["meal_preparation_count"])
        if "No Meal Preparation" in meal_preparation_check:
            day_result["Meal Preparation Check"] = meal_preparation_check

        wake_up_check = self.check_wake_up_time(day_data["wake_up_time"])
        if "Abnormal" in wake_up_check:
            day_result["Wake-up Check"] = wake_up_check

        sleep_start_time_check = self.check_sleep_start_time(day_data["sleep_start_time"])
        if "Abnormal" in sleep_start_time_check:
            day_result["Sleep Start Time Check"] = sleep_start_time_check

        return day_result


# # Usage of RuleBasedMonitor for real-time simulation
# rule_monitor = GeneralisedMonitoring()

# # Simulate processing one row (one day) at a time
# for idx, row in df_daily_stats.iterrows():
#     print(f"Monitoring for Date: {idx}")
#     daily_report = rule_monitor.monitor_one_day(row)

#     # Display issues for the current day
#     for key, value in daily_report.items():
#         print(f"{key}: {value}")
#     print("\n")
