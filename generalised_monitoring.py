class GeneralisedMonitoring:
    def __init__(self):
        """
        Initialize the monitor with optional custom thresholds for each activity.
        Default thresholds are defined if not provided.
        """
        self.question_1 = "Do You Sleep well?"
        self.question_2 = "Does the person experience problems with sleeping?"
        self.question_3 = "What is person's general level of activity during the day?"
        self.question_4 = "Does the person have poor personal hygiene?"
        self.question_5 = "Does the person fail to eat a healthy diet?"
        self.question_6 = "Does the person's day lack structure?"

        self.thresholds = {
            "sleep_min": 6,
            "sleep_max": 10,
            "sleep_disturbances_max": 3,
            "eating_min": 1,
            "eating_max": 6,
            "cooking_min": 1,
            "wake_up_min": 5,  # 5 AM
            "wake_up_max": 10,  # 10 AM
            "sleep_start_min": 19,  # 7 PM
            "sleep_start_max": 1,  # 1 AM
        }
        self.LOWER_UPPER_LIMIT = {
            "sleep_duration": (0, 24),
            "sleep_disturbances": (0, 5),
            "sleep_start_time": (0, 24),
            "wake_up_time": (0, 24),
            "eating_count": (0, 5),
            "cooking_count": (0, 5),
            "active_duration": (0, 24),
        }

        self.PERFECT_RANGES = {
            "sleep_duration": (6, 10),
            "sleep_disturbances": (0, 1),
            "sleep_start_time": (21, 22),
            "wake_up_time": (5, 6),
            "eating_count": (2, 2),
            "cooking_count": (2, 2),
            "active_duration": (3, 4),
        }

        self.QUESTIONS_MAPPING = {
            "sleep_duration": (self.question_1, self.question_2),
            "sleep_disturbances": (self.question_1, self.question_2),
            "sleep_start_time": (self.question_1, self.question_2),
            "wake_up_time": (self.question_1, self.question_2),
            "eating_count": (self.question_5,),
            "cooking_count": (self.question_5,),
            "active_duration": (self.question_3,),
        }

    def calculate_score(self, feature, value):
        lower_limit, upper_limit = self.LOWER_UPPER_LIMIT[feature]
        lower, upper = self.PERFECT_RANGES[feature]
        # If value is within the [lower, upper] range, score is 1
        value = float(value)
        if lower <= value <= upper:
            return 1.0

        # If value is below the lower range
        if lower_limit <= value < lower:
            slope = 1 / (lower - lower_limit)  # Slope from lower_limit to lower
            return round(slope * (value - lower_limit), 1)

        # If value is above the upper range
        if upper < value <= upper_limit:
            slope = 1 / (upper_limit - upper)  # Slope from upper to upper_limit
            return round(1 - slope * (value - upper), 1)

        # If value is outside [lower_limit, upper_limit], return 0
        return 0.0

    def check_sleep_duration(self, sleep_duration):
        """
        Check if sleep duration is within normal human range (4-12 hours).
        """
        if sleep_duration < self.thresholds["sleep_min"]:
            return f"Sleep Deprivation ({round(sleep_duration,1)} Hours)"
        elif sleep_duration > self.thresholds["sleep_max"]:
            return f"Oversleeping ({round(sleep_duration,1)} Hours)"
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
        if meal_preparation_count < self.thresholds["cooking_min"]:
            return "No Cooking Detected"
        return "Cooking Normally"

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
        Check if the sleep start time is within a reasonable range (e.g., 7 PM - 1 AM).
        """
        if sleep_start_time is not None:
            if sleep_start_time < self.thresholds["sleep_start_min"] and sleep_start_time >= 12:
                return "Abnormal Early Sleep Time (Before 7 PM)"
            elif sleep_start_time > self.thresholds["sleep_start_max"] and sleep_start_time < 12:
                return "Sleeping Late Sleep Time (After 1 AM)"
            return "Normal Sleep Start Time"
        return "Sleep Start Time Not Available"

    def highlight_risks(self, day_data):
        """
        Apply rule-based checks to a single day's data and return only those with risks.
        """
        day_result = {}

        # Apply each rule-based check
        sleep_check = self.check_sleep_duration(day_data["sleep_duration"])
        if "Deprivation" in sleep_check or "Oversleeping" in sleep_check:
            day_result["sleep_duration"] = sleep_check

        sleep_disturbance_check = self.check_sleep_disturbances(day_data["sleep_disturbances"])
        if "Poor Sleep Quality" in sleep_disturbance_check:
            day_result["sleep_disturbances"] = sleep_disturbance_check

        eating_check = self.check_eating_events(day_data["eating_count"])
        if "No Eating Events" in eating_check or "Excessive Eating" in eating_check:
            day_result["eating_count"] = eating_check

        meal_preparation_check = self.check_meal_preparation(day_data["cooking_count"])
        if "No Meal Preparation" in meal_preparation_check:
            day_result["cooking_count"] = meal_preparation_check

        wake_up_check = self.check_wake_up_time(day_data["wake_up_time"])
        if "Abnormal" in wake_up_check:
            day_result["wake_up_time"] = wake_up_check

        sleep_start_time_check = self.check_sleep_start_time(day_data["sleep_start_time"])
        if "Abnormal" in sleep_start_time_check:
            day_result["sleep_start_time"] = sleep_start_time_check

        return day_result

    def get_scores(self, day_data):
        scores = {}
        scores["sleep_duration"] = self.calculate_score("sleep_duration", day_data["sleep_duration"])
        scores["sleep_disturbances"] = self.calculate_score("sleep_disturbances", day_data["sleep_disturbances"])
        scores["sleep_start_time"] = self.calculate_score("sleep_start_time", day_data["sleep_start_time"])
        scores["wake_up_time"] = self.calculate_score("wake_up_time", day_data["wake_up_time"])
        scores["eating_count"] = self.calculate_score("eating_count", day_data["eating_count"])
        scores["cooking_count"] = self.calculate_score("cooking_count", day_data["cooking_count"])
        scores["active_duration"] = self.calculate_score("active_duration", day_data["active_duration"])

        return scores


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
