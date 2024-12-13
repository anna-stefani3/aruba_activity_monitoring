class GeneralisedModel:
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

        self.FEATURE_MAPPING = {
            self.question_1: ("sleep_duration", "sleep_disturbances"),
            self.question_2: ("sleep_disturbances",),
            # self.question_3: {},
            # self.question_4: {},
            self.question_5: ("eating_count", "cooking_count"),
            # self.question_6: {},
        }

    def get_score_quality(self, score):
        if score <= 3:
            quality_range = "Poor"
        elif score <= 7:
            quality_range = "Normal"
        elif score <= 9:
            quality_range = "Good"
        else:
            quality_range = "Excellent"
        return quality_range

    def get_sleep_duration_score(self, sleep_duration: float) -> tuple[int, str]:
        if sleep_duration <= 3:
            score = 3  # Extreme insufficient sleep
        elif 3 < sleep_duration <= 6:
            score = 7  # Insufficient sleep
        elif 6 < sleep_duration <= 10:
            score = 10  # Normal sleep
        elif 10 < sleep_duration <= 14:
            score = 7  # Oversleeping
        else:
            score = 3  # Extreme oversleeping

        quality_range = self.get_score_quality(score)
        return score, quality_range

    def get_sleep_duration_label(self, sleep_duration: float) -> tuple[int, str]:
        if sleep_duration <= 3:
            label = "Extreme insufficient sleep"
        elif 3 < sleep_duration <= 6:
            label = "Insufficient sleep"
        elif 6 < sleep_duration <= 10:
            label = "Normal sleep"
        elif 10 < sleep_duration <= 14:
            label = "Oversleeping"
        else:
            label = "Extreme oversleeping"
        return label

    def get_sleep_disturbance_score(self, sleep_disturbances):
        if sleep_disturbances <= 2:
            score = 10
        elif sleep_disturbances <= 5:
            score = 5
        else:
            score = 0
        quality_range = self.get_score_quality(score)
        return score, quality_range

    def get_sleep_disturbance_label(self, sleep_disturbances):
        if sleep_disturbances <= 2:
            label = "Normal"
        elif sleep_disturbances <= 5:
            label = "Abnormal"
        else:
            label = "Emergency"
        return label

    def get_eating_count_score(self, eating_count: float) -> tuple[int, str]:
        if eating_count == 0:
            score = 0  # No Food taken
        elif eating_count == 1:
            score = 5  # Bare Minimum food Taken
        elif eating_count == 2:
            score = 8  # Normal Eating
        elif eating_count == 3:
            score = 10  # Perfect Eating
        elif eating_count == 4:
            score = 7  # Slight Over eating
        elif eating_count == 5:
            score = 4  # Over eating
        else:
            score = 1  # Extreme Over-eating

        quality_range = self.get_score_quality(score)
        return score, quality_range

    def get_cooking_count_score(self, cooking_count: float) -> tuple[int, str]:
        if cooking_count == 0:
            score = 0  # No Food cooked
        elif cooking_count == 1:
            score = 5  # Bare Minimum food cooked
        elif cooking_count == 2:
            score = 8  # Normal cooking
        elif cooking_count == 3:
            score = 10  # Perfect cooking
        elif cooking_count == 4:
            score = 7  # Slight Over cooking
        elif cooking_count == 5:
            score = 4  # Over cooking
        else:
            score = 1  # Extreme Over-cooking

        quality_range = self.get_score_quality(score)
        return score, quality_range

    def get_egrist_score(self, question, scores_dict) -> int:
        score = 0
        features_count = len(self.FEATURE_MAPPING[question])
        for feature in self.FEATURE_MAPPING[question]:
            score += scores_dict[feature]

        return score // features_count  # returns average score
