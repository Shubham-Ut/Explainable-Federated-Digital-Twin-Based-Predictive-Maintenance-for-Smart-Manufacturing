def calculate_health_score(predicted_rul, max_rul):
    """
    Convert predicted RUL into health score percentage.
    """

    if max_rul <= 0:
        return 0

    score = (predicted_rul / max_rul) * 100

    if score > 100:
        score = 100

    if score < 0:
        score = 0

    return round(score, 2)