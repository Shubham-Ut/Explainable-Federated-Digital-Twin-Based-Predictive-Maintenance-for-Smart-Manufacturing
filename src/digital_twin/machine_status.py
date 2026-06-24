def get_machine_status(predicted_rul):
    """
    Convert predicted RUL into machine status.
    """

    if predicted_rul > 80:
        return "Healthy"

    elif predicted_rul > 30:
        return "Warning"

    else:
        return "Critical"


def get_failure_risk(predicted_rul):
    """
    Convert predicted RUL into failure risk level.
    """

    if predicted_rul > 80:
        return "Low"

    elif predicted_rul > 30:
        return "Medium"

    else:
        return "High"