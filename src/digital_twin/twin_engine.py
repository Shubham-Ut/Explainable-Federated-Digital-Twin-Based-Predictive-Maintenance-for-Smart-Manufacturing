from src.digital_twin.health_score import calculate_health_score
from src.digital_twin.machine_status import get_machine_status, get_failure_risk


def create_digital_twin(engine_id, predicted_rul, max_rul):
    """
    Create digital twin output for one machine/engine.
    """

    health_score = calculate_health_score(predicted_rul, max_rul)
    machine_status = get_machine_status(predicted_rul)
    failure_risk = get_failure_risk(predicted_rul)

    twin_data = {
        "Engine ID": engine_id,
        "Predicted RUL": round(float(predicted_rul), 2),
        "Health Score (%)": health_score,
        "Machine Status": machine_status,
        "Failure Risk": failure_risk,
    }

    return twin_data