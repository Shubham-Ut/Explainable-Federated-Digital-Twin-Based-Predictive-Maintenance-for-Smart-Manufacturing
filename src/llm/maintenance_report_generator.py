from src.llm.prompt_builder import build_maintenance_prompt
from src.llm.gemini_client import get_gemini_response


def generate_ai_maintenance_report(
    digital_twin_output,
    shap_reasons,
    retrieved_docs
):
    prompt = build_maintenance_prompt(
        digital_twin_output=digital_twin_output,
        shap_reasons=shap_reasons,
        retrieved_docs=retrieved_docs
    )

    report = get_gemini_response(prompt)

    return report