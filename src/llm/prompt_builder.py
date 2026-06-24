def build_maintenance_prompt(
    digital_twin_output,
    shap_reasons,
    retrieved_docs
):
    knowledge_text = ""

    for doc in retrieved_docs:
        knowledge_text += f"\nSource: {doc['filename']}\n"
        knowledge_text += f"Content: {doc['content']}\n"

    prompt = f"""
You are an expert AI Maintenance Engineer for smart manufacturing systems.

Generate a professional maintenance report using the following information.

Machine Information:
Engine ID: {digital_twin_output["Engine ID"]}
Predicted RUL: {digital_twin_output["Predicted RUL"]} cycles
Health Score: {digital_twin_output["Health Score (%)"]}%
Machine Status: {digital_twin_output["Machine Status"]}
Failure Risk: {digital_twin_output["Failure Risk"]}

Main SHAP Factors:
{", ".join(shap_reasons)}

Retrieved Maintenance Knowledge:
{knowledge_text}

Prepare the report in this format:

1. Machine Health Summary
2. Root Cause Analysis
3. Risk Assessment
4. Recommended Maintenance Actions
5. Maintenance Priority
6. Final Decision

Use simple, professional, engineering-style language.
Do not make unsupported claims.
"""
    return prompt