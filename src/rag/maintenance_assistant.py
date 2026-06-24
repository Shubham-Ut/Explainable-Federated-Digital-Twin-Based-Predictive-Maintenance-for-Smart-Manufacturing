from src.rag.document_loader import load_documents
from src.rag.simple_retriever import retrieve_relevant_documents


def generate_maintenance_recommendation(
    failure_risk,
    machine_status,
    predicted_rul,
    shap_reasons
):
    query = f"""
    Failure Risk: {failure_risk}
    Machine Status: {machine_status}
    Predicted RUL: {predicted_rul}
    Main Causes: {', '.join(shap_reasons)}
    """

    documents = load_documents()

    retrieved_docs = retrieve_relevant_documents(
        query=query,
        documents=documents,
        top_k=3
    )

    recommendation = []

    recommendation.append("AI Maintenance Recommendation")
    recommendation.append("--------------------------------")
    recommendation.append(f"Machine Status: {machine_status}")
    recommendation.append(f"Failure Risk: {failure_risk}")
    recommendation.append(f"Predicted RUL: {round(float(predicted_rul), 2)} cycles")
    recommendation.append("")
    recommendation.append("Main Factors Identified:")
    
    for reason in shap_reasons:
        recommendation.append(f"- {reason}")

    recommendation.append("")
    recommendation.append("Recommended Maintenance Actions:")

    if failure_risk == "High":
        recommendation.append("- Schedule maintenance immediately.")
        recommendation.append("- Stop or reduce machine load if abnormal behavior continues.")
    elif failure_risk == "Medium":
        recommendation.append("- Monitor the machine closely.")
        recommendation.append("- Plan maintenance in the next available maintenance window.")
    else:
        recommendation.append("- Continue normal operation.")
        recommendation.append("- Keep monitoring sensor trends.")

    recommendation.append("")

    if retrieved_docs:
        recommendation.append("Knowledge Retrieved from Maintenance Documents:")

        for doc in retrieved_docs:
            recommendation.append(f"\nSource: {doc['filename']}")
            recommendation.append(f"Relevance Score: {round(doc['score'], 3)}")
            recommendation.append(doc["content"])

    else:
        recommendation.append("No maintenance documents found.")

    return "\n".join(recommendation), retrieved_docs