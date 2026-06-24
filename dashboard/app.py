import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.explainability.shap_analysis import get_single_prediction_explanation
from src.digital_twin.twin_engine import create_digital_twin
from src.rag.maintenance_assistant import generate_maintenance_recommendation
from src.llm.maintenance_report_generator import generate_ai_maintenance_report
from src.visualization.dashboard_plots import plot_sensor
from src.visualization.factory_3d_twin import create_factory_3d_twin


DATA_PATH = "data/processed/train_engineered.csv"
SHAP_IMAGE_PATH = "reports/figures/shap_summary.png"
RESULTS_DIR = "reports/results"


def extract_shap_reasons(explanation_df, top_n=5):
    return explanation_df.head(top_n)["Feature"].tolist()


def status_icon(value):
    if value in ["High", "Critical"]:
        return "🔴"
    if value in ["Medium", "Warning"]:
        return "🟠"
    return "🟢"


def build_component_table(fault_info, risk):
    components = [
        "Uncoiler Section",
        "Roller Section",
        "Press Section",
        "Hydraulic Section",
        "Bearing Section",
        "Conveyor Section",
    ]

    rows = []

    for component in components:
        status = "Healthy"
        icon = "🟢"

        if component == fault_info["Fault Component"]:
            if risk == "High":
                status = "Critical"
                icon = "🔴"
            elif risk == "Medium":
                status = "Warning"
                icon = "🟠"
            else:
                status = "Monitor"
                icon = "🟡"

        rows.append({
            "Component": component,
            "Status": f"{icon} {status}"
        })

    return pd.DataFrame(rows)


st.set_page_config(
    page_title="AI Maintenance Engineer",
    page_icon="⚙️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: #EAF7F4;
    }

    section[data-testid="stSidebar"] {
        background: #242235;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .main-title {
        background: #ffffff;
        padding: 18px 24px;
        border-radius: 4px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
        border-left: 6px solid #ff7f6e;
        margin-bottom: 18px;
    }

    .main-title h1 {
        color: #242235;
        font-size: 30px;
        font-weight: 900;
        margin-bottom: 3px;
        letter-spacing: 0.5px;
    }

    .main-title p {
        color: #64748b;
        font-size: 14px;
        margin: 0;
    }

    .metric-card {
        background: #ffffff;
        padding: 16px;
        border-radius: 4px;
        box-shadow: 0 5px 16px rgba(15, 23, 42, 0.08);
        border-top: 35px solid #242235;
        min-height: 120px;
    }

    .metric-label {
        margin-top: -42px;
        color: #ffffff;
        font-size: 12px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    .metric-value {
        margin-top: 22px;
        font-size: 28px;
        color: #242235;
        font-weight: 900;
    }

    .metric-sub {
        color: #64748b;
        font-size: 13px;
        margin-top: 2px;
    }

    .section-card {
        background: #ffffff;
        padding: 18px;
        border-radius: 4px;
        box-shadow: 0 5px 16px rgba(15, 23, 42, 0.08);
        margin-bottom: 18px;
    }

    .small-title {
        background: #242235;
        color: #ffffff;
        padding: 10px 14px;
        border-radius: 3px;
        font-size: 14px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 14px;
    }

    div[data-testid="stButton"] button {
        background: #ff7f6e;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 800;
    }

    div[data-testid="stButton"] button:hover {
        background: #ff6755;
        color: white;
    }

    div[data-testid="stDownloadButton"] button {
        background: #242235;
        color: white;
        border-radius: 4px;
        border: none;
        font-weight: 800;
    }

    div[data-testid="stTabs"] button {
        color: #242235;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">
        <h1>⚙️ AI Maintenance Engineer</h1>
        <p>Smart Manufacturing Digital Twin Dashboard | Predictive Maintenance | SHAP | RAG | Generative AI</p>
    </div>
    """,
    unsafe_allow_html=True
)

if not os.path.exists(DATA_PATH):
    st.error("train_engineered.csv not found. Please run Phase 3 first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.sidebar.title("☰ LOGOTYPE")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home Dashboard",
        "Digital Twin",
        "Explainability",
        "Maintenance Report",
        "Research Results",
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Machine Control")

engine_ids = sorted(df["engine_id"].unique())
selected_engine = st.sidebar.selectbox("Engine ID", engine_ids)

engine_data = df[df["engine_id"] == selected_engine]
cycle_list = sorted(engine_data["cycle"].unique())
selected_cycle = st.sidebar.selectbox("Cycle", cycle_list)

selected_sensor = st.sidebar.selectbox(
    "Sensor Trend",
    [
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_8",
        "sensor_9",
        "sensor_11",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_15",
        "sensor_20",
        "sensor_21",
    ],
    index=6
)

selected_row = engine_data[engine_data["cycle"] == selected_cycle]

if selected_row.empty:
    st.error("No data found for this engine and cycle.")
    st.stop()

row_index = selected_row.index[0]

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

if st.sidebar.button("🚀 Run AI Analysis", use_container_width=True):
    with st.spinner("Running AI Maintenance Analysis..."):

        predicted_rul, explanation_df = get_single_prediction_explanation(
            df,
            row_index=row_index,
            top_n=10
        )

        digital_twin_output = create_digital_twin(
            engine_id=selected_engine,
            predicted_rul=predicted_rul,
            max_rul=125
        )

        shap_reasons = extract_shap_reasons(explanation_df, top_n=5)

        fig_3d, fault_info = create_factory_3d_twin(
            shap_reasons=shap_reasons,
            failure_risk=digital_twin_output["Failure Risk"]
        )

        rag_recommendation, retrieved_docs = generate_maintenance_recommendation(
            failure_risk=digital_twin_output["Failure Risk"],
            machine_status=digital_twin_output["Machine Status"],
            predicted_rul=digital_twin_output["Predicted RUL"],
            shap_reasons=shap_reasons
        )

        ai_report = generate_ai_maintenance_report(
            digital_twin_output=digital_twin_output,
            shap_reasons=shap_reasons,
            retrieved_docs=retrieved_docs
        )

    st.session_state["analysis_done"] = True
    st.session_state["digital_twin_output"] = digital_twin_output
    st.session_state["explanation_df"] = explanation_df
    st.session_state["shap_reasons"] = shap_reasons
    st.session_state["fig_3d"] = fig_3d
    st.session_state["fault_info"] = fault_info
    st.session_state["rag_recommendation"] = rag_recommendation
    st.session_state["ai_report"] = ai_report

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(
        f"{RESULTS_DIR}/dashboard_ai_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(ai_report)

    st.success("Analysis completed successfully!")


if page == "Home Dashboard":

    if st.session_state["analysis_done"]:
        output = st.session_state["digital_twin_output"]
    else:
        output = {
            "Predicted RUL": 0,
            "Health Score (%)": 0,
            "Machine Status": "Not Run",
            "Failure Risk": "Not Run"
        }

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted RUL</div>
                <div class="metric-value">{output['Predicted RUL']:.2f}</div>
                <div class="metric-sub">cycles remaining</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Health Score</div>
                <div class="metric-value">{output['Health Score (%)']:.2f}%</div>
                <div class="metric-sub">machine condition</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Machine Status</div>
                <div class="metric-value">{status_icon(output['Machine Status'])} {output['Machine Status']}</div>
                <div class="metric-sub">current state</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Failure Risk</div>
                <div class="metric-value">{status_icon(output['Failure Risk'])} {output['Failure Risk']}</div>
                <div class="metric-sub">predicted risk</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">Percentage / RUL Trend</div>', unsafe_allow_html=True)

        fig_rul = px.line(
            engine_data,
            x="cycle",
            y="RUL",
            title="Remaining Useful Life Trend"
        )
        st.plotly_chart(fig_rul, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">Sensor Health Care</div>', unsafe_allow_html=True)

        fig_sensor = plot_sensor(engine_data, selected_sensor)
        st.plotly_chart(fig_sensor, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="small-title">Selected Machine Record</div>', unsafe_allow_html=True)
    st.dataframe(selected_row, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Digital Twin":

    if not st.session_state["analysis_done"]:
        st.warning("Run AI Analysis first from the sidebar.")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">3D Factory Digital Twin</div>', unsafe_allow_html=True)

        st.plotly_chart(
            st.session_state["fig_3d"],
            use_container_width=True
        )

        fault = st.session_state["fault_info"]

        f1, f2, f3 = st.columns(3)
        f1.info(f"Fault Component: {fault['Fault Component']}")
        f2.info(f"Fault Sensor: {fault['Fault Sensor']}")
        f3.info(f"SHAP Reason: {fault['SHAP Feature']}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">Component Health Status</div>', unsafe_allow_html=True)

        component_df = build_component_table(
            st.session_state["fault_info"],
            st.session_state["digital_twin_output"]["Failure Risk"]
        )

        st.dataframe(component_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


elif page == "Explainability":

    if not st.session_state["analysis_done"]:
        st.warning("Run AI Analysis first from the sidebar.")
    else:
        explanation_df = st.session_state["explanation_df"]

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">Top SHAP Explanation</div>', unsafe_allow_html=True)

        st.dataframe(explanation_df, use_container_width=True)

        top_exp = explanation_df.head(10)

        fig_shap = px.bar(
            top_exp,
            x="Absolute_Contribution",
            y="Feature",
            orientation="h",
            title="Top SHAP Feature Contributions"
        )

        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">Global SHAP Summary</div>', unsafe_allow_html=True)

        if os.path.exists(SHAP_IMAGE_PATH):
            st.image(SHAP_IMAGE_PATH, use_container_width=True)
        else:
            st.warning("SHAP summary image not found. Run Phase 4 first.")

        st.markdown("</div>", unsafe_allow_html=True)


elif page == "Maintenance Report":

    if not st.session_state["analysis_done"]:
        st.warning("Run AI Analysis first from the sidebar.")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">RAG Maintenance Recommendation</div>', unsafe_allow_html=True)
        st.text(st.session_state["rag_recommendation"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">AI Maintenance Engineer Report</div>', unsafe_allow_html=True)

        st.markdown(st.session_state["ai_report"])

        st.download_button(
            label="⬇️ Download AI Maintenance Report",
            data=st.session_state["ai_report"],
            file_name="ai_maintenance_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)


elif page == "Research Results":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="small-title">Model Performance Comparison</div>', unsafe_allow_html=True)

    model_results_path = "reports/results/model_comparison.csv"

    if os.path.exists(model_results_path):
        model_df = pd.read_csv(model_results_path)
        st.dataframe(model_df, use_container_width=True)

        fig_model = px.bar(
            model_df,
            x="Model",
            y="R2",
            title="Model R² Score Comparison"
        )
        st.plotly_chart(fig_model, use_container_width=True)
    else:
        st.warning("model_comparison.csv not found. Run Phase 3 first.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="small-title">Federated Client Results</div>', unsafe_allow_html=True)

    federated_path = "reports/results/federated_client_results.csv"

    if os.path.exists(federated_path):
        fed_df = pd.read_csv(federated_path)
        st.dataframe(fed_df, use_container_width=True)

        fig_fed = px.bar(
            fed_df,
            x="Client",
            y="R2",
            title="Federated Client R² Score"
        )
        st.plotly_chart(fig_fed, use_container_width=True)
    else:
        st.warning("federated_client_results.csv not found. Run Phase 6 first.")

    st.markdown("</div>", unsafe_allow_html=True)