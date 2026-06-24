import plotly.graph_objects as go


SENSOR_TO_COMPONENT = {
    "sensor_4": "Roller Section",
    "sensor_11": "Press Section",
    "sensor_14": "Hydraulic Section",
    "sensor_15": "Bearing Section",
    "sensor_21": "Conveyor Section",
}


COMPONENT_POSITIONS = {
    "Uncoiler Section": {"x": 0, "y": 0, "z": 0},
    "Roller Section": {"x": 2, "y": 0, "z": 0},
    "Press Section": {"x": 4, "y": 0, "z": 0},
    "Hydraulic Section": {"x": 6, "y": 0, "z": 0},
    "Bearing Section": {"x": 3, "y": 1.5, "z": 0},
    "Conveyor Section": {"x": 8, "y": 0, "z": 0},
}


def get_component_from_shap(shap_reasons):
    for feature in shap_reasons:
        for sensor, component in SENSOR_TO_COMPONENT.items():
            if sensor in feature:
                return component, sensor, feature

    return "General Machine Area", "Unknown", shap_reasons[0] if shap_reasons else "Unknown"


def get_color(component, fault_component, failure_risk):
    if component == fault_component:
        if failure_risk == "High":
            return "red"
        elif failure_risk == "Medium":
            return "orange"
        else:
            return "yellow"

    return "green"


def create_factory_3d_twin(shap_reasons, failure_risk):
    fault_component, fault_sensor, shap_feature = get_component_from_shap(shap_reasons)

    fig = go.Figure()

    for component, pos in COMPONENT_POSITIONS.items():
        color = get_color(component, fault_component, failure_risk)

        fig.add_trace(
            go.Scatter3d(
                x=[pos["x"]],
                y=[pos["y"]],
                z=[pos["z"]],
                mode="markers+text",
                marker=dict(
                    size=35,
                    color=color,
                    opacity=0.85
                ),
                text=[component],
                textposition="top center",
                name=component
            )
        )

    # connection line
    x_vals = [pos["x"] for pos in COMPONENT_POSITIONS.values()]
    y_vals = [pos["y"] for pos in COMPONENT_POSITIONS.values()]
    z_vals = [pos["z"] for pos in COMPONENT_POSITIONS.values()]

    fig.add_trace(
        go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode="lines",
            line=dict(width=6, color="gray"),
            name="Production Line"
        )
    )

    fig.update_layout(
        title=f"3D Factory Digital Twin | Fault Area: {fault_component}",
        scene=dict(
            xaxis_title="Production Flow",
            yaxis_title="Machine Zone",
            zaxis_title="Height"
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fault_info = {
        "Fault Component": fault_component,
        "Fault Sensor": fault_sensor,
        "SHAP Feature": shap_feature
    }

    return fig, fault_info