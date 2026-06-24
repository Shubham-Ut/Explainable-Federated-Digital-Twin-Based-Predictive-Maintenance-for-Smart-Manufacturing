import plotly.express as px

def plot_sensor(
    df,
    sensor
):

    fig = px.line(
        df,
        x="cycle",
        y=sensor,
        title=f"{sensor} Trend"
    )

    return fig

