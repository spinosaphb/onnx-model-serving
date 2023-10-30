import plotly.graph_objects as go


def create_inference_time_boxplot(inference_times):
    fig = go.Figure()
    fig.add_trace(go.Box(y=inference_times, name="Tempo de Inferência"))
    fig.update_layout(
        title="Distribuição de Tempo de Inferência",
        xaxis_title="Métricas",
        yaxis_title="Valor",
    )
    return fig


def create_resource_usage_boxplot(cpu_percentages, memory_percentages):
    fig = go.Figure()
    fig.add_trace(go.Box(y=cpu_percentages, name="Uso da CPU"))
    fig.add_trace(go.Box(y=memory_percentages, name="Uso de Memória"))
    fig.update_layout(
        title="Uso de Recursos do Sistema",
        xaxis_title="Métricas",
        yaxis_title="Valor",
    )
    return fig
