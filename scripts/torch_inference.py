import torch
import time
import psutil
import plotly.graph_objects as go
import torchvision.datasets as datasets
from torchvision import transforms, models
import onnxruntime as ort
from tqdm import tqdm

from scripts import utils

model_path_pytorch = "workspace/models/resnet_model_100_epochs.pth"
pytorch_model = models.resnet50(pretrained=False)
pytorch_model.fc = torch.nn.Linear(pytorch_model.fc.in_features, 2)

pytorch_model.load_state_dict(torch.load(model_path_pytorch, map_location=torch.device('cpu')))

class_labels = ["cat", "dog"]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder("sample_data/cats-vs-dogs", transform=preprocess)

inference_times = []
cpu_percentages = []
memory_percentages = []

def run_pytorch_inference(model, data):
    inputs = data.unsqueeze(0)

    start_time_pytorch = time.time()
    pytorch_outputs = model(inputs)
    elapsed_time_pytorch = time.time() - start_time_pytorch

    return elapsed_time_pytorch

def collect_system_stats(label):
    cpu_percentages[label].append(psutil.cpu_percent())
    memory_percentages[label].append(psutil.virtual_memory().percent)

for image, _ in tqdm(dataset):
    elapsed_time_pytorch = run_pytorch_inference(pytorch_model, image)
    inference_times.append(elapsed_time_pytorch)
    collect_system_stats('pytorch')


def create_inference_time_box_plot(data, title, x_title, y_title):
    fig = go.Figure()
    fig.add_trace(go.Box(y=data, name=title))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title
    )
    return fig

def create_resource_usage_box_plot(cpu_data, memory_data, title, x_title, y_title):
    fig = go.Figure()
    fig.add_trace(go.Box(y=cpu_data, name="CPU Usage (%)"))
    fig.add_trace(go.Box(y=memory_data, name="Memory Usage (%)"))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title
    )
    return fig

inference_time_fig = utils.create_inference_time_boxplot(inference_times)
resource_usage_fig = utils.create_resource_usage_boxplot(cpu_percentages, memory_percentages)

inference_time_fig.write_image("workspace/vis/inference_times_boxplot_pytorch.png")
resource_usage_fig.write_image("workspace/vis/system_resource_usage_boxplot_pytorch.png")