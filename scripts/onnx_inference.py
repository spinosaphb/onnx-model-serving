import onnxruntime as ort
import time
import psutil
import torchvision.datasets as datasets
from torchvision import transforms
from tqdm import tqdm
import torch

from scripts import utils

model_path = "workspace/models/resnet_model_100_epochs.onnx"
ort_session = ort.InferenceSession(model_path)

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

def run_inference(model: ort.InferenceSession, data: torch.Tensor):
    inputs = data.unsqueeze(0)

    start_time = time.time()
    outputs = model.run(None, {"input.1": inputs.numpy()})
    elapsed_time = time.time() - start_time
    return outputs, elapsed_time

def collect_system_stats():
    cpu_percentages.append(psutil.cpu_percent())
    memory_percentages.append(psutil.virtual_memory().percent)


for image, _ in tqdm(dataset):
    _, elapsed_time = run_inference(ort_session, image)
    inference_times.append(elapsed_time)
    collect_system_stats()

inference_time_boxplot = utils.create_inference_time_boxplot(inference_times)
inference_time_boxplot.write_image("workspace/vis/inference_times_boxplot_onnx.png")

resource_usage_boxplot = utils.create_resource_usage_boxplot(cpu_percentages, memory_percentages)
resource_usage_boxplot.write_image("workspace/vis/system_resource_usage_boxplot_onnx.png")
