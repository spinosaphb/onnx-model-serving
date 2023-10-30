import torch
from torch import nn
from torchvision import models, transforms, datasets
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model():
    model = models.resnet50(pretrained=True)
    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, 2)
    return model


def create_data_loaders(data_dir, batch_size=32, train_ratio=0.8):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=preprocess)

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_loader


def train_model(model, train_data_loader, num_epochs=100):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for images, labels in tqdm(train_data_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            print(f"\noutput shape: {outputs.shape}")
            print(f"labels shape: {labels.shape}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_data_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')


def save_model(model, model_dir):
    model_path = f"{model_dir}/model.pth"
    torch.save(model.state_dict(), model_path)


def save_model_onnx(model, model_dir):
    input_ = torch.randn(1, 3, 224, 224).to(device)  # Create a dummy input with the desired shape
    torch.onnx.export(model, input_, model_dir, opset_version=11)


if __name__ == "__main__":
    model = create_model()
    train_data_loader, _ = create_data_loaders("workspace/datasets/dogs-vs-cats")
    train_model(model, train_data_loader)
    save_model(model, "workspace/models/resnet_model_100_epochs.pth")
    save_model_onnx(model, "workspace/models/resnet_model_100_epochs.onnx")
