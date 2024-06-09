import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RecallAtKSurrogateLoss(nn.Module):
    def __init__(self, k_values, tau1=1.0, tau2=0.01):
        super(RecallAtKSurrogateLoss, self).__init__()
        self.k_values = k_values
        self.tau1 = tau1
        self.tau2 = tau2

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        similarity_matrix = torch.matmul(embeddings, embeddings.t())

        loss = 0
        for i in range(batch_size):
            positive_indices = labels == labels[i]
            negative_indices = labels != labels[i]

            positive_similarities = similarity_matrix[i][positive_indices]
            negative_similarities = similarity_matrix[i][negative_indices]

            for k in self.k_values:
                top_k_negatives = torch.topk(negative_similarities, k, largest=True).values
                recall_at_k = F.sigmoid(self.tau1 * (top_k_negatives.max() - positive_similarities))

                loss += (1 - recall_at_k.mean())

        return loss / batch_size

class SiMix:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def mixup(self, embeddings, labels):
        batch_size = embeddings.size(0)
        indices = torch.randperm(batch_size)
        mixed_embeddings = self.alpha * embeddings + (1 - self.alpha) * embeddings[indices]
        mixed_labels = self.alpha * labels + (1 - self.alpha) * labels[indices]

        return mixed_embeddings, mixed_labels

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)
        
        simix = SiMix()
        mixed_embeddings, mixed_labels = simix.mixup(embeddings, labels)

        loss = criterion(mixed_embeddings, mixed_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# 模型架構
class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.normalize(self.fc(x))
        return x

# 主程式碼
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 128
model = SimpleCNN(embedding_dim).to(device)
criterion = RecallAtKSurrogateLoss(k_values=[1, 2, 4, 8, 16])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假設 dataloader 已經定義
# dataloader = ...

num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = train_model(model, dataloader, criterion, optimizer, device)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
