# Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
from torch.nn.functional import softmax

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters
n_classes = 5
n_features = 100
n_samples = 1000
n_labeled = 100
latent_dim = 10
n_hidden = 20
batch_size = 100

# CVAE Model
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features + n_classes, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, latent_dim * 2)  # Mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_classes, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, c):
        z = self.encoder(torch.cat((x, c), dim=1))
        mu, log_var = z.chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decoder(torch.cat((z, c), dim=1))
        return reconstructed_x, mu, log_var

# Custom dataset for our synthetic data
class SyntheticDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# CVAE Loss Function (with KL Divergence)
def cvae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss / x.size(0)

# Train CVAE
cvae = CVAE()
optimizer = optim.Adam(cvae.parameters(), lr=0.001)

# Generate random data with class structure
class_means = np.random.rand(n_classes, n_features)
class_covs = np.array([np.eye(n_features) for _ in range(n_classes)])
class_data = []
class_labels = []
for i in range(n_classes):
    class_data.append(np.random.multivariate_normal(class_means[i], class_covs[i], n_samples // n_classes))
    class_labels.append(np.full(n_samples // n_classes, i))

# Shuffle data and split into labeled and unlabeled sets
data = np.concatenate(class_data)
labels = np.concatenate(class_labels)
perm = np.random.permutation(n_samples)
data = data[perm]
labels = labels[perm]

labeled_data = data[:n_labeled]
labeled_labels = labels[:n_labeled]
unlabeled_data = data[n_labeled:]

labeled_dataset = SyntheticDataset(labeled_data, labeled_labels)
unlabeled_dataset = SyntheticDataset(unlabeled_data, np.zeros(len(unlabeled_data)))

labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

# CVAE Training Loop (Semi-Supervised Approach)
for epoch in range(100):
    # Train on labeled data
    for batch_idx, (data, labels) in enumerate(labeled_loader):
        data = data.float()
        labels = nn.functional.one_hot(labels.long(), n_classes).float()
        reconstructed_x, mu, log_var = cvae(data, labels)
        loss = cvae_loss(reconstructed_x, data, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Train on unlabeled data (unsupervised)
    for batch_idx, (data, _) in enumerate(unlabeled_loader):
        data = data.float()
        labels = torch.zeros((data.size(0), n_classes)).float()
        reconstructed_x, mu, log_var = cvae(data, labels)
        loss = cvae_loss(reconstructed_x, data, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Generate synthetic data using the trained CVAE
synthetic_data = []
synthetic_labels = []
for i in range(n_classes):
    z = np.random.randn(n_samples // n_classes, latent_dim)
    c = np.zeros((n_samples // n_classes, n_classes))
    c[:, i] = 1
    synthetic_class_data = cvae.decoder(torch.from_numpy(np.concatenate((z, c), axis=1)).float()).detach().numpy()
    synthetic_data.append(synthetic_class_data)
    synthetic_labels.append(np.full(n_samples // n_classes, i))

synthetic_data = np.concatenate(synthetic_data)
synthetic_labels = np.concatenate(synthetic_labels)

# Simplified Inception Score
def inception_score(synthetic_data, eps=1e-16):
    synthetic_data = torch.from_numpy(synthetic_data).float()
    preds = softmax(synthetic_data, dim=1)
    py = torch.mean(preds, dim=0)
    kl_div = preds * (torch.log(preds + eps) - torch.log(py + eps))
    score = torch.exp(torch.mean(torch.sum(kl_div, dim=1))).item()
    return score

# Simplified FID using mean and covariance
def calculate_fid(real_data, fake_data):
    real_data = real_data.cpu().numpy() if isinstance(real_data, torch.Tensor) else real_data
    fake_data = fake_data.cpu().numpy() if isinstance(fake_data, torch.Tensor) else fake_data
    
    real_data = torch.from_numpy(real_data).float()
    fake_data = torch.from_numpy(fake_data).float()
    
    real_mean = torch.mean(real_data, dim=0)
    fake_mean = torch.mean(fake_data, dim=0)
    real_cov = torch.cov(real_data.T)
    fake_cov = torch.cov(fake_data.T)
    
    mean_diff = torch.sum((real_mean - fake_mean) ** 2)
    cov_sqrt = torch.sqrt(real_cov + fake_cov - 2 * torch.sqrt(real_cov @ fake_cov))
    fid = mean_diff + torch.trace(cov_sqrt)
    
    return fid.item()

# Use simplified Inception Score and FID
is_score = inception_score(synthetic_data)
fid_score = calculate_fid(data, synthetic_data)

print(f'Inception Score: {is_score}')
print(f'Fréchet Inception Distance: {fid_score}')

# Train a classifier on the synthetic data
svm_model = svm.SVC(kernel='linear')
svm_model.fit(synthetic_data, synthetic_labels)

# Test the classifier on a holdout set
holdout_data = []
holdout_labels = []
for i in range(n_classes):
    holdout_class_data = np.random.multivariate_normal(class_means[i], class_covs[i], n_samples // n_classes)
    holdout_data.append(holdout_class_data)
    holdout_labels.append(np.full(n_samples // n_classes, i))

holdout_data = np.concatenate(holdout_data)
holdout_labels = np.concatenate(holdout_labels)

predicted_labels = svm_model.predict(holdout_data)

# Print accuracy, precision, recall, and F1 score
accuracy = accuracy_score(holdout_labels, predicted_labels)
precision = precision_score(holdout_labels, predicted_labels, average='weighted')
recall = recall_score(holdout_labels, predicted_labels, average='weighted')
f1 = f1_score(holdout_labels, predicted_labels, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Perform t-SNE to visually explore class separation and diversity of synthetic data
tsne = TSNE(n_components=2)
tsne_data = tsne.fit_transform(synthetic_data)
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=synthetic_labels)
plt.title("t-SNE Visualization of Synthetic Data")
plt.show()
