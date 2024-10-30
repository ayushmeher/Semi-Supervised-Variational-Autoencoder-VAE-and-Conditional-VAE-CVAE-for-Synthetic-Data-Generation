# Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters
n_classes = 5
n_features = 100
n_samples = 1000
n_labeled = 100
latent_dim = 10  # Updated to match latent dimension
n_hidden = 20
batch_size = 100

# VAE Model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, latent_dim * 2)  # Mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        z = self.encoder(x)
        mu, log_var = z.chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decoder(z)
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

# VAE Loss Function (with KL Divergence)
def vae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss / x.size(0)

# Train VAE
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

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

# VAE Training Loop (Semi-Supervised Approach)
for epoch in range(100):
    # Train on labeled data
    for batch_idx, (data, labels) in enumerate(labeled_loader):
        data = data.float()
        reconstructed_x, mu, log_var = vae(data)
        loss = vae_loss(reconstructed_x, data, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Train on unlabeled data (unsupervised)
    for batch_idx, (data, _) in enumerate(unlabeled_loader):
        data = data.float()
        reconstructed_x, mu, log_var = vae(data)
        loss = vae_loss(reconstructed_x, data, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Generate synthetic data using the trained VAE
synthetic_data = []
synthetic_labels = []
for i in range(n_classes):
    z = np.random.randn(n_samples // n_classes, latent_dim)  # Sample from latent space
    synthetic_class_data = vae.decoder(torch.from_numpy(z).float()).detach().numpy()
    synthetic_data.append(synthetic_class_data)
    synthetic_labels.append(np.full(n_samples // n_classes, i))

synthetic_data = np.concatenate(synthetic_data)
synthetic_labels = np.concatenate(synthetic_labels)

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

# Print accuracy and confusion matrix
accuracy = accuracy_score(holdout_labels, predicted_labels)
conf_mat = confusion_matrix(holdout_labels, predicted_labels)
print(f'Accuracy: {accuracy}')
print(conf_mat)

# Apply PCA to reduce dimensionality and visualize the data
pca = PCA(n_components=2)
pca_data = pca.fit_transform(synthetic_data)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=synthetic_labels)
plt.show()
