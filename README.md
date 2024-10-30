
# Semi-Supervised Variational Autoencoder and Classifier for Synthetic Data Generation

This project implements **Variational Autoencoders (VAE)** and **Conditional Variational Autoencoders (CVAE)** for synthetic data generation in a semi-supervised learning setting. Using PyTorch, the models generate class-conditional data, and the synthetic samples are evaluated using simplified **Inception Score** and **Fréchet Inception Distance (FID)** metrics. A linear SVM classifier is trained on the generated data and tested on a holdout set to measure its effectiveness.

## Project Structure

- **vae_trainer.py**: Implements the VAE model, including training and data generation, along with classifier training and performance evaluation.
- **cvae_trainer.py**: Implements the CVAE model for class-conditional generation, simplified inception score, FID calculation, and visualization of synthetic data.

## Key Features

1. **Variational Autoencoder (VAE)**: Trained on labeled and unlabeled data to generate synthetic samples.
2. **Conditional Variational Autoencoder (CVAE)**: Extends the VAE with class conditioning, allowing generation of samples for specific classes.
3. **Semi-Supervised Training**: Uses a small labeled dataset and a larger unlabeled dataset to enhance learning.
4. **Synthetic Data Evaluation**: Calculates simplified **Inception Score** and **FID** for quality and diversity assessment of synthetic data.
5. **Classifier Training**: A linear SVM classifier is trained on synthetic data and evaluated on a holdout set.

## Requirements

- Python 3.x
- Libraries: PyTorch, scikit-learn, numpy, matplotlib

## Usage

1. **Train the VAE Model**:
   Run `vae_trainer.py` to train a standard VAE on both labeled and unlabeled data.
   ```bash
   python vae_trainer.py
   ```
   - Generates synthetic data and evaluates with an SVM classifier on a holdout test set.

2. **Train the CVAE Model**:
   Run `cvae_trainer.py` to train the CVAE for class-conditional synthetic data generation.
   ```bash
   python cvae_trainer.py
   ```
   - Includes data evaluation with Inception Score and FID metrics.

## Example Workflow

1. **Data Generation**:
   - VAE and CVAE models are trained to learn the distribution of the classes.
   - Both labeled and unlabeled samples are used, with the CVAE conditioned on specific class labels for targeted data generation.

2. **Data Evaluation**:
   - Simplified **Inception Score**: Measures the diversity and quality of generated data.
   - **FID Calculation**: Computes FID to compare synthetic and real data distributions.

3. **Classifier Training and Testing**:
   - Linear SVM classifier is trained on the synthetic dataset.
   - Evaluated on a holdout set to measure classification accuracy and generalization.

4. **Visualization**:
   - Dimensionality reduction with PCA (VAE) and t-SNE (CVAE) for visual exploration of synthetic data distribution.

## File Details

- **vae_trainer.py**: Trains VAE, generates synthetic data, evaluates with Inception Score and FID, trains and tests an SVM classifier.
- **cvae_trainer.py**: Trains CVAE with class conditioning, generates data, evaluates with Inception Score and FID, and visualizes using t-SNE.

## Example Output

- **Accuracy, Precision, Recall, and F1 Score** on holdout test set for classifier performance.
- **Inception Score** and **FID** to assess synthetic data quality.
- **Confusion Matrix** and **PCA/t-SNE Visualization** to analyze class distribution.

## Contributing

Contributions are welcome to expand the models, enhance evaluation metrics, or explore alternative semi-supervised learning techniques.

## Output 
![Screenshot 2024-10-29 202706](https://github.com/user-attachments/assets/7e6c938d-f622-4fcc-bb38-5b5434e01554)


