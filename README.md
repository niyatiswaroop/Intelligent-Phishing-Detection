# 🛡️ Intelligent Phishing Detection & Threat Analyzer

An advanced, hybrid machine learning application designed to detect phishing emails and analyze threat levels. This project combines traditional machine learning (Naïve Bayes), deep learning (Multi-Layer Perceptron), Genetic Algorithms for ensemble optimization, and Fuzzy Logic to provide a comprehensive threat score.

## 🚀 Features
- **Hybrid Machine Learning Ensemble**: Combines predictions from a Naïve Bayes classifier and an MLP Neural Network.
- **Genetic Algorithm Optimization**: Automatically optimizes the weighting between the two models to achieve the highest accuracy.
- **Fuzzy Logic Threat Assessment**: Uses `scikit-fuzzy` to evaluate the combined spam score alongside sender domain trust to generate a final, human-readable threat level (Safe to Critical).
- **Interactive Web Interface**: Built with Streamlit for a clean, user-friendly experience.

## 🛠️ Technology Stack
- **Frontend**: Streamlit
- **Machine Learning**: Scikit-Learn (Naïve Bayes, TF-IDF Vectorization)
- **Deep Learning**: TensorFlow / Keras (MLP)
- **Fuzzy Logic**: Scikit-Fuzzy
- **Data Manipulation**: Pandas, NumPy

## 📂 Project Structure
- `app.py`: The main Streamlit web application.
- `train_model.py`: Script to train the Naïve Bayes and MLP models, optimize weights using a GA, and save the models.
- `create_dataset.py` & `dataset_preprocess.py`: Scripts for preparing and preprocessing the CSV datasets.
- `requirements.txt`: Python dependencies.
- `*.pkl` & `*.keras`: Pre-trained model files and vectorizers.

## ⚙️ Installation & Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd Intelligent-Phishing-Detection
   ```

2. **Install the dependencies**:
   Make sure you have Python installed. Run the following command:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   You can start the Streamlit app immediately using the pre-trained models included in the repository.
   ```bash
   streamlit run app.py
   ```

## 🧠 Training the Models (Optional)
If you wish to retrain the models on your own data:
1. Ensure your `phishing_dataset.csv` is in the project root directory.
2. Run the training script:
   ```bash
   python train_model.py
   ```
*(Note: Training on the full dataset may take a significant amount of time depending on your hardware. You may want to adjust the `batch_size` and `epochs` in `train_model.py` for faster training.)*

## ⚠️ Important Note on Datasets
The dataset files (`phishing_dataset.csv` and `spam_Emails_data.csv`) are very large (~350MB each). They are **ignored** via `.gitignore` and should not be committed to GitHub to avoid exceeding file size limits.

## 📜 License
This project is created for educational and research purposes (Problem Based Learning).
