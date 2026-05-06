import pandas as pd
import joblib
import random
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout



try:
    df = pd.read_csv("phishing_dataset.csv")
except FileNotFoundError:
    print("Error: Run '1_create_dataset.py' first!")
    exit()

df = df.dropna(subset=["text"])

df["text"] = df["text"].astype(str)

df = df[df["text"].str.strip() != ""]

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=100)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

input_features = 100
hidden_1_dim = 64
hidden_2_dim = 32
hidden_3_dim = 16
output_dim = 1

mlp_model = Sequential([
    Dense(hidden_1_dim, input_shape=(input_features,)),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    Dense(hidden_2_dim, activation='relu'),
    Dense(hidden_3_dim, activation='tanh'),
    Dense(output_dim, activation='sigmoid')
])

mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train_vec, y_train, epochs=20, batch_size=8, verbose=1)

nb_probs = nb_model.predict_proba(X_test_vec)[:, 1]
mlp_probs = mlp_model.predict(X_test_vec).flatten()

best_weight_nb = 0.5
best_accuracy = 0

population = [random.uniform(0, 1) for _ in range(10)]

for _ in range(20):

    results = []

    for weight in population:

        ensemble_prob = (weight * nb_probs) + ((1.0 - weight) * mlp_probs)

        predictions = (ensemble_prob >= 0.5).astype(int)

        acc = accuracy_score(list(y_test), predictions)

        results.append((weight, acc))

    results.sort(key=lambda x: x[1], reverse=True)

    top3 = [results[0][0], results[1][0], results[2][0]]

    if results[0][1] > best_accuracy:
        best_accuracy = results[0][1]
        best_weight_nb = results[0][0]

    population = top3.copy()

    while len(population) < 10:

        p1 = random.choice(top3)
        p2 = random.choice(top3)

        child = (p1 + p2) / 2

        population.append(child)

weights = {
    "nb_weight": best_weight_nb,
    "mlp_weight": 1.0 - best_weight_nb
}

print(f"GA Optimization Complete! Optimal Blend: NB={weights['nb_weight']:.2f}, MLP={weights['mlp_weight']:.2f}")

joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(weights, 'weights.pkl')

mlp_model.save('mlp_model.keras')

print("Training complete! You can now run the Streamlit app.")