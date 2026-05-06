import pandas as pd

# Load your original CSV
# Replace 'input.csv' with your file name
df = pd.read_csv("spam_Emails_data.csv")

# Convert labels:
# Spam -> 1
# Ham -> 0
df["label"] = df["label"].map({
    "Spam": 1,
    "Ham": 0
})

# Keep only required columns in order: text,label
df = df[["text", "label"]]

# Save new CSV
df.to_csv("phishing_dataset.csv", index=False)

print("Conversion completed successfully!")
print(df.head())