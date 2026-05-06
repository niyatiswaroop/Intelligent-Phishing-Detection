import random
import pandas as pd
import os

print("--- STEP 1: DATASET CREATION ---")

def generate_dataset(filename="phishing_dataset.csv"):
    spam_phrases = [
        "Urgent update your password now", "You have won a lottery prize claim", 
        "Your bank account is suspended", "Click here for free viagra",
        "Invoice attached please pay immediately", "Exclusive offer just for you"
    ]
    ham_phrases = [
        "Meeting at 3pm tomorrow", "Can you review this code?", 
        "Happy birthday man", "Lunch is in the fridge",
        "Project deadline extended to Friday", "Here are the meeting notes"
    ]
    
    data = []
    for _ in range(150):
        data.append({"text": f"{random.choice(spam_phrases)} {random.randint(100,999)}", "label": 1})
        data.append({"text": f"{random.choice(ham_phrases)} {random.randint(100,999)}", "label": 0})
        
    df = pd.DataFrame(data)
    
    df.to_csv(filename, index=False)
    print(f"Dataset successfully created and saved as '{filename}'")
    print(f"Total records: {len(df)}")

if __name__ == "__main__":
    generate_dataset()