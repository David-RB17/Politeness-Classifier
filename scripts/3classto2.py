import pandas as pd

# Load your CSV file
df = pd.read_csv("G:\Python Projects\politeness-classifier-jp\data\processed\BunnyGirl300-Preprocessed.csv")  # Replace with your actual filename

# Convert labels: 0 (ため口) and 1 (普通) → 0 (informal), 2 (敬語) → 1 (formal)
df['label'] = df['label'].apply(lambda x: 0 if x in [0, 1] else 1)

# Optional: Save to a new CSV file
df.to_csv("G:\Python Projects\politeness-classifier-jp\data\processed\BunnyGirl800-Preprocessed-binary.csv", index=False)