# download_tweeteval.py
from datasets import load_dataset

# Load the TweetEval sentiment dataset
dataset = load_dataset("tweeteval", "sentiment")

# Check the splits
print(dataset)
