from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


if __name__ == "__main__":
    data = pd.read_excel("data/detect_dataset.xlsx")
    train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)
    train.to_excel("split_data/train.xlsx", index=False)
    test.to_excel("split_data/test.xlsx", index=False)

