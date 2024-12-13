"""Data Loader Test"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.GSAM.data.loaders import load_short_text_dataset, load_word_dataset, load_large_english_dataset

if __name__ == "__main__":
    dataset = load_short_text_dataset()
    print(dataset)