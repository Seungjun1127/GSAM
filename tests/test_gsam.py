"""GSAM Logic Test"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.GSAM.gsam import GSAM

if __name__ == "__main__":
    gsam = GSAM()
    print(gsam)