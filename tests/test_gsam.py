"""GSAM Logic Test"""

import sys
import os

# src 폴더를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.GSAM.gsam import GSAM

if __name__ == "__main__":
    gsam = GSAM()
    print(gsam)