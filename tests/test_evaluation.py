"""Entire pipeline test"""

import sys
import os
import numpy as np
import torch

# src 폴더를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.GSAM.data.loaders import load_word_dataset
from src.GSAM.models.model_loader import LargeModelLoader
from src.GSAM.metrics.gsam import compute_gsam, batch_compute_gsam, compare_models

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_gsam_pipeline():
    # 데이터 로드
    dataset = load_word_dataset()
    print("Loaded dataset for testing.")

    # 모델 로드
    model_loader = LargeModelLoader()
    model_name = 'llama2-7b'  # Llama-2-7 모델 이름으로 변경
    model, tokenizer = model_loader.load_model(model_name, device='cpu')  # device를 'cpu'로 설정
    
    # 패딩 토큰 설정
    tokenizer.pad_token = tokenizer.eos_token  # EOS 토큰을 패딩 토큰으로 설정

    # 데이터셋에서 여러 샘플을 가져와서 토큰화
    samples = [dataset[i]['sentence'] for i in range(10)]  # 첫 10개 샘플에서 'sentence' 필드 추출
    inputs = tokenizer(samples, return_tensors='pt', padding=True, truncation=True)

    # 모델의 activations 추출
    with torch.no_grad():
        activations = model_loader.extract_activation(model, inputs['input_ids'], inputs['attention_mask'])

    # GSAM 계산
    gsam_score = compute_gsam(activations.numpy())
    print(f"GSAM Score: {gsam_score}")

if __name__ == "__main__":
    test_gsam_pipeline()
