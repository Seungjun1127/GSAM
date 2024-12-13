"""Entire pipeline test"""

import sys
import os
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.GSAM.data.loaders import load_word_dataset
from src.GSAM.models.model_loader import LargeModelLoader
from src.GSAM.metrics.gsam import compute_gsam, batch_compute_gsam, compare_models

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_gsam_pipeline():

    torch.cuda.empty_cache()

    # 데이터 로드
    dataset = load_word_dataset()
    print("Loaded dataset for testing.")

    # 모델 로드
    model_loader = LargeModelLoader()
    model_name = 'pythia-7b'  # Llama-2-7 모델 이름으로 변경

    # CUDA 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = model_loader.load_model(model_name, device=device)  # device를 설정

    # 패딩 토큰 설정
    tokenizer.pad_token = tokenizer.eos_token  # EOS 토큰을 패딩 토큰으로 설정

    # 데이터셋에서 여러 샘플을 가져와서 토큰화
    samples = [dataset[i]['sentence'] for i in range(10)]  # 전체 데이터셋에서 'sentence' 필드 추출
    inputs = tokenizer(samples, return_tensors='pt', padding=True, truncation=True)

    # 모델의 activations 추출
    with torch.no_grad():
        activations = model_loader.extract_activation(model, inputs['input_ids'].to(device), inputs['attention_mask'].to(device))

    # GSAM 계산
    #metric = 'kl_divergence'  # metric 변수를 정의
    metric = 'neg_log_likelihood'
    gsam_score = compute_gsam(activations.cpu().numpy(), metric=metric)  # CPU로 이동하여 NumPy 배열로 변환
    print(f"GSAM Score: {gsam_score}")

    # GSAM 결과를 텍스트 파일에 기록
    with open("gsam_results.txt", "a") as f:  # 'a' 모드로 파일 열기 (추가 모드)
        f.write(f"Model: {model_name}, Metric: {metric}, GSAM Score: {gsam_score}\n")  # 결과 기록

if __name__ == "__main__":
    test_gsam_pipeline()
