# Gaussianity Metric

**Gaussianity** is a novel evaluation metric designed to assess the internal semantic representation quality of Large Language Models (LLMs). Instead of focusing solely on the final output (e.g., answer accuracy, fluency, or generation quality), Gaussianity aims to measure how uniformly and coherently the internal token-level hidden states are organized in the semantic space. It does so by evaluating the closeness of the model’s hidden state distribution to a multivariate Gaussian distribution after applying a standardized dimensionality reduction procedure.

### Key Features

- **Task-Agnostic Evaluation:**  
  Gaussianity does not rely on downstream tasks or labeled datasets. It can be applied to any set of text inputs to assess the “semantic alignment” quality of a model’s internal representations.

- **Distribution-Based Assessment:**  
  Instead of treating hidden states as arbitrary vectors, Gaussianity quantifies their statistical properties. By fitting a multivariate Gaussian to the reduced representations, Gaussianity provides a measure of isotropy and uniformity in the semantic embedding space.

- **Model-Agnostic and Scalable:**  
  Gaussianity can be applied to various LLMs (including open-source models like LLaMA-7B, MPT, and others) without specialized customization. With minimal setup, it can help researchers and practitioners compare different models’ internal representation quality on a common scale.

---

## Installation, How to run

1. **Clone this repository:**
    ```bash
    git clone https://github.com/Seungjun1127/Gaussianity.git
    cd Gaussianity
    ```

2. **Set up a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
   
   Typical dependencies include:
   - PyTorch (for running LLM inference)
   - Transformers (for loading pre-trained models from Hugging Face)
   - Scikit-learn (for PCA or other dimensionality reduction methods)
   - NumPy, SciPy, and other statistical libraries

4. **Setting:**
   - set model
    ```bash
    model_name = 'falcon-7b'
    ```
   - set number of samples
   ```bash
   samples = [dataset[i]['sentence'] for i in range(300)]
   ```

5. **Run evaluation:**
    ```bash
    python tests/test_evaluation.py
    ```


---


## Interpreting the Results

- A **high Gaussianity score** suggests that the model’s semantic space is well-organized, isotropic, and evenly distributed, potentially indicating good generalization capacity.
- A **low Gaussianity score** suggests more anisotropy or multimodality, meaning the model’s internal semantics might be skewed or clustered, hinting at potential limitations in semantic alignment.

Use these insights to compare different models (e.g., base vs. instruction-tuned, different architectures, or models before and after fine-tuning).

## Contact

For questions or discussion, please open an issue on this repository.
