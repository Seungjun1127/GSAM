# Gaussian Semantic Alignment Metric (GSAM)

**GSAM** is a novel evaluation metric designed to assess the internal semantic representation quality of Large Language Models (LLMs). Instead of focusing solely on the final output (e.g., answer accuracy, fluency, or generation quality), GSAM aims to measure how uniformly and coherently the internal token-level hidden states are organized in the semantic space. It does so by evaluating the closeness of the model’s hidden state distribution to a multivariate Gaussian distribution after applying a standardized dimensionality reduction procedure.

### Key Features

- **Task-Agnostic Evaluation:**  
  GSAM does not rely on downstream tasks or labeled datasets. It can be applied to any set of text inputs to assess the “semantic alignment” quality of a model’s internal representations.

- **Distribution-Based Assessment:**  
  Instead of treating hidden states as arbitrary vectors, GSAM quantifies their statistical properties. By fitting a multivariate Gaussian to the reduced representations, GSAM provides a measure of isotropy and uniformity in the semantic embedding space.

- **Model-Agnostic and Scalable:**  
  GSAM can be applied to various LLMs (including open-source models like LLaMA-7B, MPT, and others) without specialized customization. With minimal setup, it can help researchers and practitioners compare different models’ internal representation quality on a common scale.

---

## Installation

1. **Clone this repository:**
    ```bash
    git clone https://github.com/Seungjun1127/GSAM.git
    cd gsam
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

---

## How It Works

**Step-by-step:**

1. **Data Collection:**  
   Prepare a corpus of sentences. These sentences should be diverse and not necessarily task-specific. The number of sentences is up to you, but a few thousand sentences are recommended for stable statistical estimates.

2. **Model Inference:**  
   Run your chosen LLM on the input sentences and extract hidden states from a specified layer. This might involve:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   model_name = "your-chosen-llm"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   
   # Example: extract hidden states for a batch of sentences
   sentences = ["This is a sample sentence.", "Here is another one.", ...]
   inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
   outputs = model(**inputs, output_hidden_states=True)
   hidden_states = outputs.hidden_states  # A tuple of hidden states per layer
   # Select a particular layer, e.g., the last layer
   selected_layer_states = hidden_states[-1]  # shape: [batch_size, seq_length, hidden_dim]
   ```

3. **Dimensionality Reduction:**  
   Apply a dimension reduction method (e.g., PCA) to the collected token-level vectors. This step transforms the high-dimensional hidden states into a manageable, fixed dimension (e.g., 50D).
   ```python
   from sklearn.decomposition import PCA
   import torch

   # Flatten the token embeddings
   token_embeddings = selected_layer_states.reshape(-1, selected_layer_states.shape[-1])
   
   pca = PCA(n_components=50)
   reduced_embeddings = pca.fit_transform(token_embeddings.detach().numpy())
   ```

4. **Gaussian Fitting and Metric Calculation:**  
   Fit a multivariate Gaussian distribution to the reduced embeddings and compute a divergence measure (e.g., KL-divergence) between the empirical distribution and the fitted Gaussian.
   ```python
   import numpy as np
   from scipy.stats import multivariate_normal

   mean = np.mean(reduced_embeddings, axis=0)
   cov = np.cov(reduced_embeddings, rowvar=False)
   
   # Calculate log-likelihoods under the fitted Gaussian
   rv = multivariate_normal(mean=mean, cov=cov)
   log_likelihoods = rv.logpdf(reduced_embeddings)
   
   # GSAM can be defined as, for instance:
   # GSAM = average log-likelihood (the higher, the closer to Gaussian)
   gsam_score = np.mean(log_likelihoods)
   print("GSAM Score:", gsam_score)
   ```

   A higher GSAM score (or lower divergence) indicates that the internal representations are more Gaussian-like and isotropic.

---

## Interpreting the Results

- A **high GSAM score** suggests that the model’s semantic space is well-organized, isotropic, and evenly distributed, potentially indicating good generalization capacity.
- A **low GSAM score** suggests more anisotropy or multimodality, meaning the model’s internal semantics might be skewed or clustered, hinting at potential limitations in semantic alignment.

Use these insights to compare different models (e.g., base vs. instruction-tuned, different architectures, or models before and after fine-tuning).

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvements, bug fixes, or new features.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or discussion, please open an issue on this repository.