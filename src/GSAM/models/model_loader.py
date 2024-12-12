from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LargeModelLoader:
    def __init__(self):
        self.available_models = {
            'llama2-7b': 'meta-llama/Llama-2-7b',
            'llama2-13b': 'meta-llama/Llama-2-13b',
            'falcon-7b': 'tiiuae/falcon-7b',
            'mpt-7b': 'mosaicml/mpt-7b',
            'pythia-7b': 'EleutherAI/pythia-6.9b'
        }
        
    def load_model(self, model_name: str, device: str = 'cuda'):
        """
        Load a pretrained large language model from HuggingFace.
        
        Args:
            model_name: Name of model to load (one of available_models keys)
            device: Device to load model on ('cuda' or 'cpu')
            
        Returns:
            model: Loaded model
            tokenizer: Model's tokenizer
        """
        if model_name not in self.available_models:
            raise ValueError(f"Unsupported model. Available models: {list(self.available_models.keys())}")
            
        model_path = self.available_models[model_name]
        
        print(f"Loading {model_name} model to {device}...")
        
        # Load model memory efficiently using 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Finished loading {model_name} model")
        
        return model, tokenizer
        
    def get_available_models(self):
        """Returns list of available models."""
        return list(self.available_models.keys())
