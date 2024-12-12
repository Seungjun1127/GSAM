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
    
    def extract_activation(self, model, input_ids, attention_mask=None):
        """
        Extract and combine hidden states from the last encoder layer and first decoder layer
        by mean pooling over tokens and residual sum each other.

        Args:
            model: HuggingFace model
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            torch.Tensor: Combined hidden states (batch_size, hidden_dim)
        """
        # Get hidden states from model output
        outputs = model(input_ids, 
                        attention_mask=attention_mask,
                        output_hidden_states=True)
        
        hidden_states = outputs.hidden_states

        # Last encoder layer (-1)
        encoder_last = hidden_states[-1]
        # First decoder layer (0)
        decoder_first = hidden_states[0]

        # Mean pooling excluding masked tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            encoder_pooled = (encoder_last * mask).sum(1) / mask.sum(1)
            decoder_pooled = (decoder_first * mask).sum(1) / mask.sum(1)
        else:
            encoder_pooled = encoder_last.mean(dim=1)
            decoder_pooled = decoder_first.mean(dim=1)

        # Combine tensors by addition
        combined = encoder_pooled + decoder_pooled
        
        return combined

