from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LargeModelLoader:
    def __init__(self):
        self.available_models = {
            'gpt-neo-125M': 'EleutherAI/gpt-neo-125M',
            'gpt-neo-1.3B': 'EleutherAI/gpt-neo-1.3B',
            'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
            'llama2-7b': 'meta-llama/Llama-2-7b-hf',
            'llama2-13b': 'meta-llama/Llama-2-13b',
            'falcon-7b': 'tiiuae/falcon-7b',
            'mpt-7b': 'mosaicml/mpt-7b',
            'pythia-7b': 'EleutherAI/pythia-6.9b'
        }
        
    def load_model(self, model_name: str, device: str = 'cpu'):

        if model_name not in self.available_models:
            raise ValueError(f"Unsupported model. Available models: {list(self.available_models.keys())}")
            
        model_path = self.available_models[model_name]
        
        print(f"Loading {model_name} model to {device}...")
        
        # Load model with appropriate device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if device == 'cuda' else None,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Finished loading {model_name} model")
        
        return model, tokenizer
        
    def get_available_models(self):
        return list(self.available_models.keys())
    
    def extract_activation(self, model, input_ids, attention_mask=None):

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

