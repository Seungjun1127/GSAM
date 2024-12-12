"""Model Loading / Inference Test"""

from src.GSAM.models.loaders import load_model

if __name__ == "__main__":
    model = load_model()
    print(model)