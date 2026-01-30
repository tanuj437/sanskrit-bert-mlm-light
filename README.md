# SanskritBERT (Light) 🕉️

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/tanuj437/SanskritBERT)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

**SanskritBERT (Light)** is a custom-trained Transformer encoder model designed specifically for the Sanskrit language. Optimized for efficiency ("Light" architecture), it is trained on a large corpus of Sanskrit texts using Masked Language Modeling (MLM).

This repository contains the source code, training scripts, and usage examples.
**The pre-trained model weights are hosted on Hugging Face:** [👉 **Download Weights Here**](https://huggingface.co/tanuj437/SanskritBERT)

## 🚀 Model Details

- **Architecture**: Transformers (Encoder-only)
- **Parameters**: ~15M (Lightweight)
- **Layers**: 6
- **Hidden Size**: 256
- **Heads**: 4
- **Context Length**: 512
- **Vocabulary**: 120,000 Sanskrit Subwords

## 📦 Installation

```bash
git clone https://github.com/tanuj437/sanskrit-bert-mlm-light.git
pip install -r requirements.txt
```

## 💻 Usage

You can use this model directly with the Hugging Face `transformers` library.

### Inference Example

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load from Hugging Face Hub
model_name = "tanuj437/SanskritBERT"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example sentence (Sanskrit)
text = "सत्यमेव जयते [MASK]"

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# Retrieve top prediction
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"Predicted token: {predicted_token}")
```

For a ready-to-run script, check `inference.py`.


## 📚 Examples & Tutorials

### Question Answering (QA)
We provide a Jupyter notebook demonstrating how to use `SanskritBERT` for Question Answering tasks (e.g., on the Bhagavad Gita dataset).
- Check `examples/qa.ipynb` to see how to load the model with `sentencepiece`, generate embeddings, and perform semantic search.

## 📂 Repository Structure

- `scripts/`: Training and evaluation scripts.
- `examples/`: Jupyter notebooks and usage examples.
- `inference.py`: Simple inference script for testing.
- `CITATION.cff`: Citation information for research.

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@misc{sanskritbert2024,
  title={SanskritBERT: A Light Transformer Model for Sanskrit},
  author={[Tanuj Saxena, Soumya Sharma, Kusum Lata]},
  year={2024},
  howpublished={\url{https://huggingface.co/tanuj437/SanskritBERT}},
}
```

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

