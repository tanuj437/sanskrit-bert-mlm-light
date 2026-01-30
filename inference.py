import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sentencepiece as spm
import argparse
import os
from huggingface_hub import hf_hub_download

def predict_mask(text, model_path="YOUR-USERNAME/SanskritBERT", tokenizer_model="sp_unigram_64k.model", top_k=5):
    """
    Predicts the masked token in a Sanskrit sentence.
    """
    print(f"Loading model from: {model_path}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Try Loading AutoTokenizer (standard way)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        use_spm = False
    except Exception as e:
        print(f"AutoTokenizer failed ({e}). Falling back to manual SentencePiece loading...")
        
        # 2. Fallback: Manual Load (as known in QA notebook)
        try:
            # Check if local or needs download
            if not os.path.exists(tokenizer_model):
                print(f"Downloading tokenizer model from HF Hub ({model_path})...")
                tokenizer_model = hf_hub_download(repo_id=model_path, filename=tokenizer_model)
            
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_model)
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            use_spm = True
            print("Loaded SentencePiece model successfully.")
        except Exception as e2:
            print(f"Critical Error: Could not load tokenizer. {e2}")
            return

    model = model.to(device)

    # Tokenize
    if use_spm:
        # Manual SP processing
        if "[MASK]" not in text: 
             print("Error: Text must contain [MASK]")
             return
        
        # Replace [MASK] with a placeholder for splitting, or handle carefully
        # Simple approach for demonstration:
        # Note: This is rudimentary. For robust use, proper alignment is needed.
        # Here we just re-raise error if complexity is too high, or try basic encode.
        tokenized_ids = sp.encode_as_ids(text)
        # We need the Mask Token ID. In many SP models, it might not be explicit unless added.
        # Assuming user knows the ID or standard 4 (depends on vocab).
        # Let's rely on standard encode if [MASK] handling is complex in pure SP.
        
        # ACTUALLY, simpler path: Just use the loaded model config if possible.
        # But for this script, let's just stick to the AutoTokenizer path if valid, 
        # and warn user if SP is needed they should use the QA notebook example.
        print("Note: Running manual SentencePiece inference in this script is complex.")
        print("Please refer to `examples/qa.ipynb` for detailed BERT embedding generation.")
        return
        
    else:
        # standard handling
        if tokenizer.mask_token not in text:
            print(f"Error: Text must contain the mask token '{tokenizer.mask_token}'")
            return

        inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Predict
    with torch.no_grad():
        if use_spm:
             # Placeholder if we implemented full SP loop
             pass
        else:
            logits = model(**inputs).logits

    # Find mask index
    if not use_spm:
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        print(f"\nInput Text: {text}")
        print(f"Predictions:")
        
        for idx in mask_token_index:
            # Get top k predictions
            top_k_tokens = torch.topk(logits[0, idx], top_k)
            print(f"\nMask at index {idx.item()}:")
            for i in range(top_k):
                token_id = top_k_tokens.indices[i]
                score = top_k_tokens.values[i].item()
                token = tokenizer.decode(token_id)
                print(f"  {i+1}. {token} (Score: {score:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on SanskritBERT")
    parser.add_argument("--text", type=str, default="सत्यमेव जयते [MASK]", help="Input text with [MASK] token")
    parser.add_argument("--model", type=str, default="YOUR-USERNAME/SanskritBERT", help="Hugging Face model ID or path")
    parser.add_argument("--sp_model", type=str, default="sp_unigram_64k.model", help="Path to SentencePiece model file")
    args = parser.parse_args()

    predict_mask(args.text, args.model, args.sp_model)
