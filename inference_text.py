# --------------------------------------------------------
# Single Sentence Inference (Predict Citation Type)
# --------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Path to your fine-tuned BERT model ---
MODEL_PATH = r"D:\Data Science Projects\Data Citation Intent Classification\models\bert_baseline_model"

# --- Load model and tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_text(text: str):
    """
    Predict whether a given text refers to dataset usage (Primary)
    or just a mention (Secondary).
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get predicted class
    pred = int(torch.argmax(logits, dim=1).item())
    id2label = {0: "Primary", 1: "Secondary"}
    
    return pred, id2label[pred]

# --- Example test ---
if __name__ == "__main__":
    text = "We trained our model using the ImageNet dataset and report the results."
    pred_num, pred_label = predict_text(text)
    
    print(f"🧠 Input: {text}")
    print(f"✅ Prediction: {pred_num} → {pred_label}")
