import torch
from torch.nn.functional import softmax

def predict_formality(text, model, tokenizer, threshold=0.5):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get logits from model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities
    probs = softmax(logits, dim=-1)
    confidence, pred = torch.max(probs, dim=1)

    label_map = {0: "informal", 1: "formal"}
    predicted_label = label_map[pred.item()]
    
    if confidence.item() >= threshold:
        return f"{text} is {predicted_label} japanese"
    else:
        return f"Not confident enough to classify — confidence: {confidence.item():.2f}"