import json
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
)
import torch

def model_fn(model_dir):
    # Load model and tokenizer from SageMaker's model directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, request_content_type):
    # Parse input data (supports JSON and raw text)
    if request_content_type == "application/json":
        data = json.loads(request_body)
        text = data["text"]
    else:  # Default to raw text
        text = request_body
    return text

def predict_fn(input_data, model_dict):
    # Tokenize and predict
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    
    inputs = tokenizer(
        input_data, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=256
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Return probabilities and predicted class
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_class = torch.argmax(probs).item()
    return {"prediction": pred_class, "probabilities": probs.tolist()[0]}

def output_fn(prediction, response_content_type):
    # Format output as JSON
    return json.dumps(prediction), "application/json"