from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Log into Hugging Face to access the model (replace "your_token_here" with your actual token)
from huggingface_hub import login
login("", add_to_git_credential=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Check if CUDA is available and move the model to GPU if it is


# Initialize the text-generation pipeline with truncation and pad_token_id settings
llm = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    truncation=True,  # Enable truncation
    pad_token_id=tokenizer.eos_token_id,  # Use the end of sequence token for padding
    device=0 if device == "cuda" else -1  # Set device to 0 for GPU, -1 for CPU
)

def generate_important_objects_description(objects, threshold=10):
    """
    Generates a description of important objects based on distance.
    
    Args:
    - objects (list of dict): Each dict has {'name': str, 'distance': float}
    - threshold (float): The maximum distance to consider an object important.

    Returns:
    - str: A description of important objects.
    """
    # Filter out objects within the threshold distance
    important_objects = [obj for obj in objects if obj['distance'] <= threshold]
    
    # Format input prompt for the LLM
    if important_objects:
        prompt = (
            "Rhe important objects to take note of within a "
            f"{threshold} inch range to a blind person. Explain the positions relative to the person's point of view. Make it clear and concise with once sentence maximum for each item. Do not say anything else after. Here are the objects and their distances:\n"
        )
        prompt += "\n".join([f"{obj['name']}: {obj['distance']} inches" for obj in important_objects])
    else:
        prompt = f"There are no objects within the {threshold} inches range."

    # Generate a response
    response = llm(prompt, max_length=500, num_return_sequences=1)[0]["generated_text"]
    return response

print("running...\n")
objects = [
    {"name": "Tree", "distance": 5},
    {"name": "Building", "distance": 15},
    {"name": "Bench", "distance": 7},
    {"name": "Fountain", "distance": 20},
]

# Call the function and print the output
description = generate_important_objects_description(objects, threshold=10)
print(description)
print("done")
