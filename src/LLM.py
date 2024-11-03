from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import cv2
from ObjectDetection import detect_objects  # Import the detect_objects function
import pyttsx3  # Import pyttsx3 for text-to-speech

# Log into Hugging Face
login("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", add_to_git_credential=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    pad_token_id=tokenizer.eos_token_id,
    device=0 if device == "cuda" else -1,
    temperature=0.5,  # Lower temperature for more focused responses
    max_new_tokens=200,   # Limit the number of new tokens generated
)

def generate_important_objects_description(threshold=100):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    # Detect objects in the captured frame
    objects = detect_objects(frame)

    # Create a dictionary to hold unique object names and their closest distances
    unique_objects = {}
    
    for obj in objects:
        if obj['distance'] <= threshold:
            # Use the object name as the key to ensure uniqueness
            if obj['name'] not in unique_objects or obj['distance'] < unique_objects[obj['name']]['distance']:
                unique_objects[obj['name']] = {'distance': obj['distance']}

    if unique_objects:
        prompt = (
            "You are an exceptionally intelligent and empathetic assistant, tasked with helping a blind person navigate their environment safely and confidently. "
            "Your goal is to provide detailed, accurate, and informative descriptions of the objects present within a specified range, while being mindful of their unique perspective. "
            "Please describe only the objects listed below, without mentioning any irrelevant information or unrelated objects. Use the format: 'Object: Distance' to present the information clearly. \n\n"
            "For each object, in addition to its name and distance, provide a concise one-sentence description of its location relative to the blind man. "
            "Indicate whether each object is positioned on the left, right, or center, and include any relevant details about its orientation or proximity to the individual. "
            "Consider the potential interactions a blind person might have with these objects; for instance, discuss how they could safely navigate around them, if any hazards might exist, and suggest actions they might take. \n\n"
            "Additionally, think about how these objects might be perceived through other senses, such as sound, texture, or even temperature. "
            "Incorporate sensory details that could assist the blind man in forming a mental image of their surroundings. For example, describe if an object is large or small, soft or hard, or if it makes a sound when touched. \n\n"
            "Remember to avoid repeating descriptions of the same objects. Your descriptions should be clear, concise, and tailored to enhance the blind man’s understanding of his immediate environment, thereby fostering a sense of security and independence. \n\n"
            "Always end with 'Please let me know if you need any adjustments or if you'd like me to continue with the next set of objects.'"
        )
        # Format the objects for the prompt
        prompt += "\n".join([f"{name}: {info['distance']} inches" for name, info in unique_objects.items()])
        prompt += "\nNow, provide your detailed descriptions of the objects, ensuring that each is accurate and helpful for the blind man to navigate safely."

    else:
        prompt = f"There are no objects within the {threshold} inches range."

    # Generate response from the language model using max_new_tokens
    response = llm(prompt, max_new_tokens=200, num_return_sequences=1)[0]["generated_text"]
    
    # Remove the prompt from the response
    response_content = response.split("Now, provide your detailed descriptions of the objects")[1] if "Now, provide your detailed descriptions of the objects" in response else response
    
    # Return the cleaned response
    response_text = response_content.strip()  # Strip whitespace to tidy up the output

    # Initialize pyttsx3 TTS engine and speak the response text
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)  # Set speech rate
    engine.setProperty("volume", 1)  # Set volume to max (1.0)
    
    engine.say(response_text)
    engine.runAndWait()  # Blocks while speaking

    return response_text
