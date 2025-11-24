import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Configuration
DATASET_PATH = "../Dataset_v2"
OUTPUT_FILE = "../output/mediation_outputs.json"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Prompts
JUDGMENT_SYSTEM_PROMPT = """You are a neutral moderator analyzing a heated argument. Your task is to provide an objective judgment in strict JSON format.

Instructions:
1. Identify the main points of the disagreement.
2. Determine which participant (if any) displays more reason, evidence, and respectful conduct.
3. Provide a clear and fair verdict (e.g., "User A", "User B", or "Both").
4. Explain why in 3-6 sentences.

Output Format (JSON only):
{
  "verdict": "<User A | User B | Both>",
  "rationale": "<Your explanation>"
}"""

STEERING_SYSTEM_PROMPT = """You are a conflict mediator. Your goal is to calm tension and encourage productive communication.

Instructions:
Write a single message addressed to both participants that:
- Acknowledges both perspectives respectfully
- Avoids taking sides or assigning blame
- Encourages shared understanding
- Uses calm, empathetic language
- Is 3–6 sentences long

Output only the message text."""

def extract_messages(conversation_part):
    """Iteratively extracts messages from nested replies."""
    messages = []
    stack = [conversation_part]
    
    while stack:
        current = stack.pop()
        
        if isinstance(current, dict):
            # Extract message details if this dict represents a message
            if 'Reply_Text' in current and 'From' in current and 'Timestamp' in current:
                message_data = {
                    'Reply_Text': current['Reply_Text'],
                    'From': current['From'],
                    'Reply_Score': current.get('Reply_Score', 0),
                    'Timestamp': current['Timestamp']
                }
                messages.append(message_data)
            
            if 'Replies' in current:
                # Add replies to stack (reversed to maintain order)
                stack.extend(reversed(current['Replies']))
        elif isinstance(current, list):
            stack.extend(reversed(current))
            
    return messages

def format_conversation(thread_messages):
    """Format a list of messages into a readable conversation string."""
    conversation = ""
    for i, msg in enumerate(thread_messages):
        author = msg.get('From', f'User_{i}')
        text = msg.get('Reply_Text', '')
        conversation += f"{author}: {text}\n\n"
    return conversation.strip()

def clean_judgment_output(output_text):
    """Extracts JSON object from the output text."""
    # Try to find a JSON block
    match = re.search(r'\{.*\}', output_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            # Validate JSON
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
    return output_text.strip()

def clean_steering_output(output_text):
    """Cleans up steering message output."""
    # Remove common preambles/postambles
    text = output_text.strip()
    # Remove quotes if the whole message is quoted
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text

def load_dataset(dataset_path):
    json_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    
    conversations_by_post = {}
    print(f"Found {len(json_files)} JSON files.")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                post_id = data.get('Post_id', os.path.basename(file_path).replace('.json', ''))

                if 'Threads' in data:
                    thread_messages = extract_messages(data['Threads'])
                    if len(thread_messages) >= 2:
                        conversations_by_post[post_id] = thread_messages
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    return conversations_by_post

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Check for Flash Attention availability
    attn_implementation = "sdpa"  # Default to scaled dot product attention (fast on Torch 2.0+)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        try:
            import flash_attn
            attn_implementation = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            print("Flash Attention 2 not installed, using SDPA")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=attn_implementation
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for batched generation

    conversations = load_dataset(DATASET_PATH)
    print(f"Loaded {len(conversations)} conversations.")

    mediation_results = []
    
    # Batch processing
    # Reduced to 8 to avoid OOM on A100 with long contexts
    BATCH_SIZE = 8
    conversation_items = list(conversations.items())
    
    for i in tqdm(range(0, len(conversation_items), BATCH_SIZE), desc="Generating Mediations"):
        batch = conversation_items[i:i + BATCH_SIZE]
        batch_post_ids = [item[0] for item in batch]
        batch_messages = [item[1] for item in batch]
        batch_texts = [format_conversation(msgs) for msgs in batch_messages]

        # Generate judgment
        judgment_prompts = []
        for text in batch_texts:
            messages = [
                {"role": "system", "content": JUDGMENT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this conversation:\n\n{text}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            judgment_prompts.append(prompt)

        # Set max_length to prevent OOM on extremely long conversations
        inputs = tokenizer(judgment_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1, # Lower temperature for more deterministic JSON
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        # Decode only the generated tokens
        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        judgment_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        judgment_outputs = [clean_judgment_output(out) for out in judgment_outputs]

        # Generate steering
        steering_prompts = []
        for text in batch_texts:
            messages = [
                {"role": "system", "content": STEERING_SYSTEM_PROMPT},
                {"role": "user", "content": f"Conversation:\n\n{text}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            steering_prompts.append(prompt)

        inputs = tokenizer(steering_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        steering_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        steering_outputs = [clean_steering_output(out) for out in steering_outputs]

        for j, post_id in enumerate(batch_post_ids):
            print(f"\n--- Post ID: {post_id} ---")
            print(f"Judgment:\n{judgment_outputs[j]}")
            print(f"Steering:\n{steering_outputs[j]}")
            print("-" * 50)
            
            mediation_results.append({
                'post_id': post_id,
                'conversation': batch_texts[j],
                'judgment': judgment_outputs[j],
                'steering': steering_outputs[j]
            })

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(mediation_results, f, indent=2)
    
    print(f"Mediation results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
