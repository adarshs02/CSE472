#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSE472 Project 2 - Part 4: User Simulator Evaluation
Optimized for ASU SOL Supercomputer with GPU and Multi-threading
"""

import os
import json
import random
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Thread-safe lock for writing results
write_lock = Lock()

def setup_environment(hf_token, dataset_path):
    """
    Setup environment for ASU SOL
    
    Args:
        hf_token: Hugging Face authentication token
        dataset_path: Path to dataset directory
    """
    # Authenticate with Hugging Face
    if hf_token:
        login(token=hf_token)
        print("✓ Authenticated with Hugging Face")
    else:
        print("⚠ Warning: No Hugging Face token provided")
    
    # Verify dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    print(f"✓ Dataset path verified: {dataset_path}")

def find_json_files(dataset_path):
    """Find all JSON files in the dataset directory"""
    json_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    
    print(f"✓ Found {len(json_files)} JSON files")
    return json_files

def extract_messages(conversation_part):
    """Recursively extracts messages from nested replies."""
    messages = []
    if isinstance(conversation_part, dict):
        if 'Reply_Text' in conversation_part and 'From' in conversation_part and 'Timestamp' in conversation_part:
            message_data = {
                'Reply_Text': conversation_part['Reply_Text'],
                'From': conversation_part['From'],
                'Reply_Score': conversation_part.get('Reply_Score', 0),
                'Timestamp': conversation_part['Timestamp']
            }
            messages.append(message_data)
        
        if 'Replies' in conversation_part and conversation_part['Replies']:
            for reply in conversation_part['Replies']:
                messages.extend(extract_messages(reply))
    elif isinstance(conversation_part, list):
        for item in conversation_part:
            messages.extend(extract_messages(item))
    return messages

def load_conversations(json_files):
    """Load and group conversations by post_id"""
    conversations_by_post = {}
    
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
    
    print(f"✓ Loaded {len(conversations_by_post)} conversations")
    return conversations_by_post

def load_model(model_name, device):
    """Load the LLM model with GPU optimization"""
    print(f"Loading model '{model_name}' on device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    print(f"✓ Model loaded successfully on {device}")
    return tokenizer, model

def select_cut_point(messages):
    """Select a random cut point in the middle portion of a conversation."""
    n = len(messages)
    if n < 4:
        return None
    
    # Cut in the middle 60% of the conversation (20% buffer on each end)
    min_cut = max(2, int(n * 0.2))
    max_cut = min(n - 2, int(n * 0.8))
    
    if min_cut >= max_cut:
        return n // 2
    
    return random.randint(min_cut, max_cut)

def get_conversation_prefix(messages, cut_index):
    """Get the conversation up to the cut point."""
    prefix_messages = messages[:cut_index]
    prefix_text = ""
    
    for msg in prefix_messages:
        author = msg.get('From', 'Unknown')
        text = msg.get('Reply_Text', '')
        prefix_text += f"{author}: {text}\n\n"
    
    target_user = messages[cut_index].get('From', 'Unknown') if cut_index < len(messages) else messages[-1].get('From', 'Unknown')
    
    return prefix_text.strip(), target_user

def get_original_continuation(messages, cut_index):
    """Get the original message that came after the cut point."""
    if cut_index < len(messages):
        return messages[cut_index].get('Reply_Text', '')
    return ""

def prepare_simulation_data(conversations_by_post):
    """Prepare all conversations for simulation"""
    simulation_data = []
    
    for post_id, messages in conversations_by_post.items():
        if len(messages) < 4:
            continue
        
        cut_index = select_cut_point(messages)
        if cut_index is None:
            continue
        
        prefix_text, target_user = get_conversation_prefix(messages, cut_index)
        original_continuation = get_original_continuation(messages, cut_index)
        
        simulation_data.append({
            'post_id': post_id,
            'messages': messages,
            'cut_index': cut_index,
            'prefix': prefix_text,
            'target_user': target_user,
            'original_continuation': original_continuation
        })
    
    print(f"✓ Prepared {len(simulation_data)} conversations for simulation")
    return simulation_data

# Mediation prompt templates
JUDGMENT_MEDIATION_PROMPT = """You are a neutral moderator analyzing a heated argument. Provide an objective judgment.

Conversation so far:
{conversation_prefix}

Instructions:
1. Identify the main points of disagreement.
2. Determine which participant displays more reason, evidence, and respectful conduct.
3. Provide a clear and fair verdict.
4. Explain why in 3-6 sentences, focusing on reasoning and tone.

Output Format (strict JSON):
{{
  "verdict": "<User A | User B | Both>",
  "rationale": "<Your explanation>"
}}"""

STEERING_MEDIATION_PROMPT = """You are a conflict mediator. Your goal is to calm tension and encourage productive communication.

Conversation so far:
{conversation_prefix}

Instructions:
1. Acknowledge valid points from both sides without picking favorites.
2. Gently point out any unproductive behaviors (personal attacks, dismissiveness).
3. Suggest a constructive path forward (e.g., clarifying terms, providing evidence, finding common ground).
4. Keep your tone calm, respectful, and solution-oriented.

Provide your mediation message (2-4 sentences):"""

def generate_mediation(tokenizer, model, prefix_text, mode='steering'):
    """Generate a mediation message."""
    if mode == 'steering':
        prompt = STEERING_MEDIATION_PROMPT.format(conversation_prefix=prefix_text[:1500])
    else:
        prompt = JUDGMENT_MEDIATION_PROMPT.format(conversation_prefix=prefix_text[:1500])
    
    messages = [{"role": "user", "content": prompt}]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    # Extract mediation text
    if mode == 'judgment':
        try:
            import re
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                mediation_data = json.loads(json_match.group())
                return f"Verdict: {mediation_data.get('verdict', 'N/A')}\nRationale: {mediation_data.get('rationale', '')}"
        except:
            pass
    
    return response.strip()

def analyze_user_tone(messages, target_user, cut_index):
    """Analyze the target user's tone and stance."""
    user_messages = [msg for msg in messages[:cut_index] if msg.get('From') == target_user]
    
    if not user_messages:
        return "neutral and respectful"
    
    total_text = " ".join([msg.get('Reply_Text', '') for msg in user_messages])
    
    has_caps = sum(1 for word in total_text.split() if word.isupper() and len(word) > 2) > 0
    has_exclamations = total_text.count('!') > 2
    has_insults = any(word in total_text.lower() for word in ['idiot', 'stupid', 'dumb', 'ridiculous', 'absurd'])
    
    if has_caps or has_exclamations or has_insults:
        return "argumentative, somewhat aggressive, and emotionally charged"
    else:
        return "assertive but relatively measured"

USER_SIMULATION_PROMPT = """You are simulating a conversation participant's response. 

Conversation so far:
{conversation_prefix}

[MODERATOR INTERVENTION]
{mediation_text}

Instructions:
You are {target_user}. Based on your previous messages in this conversation, your communication style is: {persona}.

Now, write {target_user}'s response to the moderator's intervention. Your response should:
1. Stay true to {target_user}'s established tone and stance from earlier in the conversation
2. React naturally to the moderator's message (you might accept it, reject it, or partially engage with it)
3. Be realistic - not all users will immediately become calm after mediation

Write ONLY {target_user}'s next message (2-4 sentences):"""

def simulate_user_reply(tokenizer, model, prefix_text, mediation_text, target_user, persona):
    """Simulate how the target user would respond after seeing the mediation."""
    prompt = USER_SIMULATION_PROMPT.format(
        conversation_prefix=prefix_text[:1000],
        mediation_text=mediation_text,
        target_user=target_user,
        persona=persona
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()

def process_conversation(item, tokenizer, model, progress_counter, total):
    """Process a single conversation - generate mediations and simulate replies"""
    try:
        # Generate mediations
        steering_mediation = generate_mediation(tokenizer, model, item['prefix'], mode='steering')
        judgment_mediation = generate_mediation(tokenizer, model, item['prefix'], mode='judgment')
        
        # Analyze user tone
        persona = analyze_user_tone(item['messages'], item['target_user'], item['cut_index'])
        
        # Simulate replies
        steering_reply = simulate_user_reply(
            tokenizer, model, item['prefix'], steering_mediation, item['target_user'], persona
        )
        judgment_reply = simulate_user_reply(
            tokenizer, model, item['prefix'], judgment_mediation, item['target_user'], persona
        )
        
        result = {
            'post_id': item['post_id'],
            'cut_index': item['cut_index'],
            'target_user': item['target_user'],
            'persona': persona,
            'conversation_prefix': item['prefix'],
            'original_continuation': item['original_continuation'],
            'steering': {
                'mediation_text': steering_mediation,
                'simulated_reply': steering_reply
            },
            'judgment': {
                'mediation_text': judgment_mediation,
                'simulated_reply': judgment_reply
            }
        }
        
        # Thread-safe progress update
        with write_lock:
            progress_counter[0] += 1
            if progress_counter[0] % 10 == 0:
                print(f"Progress: {progress_counter[0]}/{total} conversations processed")
        
        return result
        
    except Exception as e:
        print(f"Error processing conversation {item['post_id']}: {e}")
        return None

def run_parallel_simulation(simulation_data, tokenizer, model, num_threads=4):
    """Run simulation with multiple threads"""
    print(f"\n{'='*80}")
    print(f"Starting parallel processing with {num_threads} threads")
    print(f"{'='*80}\n")
    
    results = []
    progress_counter = [0]  # Use list for mutable counter
    total = len(simulation_data)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_conversation, item, tokenizer, model, progress_counter, total): item
            for item in simulation_data
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_item):
            result = future.result()
            if result:
                results.append(result)
    
    print(f"\n✓ Completed processing {len(results)} conversations")
    return results

def save_results(results, output_file):
    """Save simulation results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {output_file}")

def display_case_studies(results, num_studies=3):
    """Display case studies showing mediation effectiveness"""
    print("\n" + "="*80)
    print("CASE STUDIES: Effectiveness of LLM-Produced Mediation")
    print("="*80)
    
    case_study_indices = [0, len(results)//2, len(results)-1] if len(results) >= 3 else [0]
    
    for idx, case_idx in enumerate(case_study_indices[:num_studies], 1):
        item = results[case_idx]
        
        print(f"\n{'='*80}")
        print(f"CASE STUDY #{idx}: Post ID: {item['post_id']}")
        print(f"{'='*80}")
        
        print(f"\n📝 CONVERSATION PREFIX (cut at message {item['cut_index']}):")
        print("-" * 80)
        print(item['conversation_prefix'][:500] + "..." if len(item['conversation_prefix']) > 500 else item['conversation_prefix'])
        
        print(f"\n👤 TARGET USER: {item['target_user']}")
        print(f"   Persona: {item['persona']}")
        
        print(f"\n📌 ORIGINAL CONTINUATION (without mediation):")
        print("-" * 80)
        print(item['original_continuation'][:300] if item['original_continuation'] else "[No continuation available]")
        
        print(f"\n🎯 STEERING MEDIATION:")
        print("-" * 80)
        print(item['steering']['mediation_text'][:300])
        
        print(f"\n💬 SIMULATED REPLY (after Steering):")
        print("-" * 80)
        print(item['steering']['simulated_reply'][:300])
        
        print(f"\n⚖️ JUDGMENT MEDIATION:")
        print("-" * 80)
        print(item['judgment']['mediation_text'][:300])
        
        print(f"\n💬 SIMULATED REPLY (after Judgment):")
        print("-" * 80)
        print(item['judgment']['simulated_reply'][:300])
        
        print("\n")

def main():
    parser = argparse.ArgumentParser(description='Part 4: User Simulator Evaluation - ASU SOL Version')
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to dataset directory containing JSON files')
    parser.add_argument('--output_file', type=str, default='user_simulation_results.json',
                        help='Output file path for results')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='Hugging Face authentication token (or set HF_TOKEN env variable)')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                        help='Model name from Hugging Face')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of parallel threads for processing')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Get HF token from args or environment
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    
    print("\n" + "="*80)
    print("CSE472 Project 2 - Part 4: User Simulator Evaluation")
    print("ASU SOL Supercomputer Version")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Threads: {args.num_threads}")
    print(f"Model: {args.model_name}")
    print("="*80 + "\n")
    
    # Setup
    setup_environment(hf_token, args.dataset_path)
    
    # Load data
    json_files = find_json_files(args.dataset_path)
    conversations = load_conversations(json_files)
    
    # Load model
    tokenizer, model = load_model(args.model_name, args.device)
    
    # Prepare simulation data
    simulation_data = prepare_simulation_data(conversations)
    
    # Run parallel simulation
    results = run_parallel_simulation(simulation_data, tokenizer, model, args.num_threads)
    
    # Save results
    save_results(results, args.output_file)
    
    # Display case studies
    display_case_studies(results)
    
    # Summary
    print("\n" + "="*80)
    print("Part 4 Complete: User Simulation Evaluation")
    print("="*80)
    print(f"✓ Processed {len(results)} conversations")
    print(f"✓ Generated {len(results) * 2} mediation interventions")
    print(f"✓ Simulated {len(results) * 2} user responses")
    print(f"✓ Results saved to: {args.output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
