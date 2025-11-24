import json
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Configuration
INPUT_FILE = "../output/mediation_outputs.json"
OUTPUT_FILE = "../output/evaluation_results.json"
JUDGE_MODEL_NAME = "Qwen/Qwen3-4B"

EVALUATION_PROMPT = """Original Conversation:
{conversation_text}

Mediation Output to Evaluate:
{mediation_text}

Instructions:
Evaluate the mediation output on:
- Clarity
- Fairness / Neutrality
- Constructiveness (ability to de-escalate conflict)
- Alignment with the conversation context

Provide:
1. A short rationale explaining your evaluation.
2. A numeric score from 1 to 5:
   1 = Poor (biased, unclear, unhelpful)
   3 = Moderate (partially reasonable, needs improvement)
   5 = Excellent (clear, fair, constructive, context-aware)

Output Format (strict JSON):
{{
  "rationale": "<Your reasoning>",
  "score": <number>
}}"""

def extract_evaluation(generated_text):
    """Extract rationale and score from LLM output."""
    # Clean up the text
    text = generated_text.strip()
    
    # Remove <think> blocks (Chain of Thought)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Remove markdown code blocks if present
    if "```" in text:
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            
    try:
        # Try to find the JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            eval_dict = json.loads(json_str)
            return {
                'rationale': eval_dict.get('rationale', 'Unable to extract rationale'),
                'score': int(eval_dict.get('score', 3))
            }
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"Error parsing JSON: {e}")

    # Fallback: extract using regex
    # Look for "rationale": "..." or 'rationale': '...'
    rationale_match = re.search(r'"rationale":\s*"(.*?)"', text, re.DOTALL) or \
                      re.search(r"'rationale':\s*'(.*?)'", text, re.DOTALL)
    
    # Look for "score": <number>
    score_match = re.search(r'"score":\s*(\d+)', text) or \
                  re.search(r"'score':\s*(\d+)", text)

    rationale = rationale_match.group(1) if rationale_match else "Unable to extract rationale"
    score = int(score_match.group(1)) if score_match else 3

    if rationale == "Unable to extract rationale":
        print(f"\n[DEBUG] Failed to extract from:\n{generated_text}\n[END DEBUG]\n")

    return {'rationale': rationale, 'score': score}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found. Please run part1_mediation.py first.")
        return

    print("Loading judge model...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
    # Check for Flash Attention availability
    attn_implementation = "sdpa"
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        try:
            import flash_attn
            attn_implementation = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            print("Flash Attention 2 not installed, using SDPA")

    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=attn_implementation
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    with open(INPUT_FILE, 'r') as f:
        mediation_data = json.load(f)
    
    print(f"Loaded {len(mediation_data)} mediation outputs for evaluation.")

    evaluation_results = []
    BATCH_SIZE = 8

    for i in tqdm(range(0, len(mediation_data), BATCH_SIZE), desc="Evaluating Mediations"):
        batch = mediation_data[i:i + BATCH_SIZE]
        
        # Evaluate judgment
        judgment_prompts = []
        for item in batch:
            messages = [
                {"role": "system", "content": "You are an impartial evaluation judge."},
                {"role": "user", "content": EVALUATION_PROMPT.format(
                    conversation_text=item['conversation'][:2000], # Increased context slightly
                    mediation_text=item['judgment']
                )}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            judgment_prompts.append(prompt)
        
        inputs = tokenizer(judgment_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        judgment_evals_raw = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        judgment_evals = [extract_evaluation(text.strip()) for text in judgment_evals_raw]

        # Evaluate steering
        steering_prompts = []
        for item in batch:
            messages = [
                {"role": "system", "content": "You are an impartial evaluation judge."},
                {"role": "user", "content": EVALUATION_PROMPT.format(
                    conversation_text=item['conversation'][:2000],
                    mediation_text=item['steering']
                )}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            steering_prompts.append(prompt)
        
        inputs = tokenizer(steering_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        steering_evals_raw = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        steering_evals = [extract_evaluation(text.strip()) for text in steering_evals_raw]

        for j, item in enumerate(batch):
            print(f"\n--- Post ID: {item['post_id']} ---")
            print(f"Judgment Score: {judgment_evals[j]['score']}")
            print(f"Judgment Rationale: {judgment_evals[j]['rationale']}")
            print(f"Steering Score: {steering_evals[j]['score']}")
            print(f"Steering Rationale: {steering_evals[j]['rationale']}")
            print("-" * 50)

            evaluation_results.append({
                'post_id': item['post_id'],
                'judgment_evaluation': {
                    'rationale': judgment_evals[j]['rationale'],
                    'score': judgment_evals[j]['score']
                },
                'steering_evaluation': {
                    'rationale': steering_evals[j]['rationale'],
                    'score': steering_evals[j]['score']
                }
            })

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"Evaluation results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
