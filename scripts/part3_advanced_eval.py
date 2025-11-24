import json
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Configuration
INPUT_FILE = "../output/mediation_outputs.json"
OUTPUT_FILE = "../output/advanced_evaluation_results.json"
JUDGE_MODEL_NAME = "Qwen/Qwen3-4B"

# Few-shot examples
FEW_SHOT_EXAMPLES = [
    {
        'conversation': """User_A: Your argument makes no sense. You clearly don't understand basic economics.

User_B: Actually, if you'd read any reputable source, you'd know I'm right. Your stubbornness is showing.""",
        'mediation': """I appreciate both of you sharing your perspectives on this economic topic. User_A, you've raised valid concerns about the argument's clarity. User_B, you've pointed to external sources to support your view. Rather than questioning each other's understanding, could we focus on the specific economic principles at hand? Perhaps each of you could cite one key source or example that supports your position, so we can have a more productive discussion.""",
        'rationale': "This mediation effectively acknowledges both participants' points without taking sides. It redirects from personal attacks to substantive discussion by suggesting concrete next steps (citing sources). The tone is calm and constructive, encouraging evidence-based dialogue.",
        'score': 5
    },
    {
        'conversation': """User_C: This policy is ridiculous and anyone who supports it is an idiot.

User_D: Well, you're clearly too biased to see the benefits. Typical.""",
        'mediation': """Let's take a step back from the personal characterizations. Both of you seem passionate about this policy issue. User_C, could you explain what specific aspects of the policy concern you? User_D, what benefits do you see? Understanding the concrete pros and cons might help us move forward.""",
        'rationale': "The mediation acknowledges passion but doesn't fully address the hostile language used. While it attempts redirection toward substance, it could be more explicit about the need for respectful discourse. The suggestion to discuss specifics is good but feels somewhat generic.",
        'score': 3
    }
]

FEW_SHOT_EXAMPLES_TEMPLATE = """Below are examples of high-quality evaluations:

Example 1:
Conversation:
{example1_conv}
Mediation:
{example1_med}
Evaluation Output:
{{
  "rationale": "{example1_rationale}",
  "score": {example1_score}
}}

Example 2:
Conversation:
{example2_conv}
Mediation:
{example2_med}
Evaluation Output:
{{
  "rationale": "{example2_rationale}",
  "score": {example2_score}
}}"""

FEW_SHOT_TASK_TEMPLATE = """
Now evaluate the following mediation using the same criteria and style:

Conversation:
{conversation_text}

Mediation Output:
{mediation_text}

Output Format (strict JSON):
{{
  "rationale": "<Your reasoning>",
  "score": <number>
}}"""

RUBRIC_EVAL_PROMPT = """Use the following rubric to evaluate the mediation:

Rubric Criteria:
- **Neutrality (0–2 points):** Does not take sides; avoids accusatory language.
- **Clarity (0–1 point):** Message is understandable, well-structured, and concise.
- **Constructiveness (0–2 points):** Encourages calm, collaboration, and future progress.

Total Score = Neutrality + Clarity + Constructiveness (range 0–5)

Now evaluate the mediation below according to the rubric.

Conversation:
{conversation_text}

Mediation Output:
{mediation_text}

Output Format (strict JSON):
{{
  "rationale": "<Explain how each rubric category applies>",
  "score": <0-5>
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

    fewshot_results = []
    rubric_results = []
    BATCH_SIZE = 8

    # Format few-shot examples string
    few_shot_examples_str = FEW_SHOT_EXAMPLES_TEMPLATE.format(
        example1_conv=FEW_SHOT_EXAMPLES[0]['conversation'],
        example1_med=FEW_SHOT_EXAMPLES[0]['mediation'],
        example1_rationale=FEW_SHOT_EXAMPLES[0]['rationale'],
        example1_score=FEW_SHOT_EXAMPLES[0]['score'],
        example2_conv=FEW_SHOT_EXAMPLES[1]['conversation'],
        example2_med=FEW_SHOT_EXAMPLES[1]['mediation'],
        example2_rationale=FEW_SHOT_EXAMPLES[1]['rationale'],
        example2_score=FEW_SHOT_EXAMPLES[1]['score']
    )

    for i in tqdm(range(0, len(mediation_data), BATCH_SIZE), desc="Advanced Evaluation"):
        batch = mediation_data[i:i + BATCH_SIZE]

        # Few-shot evaluation
        fewshot_prompts = []
        for item in batch:
            task_str = FEW_SHOT_TASK_TEMPLATE.format(
                conversation_text=item['conversation'][:2000],
                mediation_text=item['steering']
            )
            full_prompt = few_shot_examples_str + task_str
            
            messages = [
                {"role": "system", "content": "You are an impartial evaluation judge."},
                {"role": "user", "content": full_prompt}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            fewshot_prompts.append(prompt)
        
        inputs = tokenizer(fewshot_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        fewshot_evals_raw = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for j, item in enumerate(batch):
            eval_result = extract_evaluation(fewshot_evals_raw[j].strip())
            print(f"\n--- Post ID: {item['post_id']} (Few-Shot) ---")
            print(f"Score: {eval_result['score']}")
            print(f"Rationale: {eval_result['rationale']}")
            print("-" * 30)
            
            fewshot_results.append({
                'post_id': item['post_id'],
                'method': 'few-shot',
                'evaluation': eval_result
            })

        # Rubric evaluation
        rubric_prompts = []
        for item in batch:
            messages = [
                {"role": "system", "content": "You are an impartial evaluation judge."},
                {"role": "user", "content": RUBRIC_EVAL_PROMPT.format(
                    conversation_text=item['conversation'][:2000],
                    mediation_text=item['steering']
                )}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            rubric_prompts.append(prompt)
        
        inputs = tokenizer(rubric_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        rubric_evals_raw = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for j, item in enumerate(batch):
            eval_result = extract_evaluation(rubric_evals_raw[j].strip())
            print(f"\n--- Post ID: {item['post_id']} (Rubric) ---")
            print(f"Score: {eval_result['score']}")
            print(f"Rationale: {eval_result['rationale']}")
            print("-" * 50)

            rubric_results.append({
                'post_id': item['post_id'],
                'method': 'rubric-based',
                'evaluation': eval_result
            })

    # Save results
    advanced_results = {
        'few_shot': fewshot_results,
        'rubric_based': rubric_results
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(advanced_results, f, indent=2)

    print(f"Advanced evaluation results saved to {OUTPUT_FILE}")
    
    # Print comparison stats
    if fewshot_results and rubric_results:
        fs_avg = sum(r['evaluation']['score'] for r in fewshot_results) / len(fewshot_results)
        rb_avg = sum(r['evaluation']['score'] for r in rubric_results) / len(rubric_results)
        print("\n=== Comparison of Evaluation Methods ===")
        print(f"Few-Shot Average Score: {fs_avg:.2f}")
        print(f"Rubric-Based Average Score: {rb_avg:.2f}")

if __name__ == "__main__":
    main()
