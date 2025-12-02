import json
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Configuration
INPUT_FILE = "../output/mediation_outputs.json"
OUTPUT_FILE = "../output/multi_agent_debate_results.json"
JUDGE_MODEL_NAME = "Qwen/Qwen3-4B"

# Multi-Agent Debate Prompts
AGENT_A_SYSTEM = "You are Agent A, an expert mediator evaluator focused on **Neutrality and Fairness**."

AGENT_A_INITIAL_PROMPT = """Evaluate the following mediation based on neutrality and fairness criteria:
- Does it avoid taking sides?
- Does it treat both parties equally?
- Is the language unbiased and non-judgmental?

Conversation:
{conversation_text}

Mediation Output:
{mediation_text}

Provide your evaluation in strict JSON format:
{{
  "rationale": "<Your detailed analysis focusing on neutrality and fairness>",
  "score": <1-5>
}}"""

AGENT_B_SYSTEM = "You are Agent B, an expert mediator evaluator focused on **Effectiveness and Conflict Resolution**."

AGENT_B_INITIAL_PROMPT = """Evaluate the following mediation based on effectiveness and conflict resolution criteria:
- Does it de-escalate tension?
- Does it offer constructive solutions?
- Does it move the conversation forward productively?

Conversation:
{conversation_text}

Mediation Output:
{mediation_text}

Provide your evaluation in strict JSON format:
{{
  "rationale": "<Your detailed analysis focusing on effectiveness>",
  "score": <1-5>
}}"""

DEBATE_ROUND_PROMPT = """You have seen the other agent's evaluation. Review their perspective and refine your own assessment.

Other Agent's Evaluation:
Rationale: {other_rationale}
Score: {other_score}

Your Previous Evaluation:
Rationale: {my_rationale}
Score: {my_score}

Consider:
- Do you agree or disagree with the other agent's perspective?
- Has their analysis revealed aspects you missed?
- Should you adjust your score or maintain your position?

Provide your updated evaluation in strict JSON format:
{{
  "rationale": "<Your refined analysis, addressing the other agent's points>",
  "score": <1-5>
}}"""

FINAL_JUDGE_SYSTEM = "You are the Chief Justice, synthesizing multiple expert perspectives into a final verdict."

FINAL_JUDGE_PROMPT = """Review the debate between two expert evaluators and provide a final, balanced assessment.

Agent A (Neutrality & Fairness):
Round 1: Score {a1_score} - {a1_rationale}
Round 2: Score {a2_score} - {a2_rationale}

Agent B (Effectiveness & Conflict Resolution):
Round 1: Score {b1_score} - {b1_rationale}
Round 2: Score {b2_score} - {b2_rationale}

Original Context:
Conversation: {conversation_text}
Mediation: {mediation_text}

Synthesize both perspectives (neutrality and effectiveness) into a final evaluation.

Provide the final verdict in strict JSON format:
{{
  "rationale": "<Comprehensive final rationale considering both perspectives>",
  "score": <1-5>
}}"""

def extract_evaluation(generated_text):
    """Extract rationale and score from LLM output."""
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
    rationale_match = re.search(r'"rationale":\s*"(.*?)"', text, re.DOTALL) or \
                      re.search(r"'rationale':\s*'(.*?)'", text, re.DOTALL)
    
    score_match = re.search(r'"score":\s*(\d+)', text) or \
                  re.search(r"'score':\s*(\d+)", text)

    rationale = rationale_match.group(1) if rationale_match else "Unable to extract rationale"
    score = int(score_match.group(1)) if score_match else 3

    if rationale == "Unable to extract rationale":
        print(f"\n[DEBUG] Failed to extract from:\n{generated_text}\n[END DEBUG]\n")

    return {'rationale': rationale, 'score': score}

def generate_response(model, tokenizer, system_prompt, user_prompt):
    """Generate a response from the model with specific system and user prompts."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_input], return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()

def multi_agent_debate(model, tokenizer, conversation, mediation):
    """
    Run a multi-agent debate to evaluate mediation quality.
    
    Process:
    1. Agent A (Neutrality) and Agent B (Effectiveness) independently evaluate
    2. Each agent sees the other's evaluation and refines their own
    3. Final Judge synthesizes both perspectives
    """
    
    # Round 1: Independent Evaluations
    print("  Round 1: Independent evaluations...")
    
    # Agent A initial evaluation
    prompt_a1 = AGENT_A_INITIAL_PROMPT.format(
        conversation_text=conversation[:2000],
        mediation_text=mediation
    )
    resp_a1 = generate_response(model, tokenizer, AGENT_A_SYSTEM, prompt_a1)
    eval_a1 = extract_evaluation(resp_a1)
    
    # Agent B initial evaluation
    prompt_b1 = AGENT_B_INITIAL_PROMPT.format(
        conversation_text=conversation[:2000],
        mediation_text=mediation
    )
    resp_b1 = generate_response(model, tokenizer, AGENT_B_SYSTEM, prompt_b1)
    eval_b1 = extract_evaluation(resp_b1)
    
    print(f"    Agent A (Neutrality): Score {eval_a1['score']}")
    print(f"    Agent B (Effectiveness): Score {eval_b1['score']}")
    
    # Round 2: Debate/Refinement
    print("  Round 2: Cross-examination and refinement...")
    
    # Agent A sees Agent B's evaluation
    prompt_a2 = DEBATE_ROUND_PROMPT.format(
        other_rationale=eval_b1['rationale'],
        other_score=eval_b1['score'],
        my_rationale=eval_a1['rationale'],
        my_score=eval_a1['score']
    )
    resp_a2 = generate_response(model, tokenizer, AGENT_A_SYSTEM, prompt_a2)
    eval_a2 = extract_evaluation(resp_a2)
    
    # Agent B sees Agent A's evaluation
    prompt_b2 = DEBATE_ROUND_PROMPT.format(
        other_rationale=eval_a1['rationale'],
        other_score=eval_a1['score'],
        my_rationale=eval_b1['rationale'],
        my_score=eval_b1['score']
    )
    resp_b2 = generate_response(model, tokenizer, AGENT_B_SYSTEM, prompt_b2)
    eval_b2 = extract_evaluation(resp_b2)
    
    print(f"    Agent A (Refined): Score {eval_a2['score']}")
    print(f"    Agent B (Refined): Score {eval_b2['score']}")
    
    # Final Synthesis
    print("  Final: Chief Justice synthesis...")
    
    final_prompt = FINAL_JUDGE_PROMPT.format(
        a1_score=eval_a1['score'],
        a1_rationale=eval_a1['rationale'][:200],  # Truncate for context length
        a2_score=eval_a2['score'],
        a2_rationale=eval_a2['rationale'][:200],
        b1_score=eval_b1['score'],
        b1_rationale=eval_b1['rationale'][:200],
        b2_score=eval_b2['score'],
        b2_rationale=eval_b2['rationale'][:200],
        conversation_text=conversation[:1000],
        mediation_text=mediation[:500]
    )
    
    final_resp = generate_response(model, tokenizer, FINAL_JUDGE_SYSTEM, final_prompt)
    final_eval = extract_evaluation(final_resp)
    
    print(f"    Final Score: {final_eval['score']}")
    
    return {
        'round1_agent_a': eval_a1,
        'round1_agent_b': eval_b1,
        'round2_agent_a': eval_a2,
        'round2_agent_b': eval_b2,
        'final_evaluation': final_eval,
        'debate_summary': {
            'neutrality_scores': [eval_a1['score'], eval_a2['score']],
            'effectiveness_scores': [eval_b1['score'], eval_b2['score']],
            'final_score': final_eval['score']
        }
    }

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found. Please run part1_mediation.py first.")
        return

    print("=" * 60)
    print("Multi-Agent Debate Evaluation System")
    print("=" * 60)
    print("\nLoading judge model...")
    
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
    
    # Limit to 50 samples as requested
    mediation_data = mediation_data[:50]
    
    print(f"\nLoaded {len(mediation_data)} mediation outputs for evaluation.")
    print("\nStarting multi-agent debate evaluation...\n")

    results = []

    # Process each mediation output
    for i, item in enumerate(tqdm(mediation_data, desc="Multi-Agent Debate")):
        print(f"\n{'='*60}")
        print(f"Post ID: {item['post_id']} ({i+1}/{len(mediation_data)})")
        print(f"{'='*60}")
        
        debate_result = multi_agent_debate(
            model, tokenizer,
            item['conversation'],
            item['steering']
        )
        
        results.append({
            'post_id': item['post_id'],
            'method': 'multi-agent-debate',
            'debate_result': debate_result
        })

    # Save results
    output_data = {
        'evaluation_method': 'multi-agent-debate',
        'description': 'Multi-agent debate with Agent A (Neutrality) and Agent B (Effectiveness)',
        'results': results
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"{'='*60}")
    
    # Print summary statistics
    if results:
        final_scores = [r['debate_result']['final_evaluation']['score'] for r in results]
        avg_score = sum(final_scores) / len(final_scores)
        
        print("\n=== Summary Statistics ===")
        print(f"Total evaluations: {len(results)}")
        print(f"Average final score: {avg_score:.2f}")
        print(f"Score distribution: {dict((s, final_scores.count(s)) for s in range(1, 6))}")

if __name__ == "__main__":
    main()
