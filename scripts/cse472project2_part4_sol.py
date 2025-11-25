#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSE472 Project 2 - Part 4: User Simulator Evaluation
Single-model (Qwen) version with live logging and attention masks.
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

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# Global lock for thread-safe printing and logging
write_lock = Lock()


def strip_think_tags(text: str) -> str:
    """Remove optional <think>...</think> blocks that some models emit."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def setup_environment(hf_token, dataset_path):
    """Initialize Hugging Face login and validate dataset path."""
    print("=" * 80)
    print("Initializing Part 4: User Simulator Evaluation")
    print("=" * 80)

    if hf_token:
        print("Logging into Hugging Face Hub...")
        login(token=hf_token)
        print("✓ Hugging Face login successful")
    else:
        print("⚠ No Hugging Face token provided. Assuming public model access.")

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")

    print(f"✓ Dataset directory verified: {dataset_path}")
    return dataset_path


def find_json_files(dataset_path):
    """Find all JSON files in the dataset directory."""
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
        if "Reply_Text" in conversation_part and "From" in conversation_part:
            messages.append(
                {
                    "Reply_Text": conversation_part.get("Reply_Text", ""),
                    "From": conversation_part.get("From", "Unknown"),
                    "Reply_Score": conversation_part.get("Reply_Score", 0),
                    "Timestamp": conversation_part.get("Timestamp", None),
                }
            )
        # Recursively process replies
        if "Replies" in conversation_part:
            replies = conversation_part["Replies"]
            if isinstance(replies, list):
                for reply in replies:
                    messages.extend(extract_messages(reply))
            elif isinstance(replies, dict):
                messages.extend(extract_messages(replies))

    elif isinstance(conversation_part, list):
        for part in conversation_part:
            messages.extend(extract_messages(part))

    return messages


def load_conversations(json_files):
    """Load and process conversations from JSON files."""
    print("\nLoading and processing conversations from JSON files...")
    conversations_by_post = {}

    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            post_id = data.get(
                "Post_id", os.path.basename(file_path).replace(".json", "")
            )

            if "Threads" in data:
                thread_messages = extract_messages(data["Threads"])
                if len(thread_messages) >= 2:
                    conversations_by_post[post_id] = thread_messages
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"✓ Loaded {len(conversations_by_post)} conversations")
    return conversations_by_post


def load_model(model_name, device):
    """Load the LLM model with GPU optimization."""
    print(f"Loading model '{model_name}' on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    print(f"✓ Model loaded successfully on {device}")
    return tokenizer, model


def select_cut_point(messages):
    """
    Select a reasonable cut point in the conversation where we will "stop"
    and then perform mediation + user simulation.
    """
    n = len(messages)
    if n < 4:
        return None

    # Cut in later portion of the conversation (60–80%)
    min_cut = max(3, int(n * 0.6))
    max_cut = min(n - 1, int(n * 0.8))

    if min_cut >= max_cut:
        # Fallback: three messages before the end
        return n - 1 if n > 3 else None

    return random.randint(min_cut, max_cut)


def get_conversation_prefix(messages, cut_index):
    """
    Build a conversation prefix up to (but not including) the message at cut_index.
    Also identifies the target_user as the author of the message at cut_index.
    """
    if cut_index is None or cut_index >= len(messages):
        return None, None

    prefix_msgs = messages[:cut_index]
    target_msg = messages[cut_index]
    target_user = target_msg.get("From", "Unknown")

    lines = []
    for msg in prefix_msgs:
        author = msg.get("From", "Unknown")
        text = msg.get("Reply_Text", "").strip().replace("\n", " ")
        lines.append(f"{author}: {text}")

    return "\n".join(lines), target_user


def get_original_continuation(messages, cut_index):
    """Return the actual message at the cut point (the ground truth continuation)."""
    if cut_index is None or cut_index >= len(messages):
        return None

    msg = messages[cut_index]
    author = msg.get("From", "Unknown")
    text = msg.get("Reply_Text", "").strip().replace("\n", " ")
    return f"{author}: {text}"


def prepare_simulation_data(conversations_by_post):
    """
    Prepare a list of simulation items, each containing:
    - post_id
    - messages
    - cut_index
    - prefix
    - target_user
    - original_continuation
    """
    print("\nPreparing simulation data...")

    simulation_data = []
    skipped_too_short = 0
    skipped_no_cut = 0

    for post_id, messages in conversations_by_post.items():
        if len(messages) < 4:
            skipped_too_short += 1
            continue

        cut_index = select_cut_point(messages)
        if cut_index is None:
            skipped_no_cut += 1
            continue

        prefix, target_user = get_conversation_prefix(messages, cut_index)
        orig_continuation = get_original_continuation(messages, cut_index)

        if not prefix or not target_user or not orig_continuation:
            skipped_no_cut += 1
            continue

        simulation_data.append(
            {
                "post_id": post_id,
                "messages": messages,
                "cut_index": cut_index,
                "prefix": prefix,
                "target_user": target_user,
                "original_continuation": orig_continuation,
            }
        )

    print(f"✓ Prepared {len(simulation_data)} simulation items")
    print(f"  Skipped {skipped_too_short} conversations (too short)")
    print(f"  Skipped {skipped_no_cut} conversations (no valid cut)")
    return simulation_data


STEERING_MEDIATION_PROMPT = """You are a conflict mediator in an online discussion forum. 
Your job is to calmly intervene at tense moments in the conversation.

Below is a conversation between multiple users. The discussion has become tense or unproductive.

CONVERSATION SO FAR:
{conversation_prefix}

YOUR TASK:
Write a short, calm mediation message that:
1. Acknowledges both sides of the disagreement
2. Points out any unproductive or hostile behavior in a neutral way
3. Suggests a more constructive way forward in the conversation
4. Encourages empathy, understanding, or de-escalation

Your tone should be:
- Neutral
- Respectful
- Calm
- Non-judgmental

Write this as if you are a forum moderator stepping into the thread.
"""


JUDGMENT_MEDIATION_PROMPT = """You are a neutral moderator reviewing an online discussion.

Below is a conversation between multiple users that has become tense or hostile.

CONVERSATION SO FAR:
{conversation_prefix}

YOUR TASK:
1. Identify what the main disagreement is about.
2. Judge which participant(s) are being more reasonable and constructive.
3. Explain your reasoning in a clear and concise way.

Then, produce a structured JSON-style output with the following keys:
- "verdict": A short phrase naming who is more reasonable (e.g., "UserA", "UserB", "both", "neither").
- "rationale": 2-4 sentences explaining your reasoning.

Example output:
{{
  "verdict": "UserA",
  "rationale": "UserA stays focused on the argument, provides evidence, and avoids personal attacks. UserB repeatedly uses insults."
}}
"""


def _encode_chat(tokenizer, messages, model):
    """
    Helper: apply chat template to messages and tokenize with attention_mask.
    Returns (input_ids, attention_mask).
    """
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    return encoded["input_ids"], encoded["attention_mask"]


def generate_mediation(tokenizer, model, prefix_text, mode="steering"):
    """
    Generate a mediation message.

    For 'steering' we allow more creativity.
    For 'judgment' we reduce randomness to make JSON-style output more stable.
    """
    if mode == "steering":
        base_prompt = STEERING_MEDIATION_PROMPT.format(
            conversation_prefix=prefix_text[-2500:]
        )
    else:
        base_prompt = JUDGMENT_MEDIATION_PROMPT.format(
            conversation_prefix=prefix_text[-2500:]
        )

    prompt = (
        base_prompt
        + "\n\nDo NOT include <think> or any hidden reasoning. "
        "Only output the final mediation message."
    )

    messages = [{"role": "user", "content": prompt}]
    input_ids, attention_mask = _encode_chat(tokenizer, messages, model)

    if mode == "judgment":
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2056,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2056,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][input_ids.shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    text = strip_think_tags(text)

    if mode == "judgment":
        # Try to parse JSON-style verdict/rationale
        try:
            import re

            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
            if json_match:
                mediation_data = json.loads(json_match.group())
                verdict = mediation_data.get("verdict", "N/A")
                rationale = mediation_data.get("rationale", "")
                return f"Verdict: {verdict}\nRationale: {rationale}"
        except Exception:
            pass

    return text.strip()


PERSONA_ANALYSIS_PROMPT = """You are analyzing a conversation participant's behavior.

TARGET USER: {target_user}

Here are some of their previous messages in this thread:

{user_messages}

Based on these messages, write a concise description of this user's communication style and emotional tone.
Focus on:
- Their tone (e.g., calm, aggressive, sarcastic, defensive, measured, emotional)
- Their approach to disagreement (e.g., evidence-based, dismissive, conciliatory)
- Any noticeable patterns (e.g., uses insults, tries to de-escalate, frequently changes topic)

Output your analysis in one or two sentences beginning with:
"Overall, this user tends to be ..."

You may optionally wrap the result in a JSON object with a "persona" field, e.g.:
{{
  "persona": "Overall, this user tends to be..."
}}
"""


def analyze_user_tone(tokenizer, model, messages, target_user, cut_index):
    """
    Analyze the target user's tone and stance using LLM-based analysis.
    """
    user_messages = [m for m in messages[:cut_index] if m.get("From") == target_user]

    if not user_messages:
        return "neutral and respectful"

    formatted = []
    for i, msg in enumerate(user_messages[-10:], 1):
        text = msg.get("Reply_Text", "").strip().replace("\n", " ")
        score = msg.get("Reply_Score", 0)
        formatted.append(f"Message {i} (score={score}): {text}")

    messages_str = "\n".join(formatted)
    prompt = PERSONA_ANALYSIS_PROMPT.format(
        target_user=target_user, user_messages=messages_str
    )
    prompt += (
        "\n\nDo NOT include <think> or any hidden reasoning. "
        "Only output the final persona description."
    )

    messages_input = [{"role": "user", "content": prompt}]

    try:
        input_ids, attention_mask = _encode_chat(tokenizer, messages_input, model)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated = outputs[0][input_ids.shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        text = strip_think_tags(text)
        persona_text = text.strip()

        # Try to recover JSON-style {"persona": "..."}
        if "{" in persona_text and "}" in persona_text:
            try:
                start_idx = persona_text.index("{")
                end_idx = persona_text.rindex("}") + 1
                maybe_json = persona_text[start_idx:end_idx]
                obj = json.loads(maybe_json)
                if isinstance(obj, dict) and "persona" in obj:
                    persona_text = str(obj["persona"]).strip()
            except Exception:
                pass

        if not persona_text:
            raise ValueError("Empty persona after decoding")

        return persona_text

    except Exception as e:
        print(f"Error generating persona for {target_user}: {e}")
        total_text = " ".join(m.get("Reply_Text", "") for m in user_messages)
        has_caps = any(
            w.isupper() and len(w) > 2 for w in total_text.split()
        )
        has_excl = total_text.count("!") > 2

        if has_caps or has_excl:
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

Write ONLY {target_user}'s next message:"""


def simulate_user_reply(tokenizer, model, prefix_text, mediation_text, target_user, persona):
    """Simulate how the target user would respond after seeing the mediation."""
    prompt = USER_SIMULATION_PROMPT.format(
        conversation_prefix=prefix_text[-1500:],
        mediation_text=mediation_text,
        target_user=target_user,
        persona=persona,
    )
    prompt += (
        "\n\nDo NOT include <think> or any hidden reasoning. "
        "Only output {target_user}'s final reply."
    )

    messages = [{"role": "user", "content": prompt}]
    input_ids, attention_mask = _encode_chat(tokenizer, messages, model)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][input_ids.shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    text = strip_think_tags(text)
    return text.strip()


def process_conversation(
    item,
    tokenizer,
    model,
    progress_counter,
    total,
    log_file=None,
    live_console_logs=False,
    log_every=1,
):
    """
    Process a single conversation - generate mediations and simulate replies.
    Also supports on-the-fly logging to console and/or a JSONL log file.
    """
    try:
        steering_mediation = generate_mediation(
            tokenizer, model, item["prefix"], mode="steering"
        )
        judgment_mediation = generate_mediation(
            tokenizer, model, item["prefix"], mode="judgment"
        )

        persona = analyze_user_tone(
            tokenizer, model, item["messages"], item["target_user"], item["cut_index"]
        )

        steering_reply = simulate_user_reply(
            tokenizer,
            model,
            item["prefix"],
            steering_mediation,
            item["target_user"],
            persona,
        )
        judgment_reply = simulate_user_reply(
            tokenizer,
            model,
            item["prefix"],
            judgment_mediation,
            item["target_user"],
            persona,
        )

        result = {
            "post_id": item["post_id"],
            "cut_index": item["cut_index"],
            "target_user": item["target_user"],
            "persona": persona,
            "conversation_prefix": item["prefix"],
            "original_continuation": item["original_continuation"],
            "steering": {
                "mediation_text": steering_mediation,
                "simulated_reply": steering_reply,
            },
            "judgment": {
                "mediation_text": judgment_mediation,
                "simulated_reply": judgment_reply,
            },
        }

        with write_lock:
            progress_counter[0] += 1
            idx = progress_counter[0]

            if idx % 10 == 0:
                print(f"Progress: {idx}/{total} conversations processed")

            if log_every < 1:
                log_every = 1

            if log_file and (idx % log_every == 0):
                try:
                    with open(log_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
                except Exception as e:
                    print(f"Error writing to log file {log_file}: {e}")

            if live_console_logs and (idx % log_every == 0):
                print("\n" + "-" * 80)
                print(f"[LIVE LOG {idx}/{total}] Post ID: {result['post_id']}")
                print(f"Target user: {result['target_user']}")
                print(f"Persona: {result['persona']}")
                print("\nSteering mediation (first 200 chars):")
                print(result["steering"]["mediation_text"][:200])
                print("\nSteering reply (first 200 chars):")
                print(result["steering"]["simulated_reply"][:200])
                print("-" * 80 + "\n")

        return result

    except Exception as e:
        print(f"Error processing conversation {item['post_id']}: {e}")
        return None


def run_parallel_simulation(
    simulation_data,
    tokenizer,
    model,
    num_threads=4,
    log_file=None,
    live_console_logs=False,
    log_every=1,
):
    """Run simulation with multiple threads using a single shared model."""
    print("\n" + "=" * 80)
    print(f"Starting parallel processing with {num_threads} threads")
    print("=" * 80 + "\n")

    if log_file:
        print(f"Streaming logs to file: {log_file} (JSONL; one conversation per line)")
    if live_console_logs:
        print(f"Live console logs enabled (logging every {log_every} conversation(s))")

    results = []
    progress_counter = [0]
    total = len(simulation_data)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_item = {
            executor.submit(
                process_conversation,
                item,
                tokenizer,
                model,
                progress_counter,
                total,
                log_file,
                live_console_logs,
                log_every,
            ): item
            for item in simulation_data
        }

        for future in as_completed(future_to_item):
            result = future.result()
            if result:
                results.append(result)

    print(f"\n✓ Completed processing {len(results)} conversations")
    return results


def save_results(results, output_file):
    """Save simulation results to JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {output_file}")


def display_case_studies(results, num_studies=3):
    """Display case studies showing mediation effectiveness."""
    if not results:
        print("No results to display for case studies.")
        return

    print("\n" + "=" * 80)
    print("CASE STUDIES: Mediation and Simulated User Responses")
    print("=" * 80)

    if len(results) <= num_studies:
        indices = list(range(len(results)))
    else:
        indices = [0, len(results) // 2, len(results) - 1][:num_studies]

    for idx, i in enumerate(indices, 1):
        item = results[i]
        print(f"\n--- Case Study {idx} (Post ID: {item['post_id']}) ---")
        print("\nConversation prefix (before mediation):")
        print("-" * 60)
        print(item["conversation_prefix"])
        print("-" * 60)

        print(f"\nTarget user: {item['target_user']}")
        print(f"Persona: {item['persona']}\n")

        print("Original continuation (ground truth):")
        print(item["original_continuation"])
        print("\n[STEERING MEDIATION & SIMULATED REPLY]")
        print("Mediation:")
        print(item["steering"]["mediation_text"])
        print("\nSimulated reply:")
        print(item["steering"]["simulated_reply"])

        print("\n[JUDGMENT MEDIATION & SIMULATED REPLY]")
        print("Mediation:")
        print(item["judgment"]["mediation_text"])
        print("\nSimulated reply:")
        print(item["judgment"]["simulated_reply"])
        print("\n" + "-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Part 4: User Simulator Evaluation - ASU SOL Version"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset directory containing JSON files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="user_simulation_results.json",
        help="Output file path for final aggregated results",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face authentication token (or set HF_TOKEN env variable)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=(
            "Hugging Face model name used for all tasks "
            "(mediation, judgment, persona, simulation)"
        ),
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for parallel processing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to run models on (e.g., "cuda" or "cpu")',
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help=(
            "Optional path to a JSONL log file; each processed conversation "
            "is appended as one line"
        ),
    )
    parser.add_argument(
        "--live_console_logs",
        action="store_true",
        help=(
            "If set, print a short summary of selected conversations to the "
            "console as they finish"
        ),
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Log every Nth conversation (for both log_file and console). "
        "Default: 1 (log all).",
    )

    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)

    print("=" * 80)
    print("CSE472 Project 2 - Part 4: User Simulator Evaluation (Single-Model Version)")
    print("=" * 80)
    print(f"Using device: {args.device}")
    print(f"Using {args.num_threads} threads for parallel processing")
    print(f"Model: {args.model_name}")
    if args.log_file:
        print(f"Log file: {args.log_file}")
    print(f"Live console logs: {args.live_console_logs}")
    print(f"Log every: {args.log_every} conversation(s)")
    print("=" * 80)

    # 1. Environment and dataset
    dataset_path = setup_environment(hf_token, args.dataset_path)
    json_files = find_json_files(dataset_path)
    conversations_by_post = load_conversations(json_files)

    if not conversations_by_post:
        print("No conversations loaded. Exiting.")
        return

    # 2. Load model
    tokenizer, model = load_model(args.model_name, args.device)

    # 3. Prepare simulation data
    simulation_data = prepare_simulation_data(conversations_by_post)

    if not simulation_data:
        print("No valid simulation data could be prepared. Exiting.")
        return

    # 4. Parallel simulation
    results = run_parallel_simulation(
        simulation_data,
        tokenizer,
        model,
        args.num_threads,
        log_file=args.log_file,
        live_console_logs=args.live_console_logs,
        log_every=args.log_every,
    )

    # 5. Save final results
    save_results(results, args.output_file)

    # 6. Display case studies
    display_case_studies(results)

    print("\n" + "=" * 80)
    print("Part 4 Complete: User Simulation Evaluation")
    print("=" * 80)
    print(f"✓ Processed {len(results)} conversations")
    print(f"✓ Generated {len(results) * 2} mediation interventions")
    print(f"✓ Simulated {len(results) * 2} user responses")
    print(f"✓ Aggregated results saved to: {args.output_file}")
    if args.log_file:
        print(f"✓ Streaming log file (JSONL) at: {args.log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

