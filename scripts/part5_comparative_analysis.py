#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSE472 Project 2 - Part 5: Comparative Analysis
LLM-based scoring of mediation quality and reply toxicity
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Set visualization style
sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 18

# Color scheme
COLORS = {
    'human': '#3498db',      # Blue
    'steering': '#2ecc71',   # Green
    'judgment': '#e74c3c',   # Red
    'original': '#95a5a6'    # Gray
}

# Thread-safe lock
write_lock = Lock()

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Part 5: Comparative Analysis - LLM-based Scoring'
    )
    
    parser.add_argument(
        '--llm_results',
        type=str,
        required=True,
        help='Path to Part 4 results JSON file'
    )
    
    parser.add_argument(
        '--human_data',
        type=str,
        required=True,
        help='Path to human mediation data file (TSV/CSV)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./part5_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--qwen_model',
        type=str,
        default='Qwen/Qwen2.5-3B-Instruct',
        help='Qwen model for scoring'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    parser.add_argument(
        '--num_threads',
        type=int,
        default=4,
        help='Number of parallel threads for scoring'
    )
    
    parser.add_argument(
        '--use_cache',
        action='store_true',
        help='Use cached scores if available'
    )
    
    return parser.parse_args()

def setup_output_directories(output_dir):
    """Create output directory structure"""
    paths = {
        'root': Path(output_dir),
        'data': Path(output_dir) / 'data',
        'visualizations': Path(output_dir) / 'visualizations',
        'report': Path(output_dir) / 'report',
        'cache': Path(output_dir) / 'cache'
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created output directory structure in {output_dir}")
    return paths

def load_llm_results(filepath):
    """Load Part 4 LLM simulation results"""
    print(f"\nLoading LLM results from {filepath}...")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} conversations from Part 4")
    
    # Validate structure
    required_keys = ['post_id', 'steering', 'judgment', 'original_continuation']
    sample = data[0] if data else {}
    
    for key in required_keys:
        if key not in sample:
            raise ValueError(f"Missing required key '{key}' in LLM results")
    
    return data

def load_human_data(filepath):
    """Load human mediation data"""
    print(f"\nLoading human mediation data from {filepath}...")
    
    # Detect file format
    if filepath.endswith('.tsv'):
        df = pd.read_csv(filepath, sep='\t')
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        # Try both
        try:
            df = pd.read_csv(filepath, sep='\t')
        except:
            df = pd.read_csv(filepath)
    
    print(f"✓ Loaded {len(df)} rows from human data")
    print(f"  Columns: {list(df.columns)}")
    
    # Parse response column (array of strings)
    if 'response' in df.columns:
        # Response might be string representation of list
        df['response_parsed'] = df['response'].apply(parse_response_array)
        # Aggregate multiple mediations into one
        df['mediation_text'] = df['response_parsed'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else str(x)
        )
    else:
        raise ValueError("Human data must have 'response' column")
    
    print(f"✓ Parsed {len(df)} human mediations")
    
    return df

def parse_response_array(response_str):
    """Parse response column which may be string representation of array"""
    if pd.isna(response_str):
        return []
    
    # If already a list
    if isinstance(response_str, list):
        return response_str
    
    # If string representation of list
    if isinstance(response_str, str):
        try:
            # Try JSON parsing
            import ast
            return ast.literal_eval(response_str)
        except:
            # If fails, treat as single string
            return [response_str]
    
    return [str(response_str)]

def load_model(model_name, device):
    """Load Qwen model for scoring"""
    print(f"\nLoading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Fix pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    print(f"✓ Model loaded successfully")
    
    return tokenizer, model

def main():
    """Main execution function"""
    print("="*80)
    print("CSE472 Project 2 - Part 5: Comparative Analysis")
    print("LLM-Based Scoring System")
    print("="*80)
    
    # Parse arguments
    args = parse_arguments()
    
    print(f"\nConfiguration:")
    print(f"  LLM Results: {args.llm_results}")
    print(f"  Human Data: {args.human_data}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Model: {args.qwen_model}")
    print(f"  Device: {args.device}")
    print(f"  Threads: {args.num_threads}")
    print(f"  Use Cache: {args.use_cache}")
    
    # Setup output directories
    paths = setup_output_directories(args.output_dir)
    
    # Load data
    llm_data = load_llm_results(args.llm_results)
    human_data = load_human_data(args.human_data)
    
    # Load model
    tokenizer, model = load_model(args.qwen_model, args.device)
    
    print("\n" + "="*80)
    print("✓ Setup Complete - Ready for Analysis")
    print("="*80)
    
    # ===========================================================================
    # ANALYSIS 1: MEDIATION QUALITY
    # ===========================================================================
    
    mediation_scores = analyze_mediation_quality(
        llm_data,
        human_data,
        tokenizer,
        model,
        paths['cache'],
        args.use_cache
    )
    
    mediation_stats = calculate_mediation_statistics(mediation_scores)
    mediation_tests = perform_statistical_tests_mediations(mediation_scores)
    
    print("\n" + "="*80)
    print("MEDIATION QUALITY STATISTICS")
    print("="*80)
    print(mediation_stats.to_string(index=False))
    
    # Save mediation statistics
    mediation_stats.to_csv(paths['data'] / 'mediation_statistics.csv', index=False)
    print(f"\n✓ Saved mediation statistics to {paths['data'] / 'mediation_statistics.csv'}")
    
    # ===========================================================================
    # ANALYSIS 2: REPLY QUALITY
    # ===========================================================================
    
    reply_scores = analyze_reply_quality(
        llm_data,
        tokenizer,
        model,
        paths['cache'],
        args.use_cache
    )
    
    reply_stats = calculate_reply_statistics(reply_scores)
    reply_improvements = calculate_improvement_metrics(reply_scores)
    reply_tests = perform_statistical_tests_replies(reply_scores)
    
    print("\n" + "="*80)
    print("REPLY QUALITY STATISTICS")
    print("="*80)
    print(reply_stats.to_string(index=False))
    
    print("\n" + "="*80)
    print("IMPROVEMENT METRICS")
    print("="*80)
    print(reply_improvements.to_string(index=False))
    
    # Save reply statistics
    reply_stats.to_csv(paths['data'] / 'reply_statistics.csv', index=False)
    reply_improvements.to_csv(paths['data'] / 'improvement_metrics.csv', index=False)
    print(f"\n✓ Saved reply statistics to {paths['data'] / 'reply_statistics.csv'}")
    print(f"✓ Saved improvement metrics to {paths['data'] / 'improvement_metrics.csv'}")
    
    # ===========================================================================
    # SAVE COMBINED RESULTS
    # ===========================================================================
    
    combined_results = {
        'mediation_quality': {
            'scores': mediation_scores,
            'statistics': mediation_stats.to_dict('records'),
            'statistical_tests': {
                k: {'statistic': float(v.statistic), 'pvalue': float(v.pvalue)}
                for k, v in mediation_tests.items()
            }
        },
        'reply_quality': {
            'scores': reply_scores,
            'statistics': reply_stats.to_dict('records'),
            'improvements': reply_improvements.to_dict('records'),
            'statistical_tests': {
                k: {'statistic': float(v.statistic), 'pvalue': float(v.pvalue)}
                for k, v in reply_tests.items()
            }
        }
    }
    
    results_file = paths['data'] / 'combined_results.json'
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n✓ Saved combined results to {results_file}")
    
    # ===========================================================================
    # VISUALIZATIONS
    # ===========================================================================
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    create_mediation_quality_visualizations(
        mediation_scores,
        mediation_stats,
        paths['visualizations']
    )
    
    create_reply_quality_visualizations(
        reply_scores,
        reply_stats,
        reply_improvements,
        paths['visualizations']
    )
    
    create_dashboard_and_tables(
        mediation_stats,
        reply_stats,
        reply_improvements,
        paths['visualizations']
    )
    
    print("\n✓ All visualizations created successfully")
    
    # ===========================================================================
    # FINAL SUMMARY
    # ===========================================================================
    
    print("\n" + "="*80)
    print("PART 5 ANALYSIS COMPLETE")
    print("="*80)
    print(f"✓ Mediation quality analysis complete")
    print(f"✓ Reply quality analysis complete")
    print(f"✓ Statistical tests performed")
    print(f"✓ All visualizations generated")
    print(f"✓ Results saved to {paths['root']}")
    print("="*80)

# ============================================================================
# TASK 2: SCORING PROMPTS
# ============================================================================

MEDIATION_SCORING_PROMPT = """You are an expert evaluator of online moderation quality. Score the following moderator intervention on a scale of 1-10 for each criterion.

Moderator Message:
{mediation_text}

Score the following (1=Poor, 10=Excellent):

1. NEUTRALITY: Does the moderator avoid taking sides or showing bias?
   - 1-3: Clearly biased, favors one party
   - 4-6: Somewhat neutral, minor bias detected
   - 7-10: Completely neutral and balanced

2. CONSTRUCTIVENESS: Does it provide actionable guidance or just criticize?
   - 1-3: Only points out problems, no solutions offered
   - 4-6: Some suggestions present, but mostly criticism
   - 7-10: Clear actionable steps and constructive path forward

3. PROFESSIONALISM: Is the tone appropriate and respectful?
   - 1-3: Condescending, harsh, or inappropriate tone
   - 4-6: Acceptable but could be warmer or more professional
   - 7-10: Calm, respectful, and highly appropriate

4. CLARITY: Is the message well-structured and easy to understand?
   - 1-3: Confusing, rambling, or unclear
   - 4-6: Understandable but could be clearer
   - 7-10: Clear, well-organized, and easy to follow

CRITICAL: Output ONLY a valid JSON object with no additional text:
{{
  "neutrality": <score 1-10>,
  "constructiveness": <score 1-10>,
  "professionalism": <score 1-10>,
  "clarity": <score 1-10>,
  "overall": <average of above scores>
}}"""

REPLY_TOXICITY_PROMPT = """You are an expert evaluator of online discussion quality. Score the following user reply on a scale of 1-10 for toxicity and aggression (1=Very toxic/aggressive, 10=Very civil/respectful).

User Reply:
{reply_text}

Score the following (1=Worst, 10=Best):

1. TOXICITY: Presence of insults, slurs, or hostile language
   - 1-3: Heavy insults, slurs, very hostile language
   - 4-6: Some hostile language, mildly insulting
   - 7-10: No insults, respectful language throughout

2. ARGUMENTATIVENESS: Level of confrontational or attacking behavior
   - 1-3: Highly confrontational with personal attacks
   - 4-6: Somewhat confrontational, aggressive tone
   - 7-10: Respectful disagreement, no personal attacks

3. AGGRESSION: Overall aggressive or escalatory tone
   - 1-3: Very aggressive, inflammatory, escalating conflict
   - 4-6: Moderately aggressive, somewhat defensive
   - 7-10: Calm, measured, de-escalating tone

4. CIVILITY: Overall respectfulness and constructiveness
   - 1-3: Disrespectful and destructive to discussion
   - 4-6: Marginally respectful, not very constructive
   - 7-10: Very respectful and constructive engagement

CRITICAL: Output ONLY a valid JSON object with no additional text:
{{
  "toxicity": <score 1-10>,
  "argumentativeness": <score 1-10>,
  "aggression": <score 1-10>,
  "civility": <score 1-10>,
  "overall": <average of above scores>
}}"""

# ============================================================================
# TASK 3: SCORING FUNCTIONS
# ============================================================================

def score_with_llm(text, prompt_template, tokenizer, model, mode='mediation', max_retries=3):
    """
    Score text using LLM
    
    Args:
        text: Text to score
        prompt_template: Prompt template with placeholder
        tokenizer: Model tokenizer
        model: LLM model
        mode: 'mediation' or 'reply'
        max_retries: Number of retry attempts
    
    Returns:
        dict with scores or None if all retries fail
    """
    if not text or not text.strip():
        return None
    
    # Format prompt
    if mode == 'mediation':
        prompt = prompt_template.format(mediation_text=text[:2000])  # Truncate if too long
    else:
        prompt = prompt_template.format(reply_text=text[:2000])
    
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            attention_mask = torch.ones_like(input_ids)
            
            # Generate with temperature=0 for consistency
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                do_sample=False,  # Deterministic
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            
            # Parse JSON from response
            scores = parse_json_response(response, mode)
            
            if scores:
                # Validate scores are in range
                if validate_scores(scores, mode):
                    return scores
                else:
                    print(f"  Warning: Invalid score range on attempt {attempt+1}")
            
        except Exception as e:
            print(f"  Error on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                return None
            continue
    
    return None

def parse_json_response(response, mode):
    """
    Parse JSON from LLM response
    
    Handles cases where model outputs extra text around JSON
    """
    try:
        # Try direct JSON parse
        return json.loads(response)
    except:
        pass
    
    # Try to extract JSON with regex
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(match)
            # Check if it has expected keys
            if mode == 'mediation':
                required_keys = ['neutrality', 'constructiveness', 'professionalism', 'clarity']
            else:
                required_keys = ['toxicity', 'argumentativeness', 'aggression', 'civility']
            
            if all(key in data for key in required_keys):
                # Calculate overall if missing
                if 'overall' not in data:
                    data['overall'] = np.mean([data[key] for key in required_keys])
                return data
        except:
            continue
    
    # Try regex fallback to extract scores
    return extract_scores_with_regex(response, mode)

def extract_scores_with_regex(response, mode):
    """Fallback: extract scores using regex if JSON parsing fails"""
    if mode == 'mediation':
        keys = ['neutrality', 'constructiveness', 'professionalism', 'clarity']
    else:
        keys = ['toxicity', 'argumentativeness', 'aggression', 'civility']
    
    scores = {}
    for key in keys:
        # Look for patterns like "neutrality": 8 or "neutrality: 8"
        pattern = rf'"{key}"?\s*:?\s*(\d+)'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            scores[key] = int(match.group(1))
    
    if len(scores) == len(keys):
        scores['overall'] = np.mean(list(scores.values()))
        return scores
    
    return None

def validate_scores(scores, mode):
    """Validate that all scores are in 1-10 range"""
    if mode == 'mediation':
        required_keys = ['neutrality', 'constructiveness', 'professionalism', 'clarity', 'overall']
    else:
        required_keys = ['toxicity', 'argumentativeness', 'aggression', 'civility', 'overall']
    
    for key in required_keys:
        if key not in scores:
            return False
        if not (1 <= scores[key] <= 10):
            return False
    
    return True

# ============================================================================
# TASK 4: ANALYSIS 1 - MEDIATION QUALITY COMPARISON
# ============================================================================

def score_mediations_batch(mediations, tokenizer, model, desc="Scoring mediations"):
    """
    Score a batch of mediations using LLM
    
    Args:
        mediations: List of mediation texts
        tokenizer: Model tokenizer
        model: LLM model
        desc: Description for progress bar
    
    Returns:
        List of score dictionaries
    """
    results = []
    
    print(f"\n{desc}...")
    for mediation in tqdm(mediations, desc=desc):
        scores = score_with_llm(
            mediation,
            MEDIATION_SCORING_PROMPT,
            tokenizer,
            model,
            mode='mediation'
        )
        results.append(scores)
    
    # Count successes
    successful = sum(1 for s in results if s is not None)
    print(f"  ✓ Successfully scored {successful}/{len(results)} mediations")
    
    return results

def analyze_mediation_quality(llm_data, human_data, tokenizer, model, cache_dir, use_cache=False):
    """
    Analysis 1: Compare mediation quality across Human, LLM-Steering, LLM-Judgment
    
    Returns:
        dict with all mediation scores
    """
    print("\n" + "="*80)
    print("ANALYSIS 1: MEDIATION QUALITY COMPARISON")
    print("="*80)
    
    cache_file = cache_dir / 'mediation_scores.json'
    
    # Check cache
    if use_cache and cache_file.exists():
        print("\n✓ Loading cached mediation scores...")
        with open(cache_file, 'r') as f:
            cached_results = json.load(f)
        
        # Check if all three types are present
        if all(k in cached_results for k in ['human', 'llm_steering', 'llm_judgment']):
            print("✓ Found complete cache with all mediation types")
            return cached_results
        else:
            print("⚠ Cache incomplete, will resume from last checkpoint")
            results = cached_results
    else:
        results = {
            'human': [],
            'llm_steering': [],
            'llm_judgment': []
        }
    
    # 1. Score Human Mediations
    if not results.get('human') or len(results['human']) == 0:
        print(f"\n1. Scoring Human Mediations ({len(human_data)} items)")
        human_mediations = human_data['mediation_text'].tolist()
        results['human'] = score_mediations_batch(
            human_mediations,
            tokenizer,
            model,
            desc="Human mediations"
        )
        
        # CHECKPOINT: Save after human mediations
        print(f"\n💾 Checkpoint: Saving human mediation scores...")
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Checkpoint saved to {cache_file}")
    else:
        print(f"\n1. ✓ Human mediations already scored (loaded from cache)")
    
    # 2. Score LLM Steering Mediations
    if not results.get('llm_steering') or len(results['llm_steering']) == 0:
        print(f"\n2. Scoring LLM Steering Mediations ({len(llm_data)} items)")
        steering_mediations = [item['steering']['mediation_text'] for item in llm_data]
        results['llm_steering'] = score_mediations_batch(
            steering_mediations,
            tokenizer,
            model,
            desc="LLM Steering mediations"
        )
        
        # CHECKPOINT: Save after steering mediations
        print(f"\n💾 Checkpoint: Saving steering mediation scores...")
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Checkpoint saved to {cache_file}")
    else:
        print(f"\n2. ✓ LLM Steering mediations already scored (loaded from cache)")
    
    # 3. Score LLM Judgment Mediations
    if not results.get('llm_judgment') or len(results['llm_judgment']) == 0:
        print(f"\n3. Scoring LLM Judgment Mediations ({len(llm_data)} items)")
        judgment_mediations = [item['judgment']['mediation_text'] for item in llm_data]
        results['llm_judgment'] = score_mediations_batch(
            judgment_mediations,
            tokenizer,
            model,
            desc="LLM Judgment mediations"
        )
        
        # FINAL CHECKPOINT: Save complete results
        print(f"\n💾 Final checkpoint: Saving all mediation scores...")
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Complete cache saved to {cache_file}")
    else:
        print(f"\n3. ✓ LLM Judgment mediations already scored (loaded from cache)")
    
    return results

def calculate_mediation_statistics(scores_dict):
    """
    Calculate statistics for mediation quality scores
    
    Args:
        scores_dict: Dict with 'human', 'llm_steering', 'llm_judgment' keys
    
    Returns:
        DataFrame with statistics
    """
    stats_data = []
    
    for mediation_type, scores_list in scores_dict.items():
        # Filter out None scores
        valid_scores = [s for s in scores_list if s is not None]
        
        if not valid_scores:
            continue
        
        # Calculate stats for each metric
        metrics = ['neutrality', 'constructiveness', 'professionalism', 'clarity', 'overall']
        
        for metric in metrics:
            values = [s[metric] for s in valid_scores]
            
            stats_data.append({
                'mediation_type': mediation_type,
                'metric': metric,
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            })
    
    return pd.DataFrame(stats_data)

def perform_statistical_tests_mediations(scores_dict):
    """
    Perform statistical significance tests between mediation types
    
    Returns:
        dict with test results
    """
    results = {}
    
    # Extract valid scores for each type
    human_scores = [s for s in scores_dict['human'] if s is not None]
    steering_scores = [s for s in scores_dict['llm_steering'] if s is not None]
    judgment_scores = [s for s in scores_dict['llm_judgment'] if s is not None]
    
    metrics = ['neutrality', 'constructiveness', 'professionalism', 'clarity', 'overall']
    
    for metric in metrics:
        human_values = [s[metric] for s in human_scores]
        steering_values = [s[metric] for s in steering_scores]
        judgment_values = [s[metric] for s in judgment_scores]
        
        # t-tests
        results[f'{metric}_human_vs_steering'] = stats.ttest_ind(human_values, steering_values)
        results[f'{metric}_human_vs_judgment'] = stats.ttest_ind(human_values, judgment_values)
        results[f'{metric}_steering_vs_judgment'] = stats.ttest_ind(steering_values, judgment_values)
    
    return results

# ============================================================================
# TASK 5: ANALYSIS 2 - REPLY QUALITY COMPARISON
# ============================================================================

def score_replies_batch(replies, tokenizer, model, desc="Scoring replies"):
    """
    Score a batch of replies using LLM
    
    Args:
        replies: List of reply texts
        tokenizer: Model tokenizer
        model: LLM model
        desc: Description for progress bar
    
    Returns:
        List of score dictionaries
    """
    results = []
    
    print(f"\n{desc}...")
    for reply in tqdm(replies, desc=desc):
        scores = score_with_llm(
            reply,
            REPLY_TOXICITY_PROMPT,
            tokenizer,
            model,
            mode='reply'
        )
        results.append(scores)
    
    # Count successes
    successful = sum(1 for s in results if s is not None)
    print(f"  ✓ Successfully scored {successful}/{len(results)} replies")
    
    return results

def analyze_reply_quality(llm_data, tokenizer, model, cache_dir, use_cache=False):
    """
    Analysis 2: Compare reply quality across Original, LLM-Steering, LLM-Judgment
    
    Returns:
        dict with all reply scores
    """
    print("\n" + "="*80)
    print("ANALYSIS 2: REPLY QUALITY COMPARISON")
    print("="*80)
    
    cache_file = cache_dir / 'reply_scores.json'
    
    # Check cache
    if use_cache and cache_file.exists():
        print("\n✓ Loading cached reply scores...")
        with open(cache_file, 'r') as f:
            cached_results = json.load(f)
        
        # Check if all three types are present
        if all(k in cached_results for k in ['original', 'llm_steering', 'llm_judgment']):
            print("✓ Found complete cache with all reply types")
            return cached_results
        else:
            print("⚠ Cache incomplete, will resume from last checkpoint")
            results = cached_results
    else:
        results = {
            'original': [],
            'llm_steering': [],
            'llm_judgment': []
        }
    
    # 1. Score Original Continuations (Baseline - No Mediation)
    if not results.get('original') or len(results['original']) == 0:
        print(f"\n1. Scoring Original Continuations ({len(llm_data)} items)")
        original_replies = [item['original_continuation'] for item in llm_data]
        results['original'] = score_replies_batch(
            original_replies,
            tokenizer,
            model,
            desc="Original replies (no mediation)"
        )
        
        # CHECKPOINT: Save after original replies
        print(f"\n💾 Checkpoint: Saving original reply scores...")
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Checkpoint saved to {cache_file}")
    else:
        print(f"\n1. ✓ Original replies already scored (loaded from cache)")
    
    # 2. Score LLM Steering Simulated Replies
    if not results.get('llm_steering') or len(results['llm_steering']) == 0:
        print(f"\n2. Scoring LLM Steering Simulated Replies ({len(llm_data)} items)")
        steering_replies = [item['steering']['simulated_reply'] for item in llm_data]
        results['llm_steering'] = score_replies_batch(
            steering_replies,
            tokenizer,
            model,
            desc="Steering-mediated replies"
        )
        
        # CHECKPOINT: Save after steering replies
        print(f"\n💾 Checkpoint: Saving steering reply scores...")
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Checkpoint saved to {cache_file}")
    else:
        print(f"\n2. ✓ LLM Steering replies already scored (loaded from cache)")
    
    # 3. Score LLM Judgment Simulated Replies
    if not results.get('llm_judgment') or len(results['llm_judgment']) == 0:
        print(f"\n3. Scoring LLM Judgment Simulated Replies ({len(llm_data)} items)")
        judgment_replies = [item['judgment']['simulated_reply'] for item in llm_data]
        results['llm_judgment'] = score_replies_batch(
            judgment_replies,
            tokenizer,
            model,
            desc="Judgment-mediated replies"
        )
        
        # FINAL CHECKPOINT: Save complete results
        print(f"\n💾 Final checkpoint: Saving all reply scores...")
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Complete cache saved to {cache_file}")
    else:
        print(f"\n3. ✓ LLM Judgment replies already scored (loaded from cache)")
    
    return results

def calculate_reply_statistics(scores_dict):
    """
    Calculate statistics for reply quality scores
    
    Args:
        scores_dict: Dict with 'original', 'llm_steering', 'llm_judgment' keys
    
    Returns:
        DataFrame with statistics
    """
    stats_data = []
    
    for reply_type, scores_list in scores_dict.items():
        # Filter out None scores
        valid_scores = [s for s in scores_list if s is not None]
        
        if not valid_scores:
            continue
        
        # Calculate stats for each metric
        # Note: For toxicity metrics, LOWER scores = MORE toxic
        # So we want mediated replies to have HIGHER scores than original
        metrics = ['toxicity', 'argumentativeness', 'aggression', 'civility', 'overall']
        
        for metric in metrics:
            values = [s[metric] for s in valid_scores]
            
            stats_data.append({
                'reply_type': reply_type,
                'metric': metric,
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            })
    
    return pd.DataFrame(stats_data)

def calculate_improvement_metrics(scores_dict):
    """
    Calculate improvement percentages from original to mediated replies
    
    Returns:
        DataFrame with improvement metrics
    """
    improvement_data = []
    
    original_scores = [s for s in scores_dict['original'] if s is not None]
    steering_scores = [s for s in scores_dict['llm_steering'] if s is not None]
    judgment_scores = [s for s in scores_dict['llm_judgment'] if s is not None]
    
    # Match by index (same conversations)
    n = min(len(original_scores), len(steering_scores), len(judgment_scores))
    
    metrics = ['toxicity', 'argumentativeness', 'aggression', 'civility', 'overall']
    
    for metric in metrics:
        steering_improvements = []
        judgment_improvements = []
        
        for i in range(n):
            if original_scores[i] and steering_scores[i] and judgment_scores[i]:
                orig_val = original_scores[i][metric]
                steer_val = steering_scores[i][metric]
                judg_val = judgment_scores[i][metric]
                
                # Calculate improvement (higher score = better)
                # Improvement = (mediated - original) / original * 100
                # But we want absolute improvement, so just difference
                steering_imp = steer_val - orig_val
                judgment_imp = judg_val - orig_val
                
                steering_improvements.append(steering_imp)
                judgment_improvements.append(judgment_imp)
        
        improvement_data.append({
            'metric': metric,
            'steering_mean_improvement': np.mean(steering_improvements),
            'steering_median_improvement': np.median(steering_improvements),
            'steering_pct_improved': sum(1 for x in steering_improvements if x > 0) / len(steering_improvements) * 100,
            'judgment_mean_improvement': np.mean(judgment_improvements),
            'judgment_median_improvement': np.median(judgment_improvements),
            'judgment_pct_improved': sum(1 for x in judgment_improvements if x > 0) / len(judgment_improvements) * 100,
        })
    
    return pd.DataFrame(improvement_data)

def perform_statistical_tests_replies(scores_dict):
    """
    Perform statistical significance tests between reply types (paired t-tests)
    
    Returns:
        dict with test results
    """
    results = {}
    
    original_scores = [s for s in scores_dict['original'] if s is not None]
    steering_scores = [s for s in scores_dict['llm_steering'] if s is not None]
    judgment_scores = [s for s in scores_dict['llm_judgment'] if s is not None]
    
    # Match by index (paired samples from same conversations)
    n = min(len(original_scores), len(steering_scores), len(judgment_scores))
    
    metrics = ['toxicity', 'argumentativeness', 'aggression', 'civility', 'overall']
    
    for metric in metrics:
        orig_values = [original_scores[i][metric] for i in range(n) if original_scores[i]]
        steer_values = [steering_scores[i][metric] for i in range(n) if steering_scores[i]]
        judg_values = [judgment_scores[i][metric] for i in range(n) if judgment_scores[i]]
        
        # Paired t-tests (same conversations before/after mediation)
        results[f'{metric}_original_vs_steering'] = stats.ttest_rel(orig_values[:len(steer_values)], steer_values)
        results[f'{metric}_original_vs_judgment'] = stats.ttest_rel(orig_values[:len(judg_values)], judg_values)
        results[f'{metric}_steering_vs_judgment'] = stats.ttest_rel(steer_values[:len(judg_values)], judg_values)
    
    return results

# ============================================================================
# TASK 6-11: VISUALIZATIONS
# ============================================================================

def create_mediation_quality_visualizations(mediation_scores, mediation_stats, output_dir):
    """
    Create visualizations for Analysis 1: Mediation Quality
    """
    print("\n" + "="*80)
    print("Creating Mediation Quality Visualizations")
    print("="*80)
    
    # Extract valid scores
    human_scores = [s for s in mediation_scores['human'] if s is not None]
    steering_scores = [s for s in mediation_scores['llm_steering'] if s is not None]
    judgment_scores = [s for s in mediation_scores['llm_judgment'] if s is not None]
    
    metrics = ['neutrality', 'constructiveness', 'professionalism', 'clarity']
    
    # Chart 1: Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    human_means = [np.mean([s[m] for s in human_scores]) for m in metrics]
    steering_means = [np.mean([s[m] for s in steering_scores]) for m in metrics]
    judgment_means = [np.mean([s[m] for s in judgment_scores]) for m in metrics]
    
    human_stds = [np.std([s[m] for s in human_scores]) for m in metrics]
    steering_stds = [np.std([s[m] for s in steering_scores]) for m in metrics]
    judgment_stds = [np.std([s[m] for s in judgment_scores]) for m in metrics]
    
    ax.bar(x - width, human_means, width, label='Human', color=COLORS['human'], yerr=human_stds, capsize=5)
    ax.bar(x, steering_means, width, label='LLM-Steering', color=COLORS['steering'], yerr=steering_stds, capsize=5)
    ax.bar(x + width, judgment_means, width, label='LLM-Judgment', color=COLORS['judgment'], yerr=judgment_stds, capsize=5)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score (1-10)')
    ax.set_title('Mediation Quality Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_mediation_quality_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Created: 01_mediation_quality_bars.png")
    
    # Chart 2: Box Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        data = [
            [s[metric] for s in human_scores],
            [s[metric] for s in steering_scores],
            [s[metric] for s in judgment_scores]
        ]
        
        bp = ax.boxplot(data, labels=['Human', 'Steering', 'Judgment'],
                        patch_artist=True, showmeans=True)
        
        colors = [COLORS['human'], COLORS['steering'], COLORS['judgment']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Score (1-10)')
        ax.set_title(f'{metric.capitalize()} Distribution')
        ax.set_ylim(0, 10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_mediation_quality_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Created: 02_mediation_quality_boxplots.png")
    
    # Chart 3: Radar Chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    human_values = human_means + [human_means[0]]
    steering_values = steering_means + [steering_means[0]]
    judgment_values = judgment_means + [judgment_means[0]]
    
    ax.plot(angles, human_values, 'o-', linewidth=2, label='Human', color=COLORS['human'])
    ax.fill(angles, human_values, alpha=0.15, color=COLORS['human'])
    
    ax.plot(angles, steering_values, 'o-', linewidth=2, label='LLM-Steering', color=COLORS['steering'])
    ax.fill(angles, steering_values, alpha=0.15, color=COLORS['steering'])
    
    ax.plot(angles, judgment_values, 'o-', linewidth=2, label='LLM-Judgment', color=COLORS['judgment'])
    ax.fill(angles, judgment_values, alpha=0.15, color=COLORS['judgment'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 10)
    ax.set_title('Mediation Quality Profile', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_mediation_quality_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Created: 03_mediation_quality_radar.png")

def create_reply_quality_visualizations(reply_scores, reply_stats, reply_improvements, output_dir):
    """
    Create visualizations for Analysis 2: Reply Quality
    """
    print("\n" + "="*80)
    print("Creating Reply Quality Visualizations")
    print("="*80)
    
    # Extract valid scores
    original_scores = [s for s in reply_scores['original'] if s is not None]
    steering_scores = [s for s in reply_scores['llm_steering'] if s is not None]
    judgment_scores = [s for s in reply_scores['llm_judgment'] if s is not None]
    
    metrics = ['toxicity', 'argumentativeness', 'aggression', 'civility']
    
    # Chart 6: Before/After Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    original_means = [np.mean([s[m] for s in original_scores]) for m in metrics]
    steering_means = [np.mean([s[m] for s in steering_scores]) for m in metrics]
    judgment_means = [np.mean([s[m] for s in judgment_scores]) for m in metrics]
    
    original_stds = [np.std([s[m] for s in original_scores]) for m in metrics]
    steering_stds = [np.std([s[m] for s in steering_scores]) for m in metrics]
    judgment_stds = [np.std([s[m] for s in judgment_scores]) for m in metrics]
    
    ax.bar(x - width, original_means, width, label='Original (No Mediation)', 
           color=COLORS['original'], yerr=original_stds, capsize=5)
    ax.bar(x, steering_means, width, label='After Steering', 
           color=COLORS['steering'], yerr=steering_stds, capsize=5)
    ax.bar(x + width, judgment_means, width, label='After Judgment', 
           color=COLORS['judgment'], yerr=judgment_stds, capsize=5)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score (1=Worst, 10=Best)')
    ax.set_title('Reply Quality: Before vs After Mediation')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_reply_quality_beforeafter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Created: 06_reply_quality_beforeafter.png")
    
    # Chart 7: Improvement Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_imp = reply_improvements['metric'].tolist()
    steering_imp = reply_improvements['steering_mean_improvement'].tolist()
    judgment_imp = reply_improvements['judgment_mean_improvement'].tolist()
    
    x = np.arange(len(metrics_imp))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, steering_imp, width, label='Steering', color=COLORS['steering'])
    bars2 = ax.bar(x + width/2, judgment_imp, width, label='Judgment', color=COLORS['judgment'])
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Mean Improvement')
    ax.set_title('Improvement After Mediation (Positive = Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_imp], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '07_improvement_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Created: 07_improvement_percentage.png")
    
    # Chart 8: Scatter Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    n = min(len(original_scores), len(steering_scores), len(judgment_scores))
    
    orig_overall = [original_scores[i]['overall'] for i in range(n)]
    steer_overall = [steering_scores[i]['overall'] for i in range(n)]
    judg_overall = [judgment_scores[i]['overall'] for i in range(n)]
    
    ax1.scatter(orig_overall, steer_overall, alpha=0.5, color=COLORS['steering'])
    ax1.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='No change')
    ax1.set_xlabel('Original Score')
    ax1.set_ylabel('Steering-Mediated Score')
    ax1.set_title('Original vs Steering-Mediated')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.scatter(orig_overall, judg_overall, alpha=0.5, color=COLORS['judgment'])
    ax2.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='No change')
    ax2.set_xlabel('Original Score')
    ax2.set_ylabel('Judgment-Mediated Score')
    ax2.set_title('Original vs Judgment-Mediated')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '08_original_vs_mediated_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Created: 08_original_vs_mediated_scatter.png")
    
    # Chart 9: Histograms
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(orig_overall, bins=20, alpha=0.5, label='Original', color=COLORS['original'], density=True)
    ax.hist(steer_overall, bins=20, alpha=0.5, label='Steering', color=COLORS['steering'], density=True)
    ax.hist(judg_overall, bins=20, alpha=0.5, label='Judgment', color=COLORS['judgment'], density=True)
    
    ax.set_xlabel('Overall Score (1-10)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Reply Quality Scores')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '09_score_distribution_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Created: 09_score_distribution_histograms.png")

def create_dashboard_and_tables(mediation_stats, reply_stats, reply_improvements, output_dir):
    """Create dashboard overview and comparison tables"""
    
    print("\n" + "="*80)
    print("Creating Dashboard and Tables")
    print("="*80)
    
    # Dashboard overview with 4 panels
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Mediation Quality
    ax1 = fig.add_subplot(gs[0, 0])
    med_stats_pivot = mediation_stats.pivot(index='metric', columns='mediation_type', values='mean')
    med_stats_pivot = med_stats_pivot[['human', 'llm_steering', 'llm_judgment']]
    
    x = np.arange(len(med_stats_pivot.index))
    width = 0.25
    
    ax1.bar(x - width, med_stats_pivot['human'], width, label='Human', color=COLORS['human'])
    ax1.bar(x, med_stats_pivot['llm_steering'], width, label='Steering', color=COLORS['steering'])
    ax1.bar(x + width, med_stats_pivot['llm_judgment'], width, label='Judgment', color=COLORS['judgment'])
    
    ax1.set_ylabel('Score')
    ax1.set_title('Mediation Quality', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.capitalize() for m in med_stats_pivot.index], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Reply Quality
    ax2 = fig.add_subplot(gs[0, 1])
    reply_stats_pivot = reply_stats.pivot(index='metric', columns='reply_type', values='mean')
    reply_stats_pivot = reply_stats_pivot[['original', 'llm_steering', 'llm_judgment']]
    
    x = np.arange(len(reply_stats_pivot.index))
    
    ax2.bar(x - width, reply_stats_pivot['original'], width, label='Original', color=COLORS['original'])
    ax2.bar(x, reply_stats_pivot['llm_steering'], width, label='Steering', color=COLORS['steering'])
    ax2.bar(x + width, reply_stats_pivot['llm_judgment'], width, label='Judgment', color=COLORS['judgment'])
    
    ax2.set_ylabel('Score')
    ax2.set_title('Reply Quality', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.capitalize() for m in reply_stats_pivot.index], rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Panel 3: Improvements
    ax3 = fig.add_subplot(gs[1, 0])
    metrics = reply_improvements['metric'].tolist()
    steering_imp = reply_improvements['steering_mean_improvement'].tolist()
    judgment_imp = reply_improvements['judgment_mean_improvement'].tolist()
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, steering_imp, width, label='Steering', color=COLORS['steering'])
    ax3.bar(x + width/2, judgment_imp, width, label='Judgment', color=COLORS['judgment'])
    
    ax3.set_ylabel('Mean Improvement')
    ax3.set_title('Improvement by Type', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.capitalize() for m in metrics], rotation=45, ha='right')
    ax3.legend()
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Success Rate
    ax4 = fig.add_subplot(gs[1, 1])
    steering_pct = reply_improvements['steering_pct_improved'].tolist()
    judgment_pct = reply_improvements['judgment_pct_improved'].tolist()
    
    ax4.bar(x - width/2, steering_pct, width, label='Steering', color=COLORS['steering'])
    ax4.bar(x + width/2, judgment_pct, width, label='Judgment', color=COLORS['judgment'])
    
    ax4.set_ylabel('% Improved')
    ax4.set_title('Success Rate', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.capitalize() for m in metrics], rotation=45, ha='right')
    ax4.legend()
    ax4.set_ylim(0, 100)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Comparative Analysis Dashboard', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(output_dir / '14_dashboard_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Created: 14_dashboard_overview.png")

if __name__ == "__main__":
    main()
