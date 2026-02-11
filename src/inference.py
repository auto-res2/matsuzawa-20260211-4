"""
Inference script for Paraphrase-Robust Auto-CoT experiments.
Implements PR-AutoCoT and baseline CW-AutoCoT methods.
"""
import sys
import json
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from src.preprocess import load_gsm8k_data
from src.model import OpenAIModel

# Transform 'run=' arguments to 'runs=' to match directory structure
# This allows command line to use 'run=X' while Hydra looks in 'runs/' directory
sys.argv = [arg.replace('run=', 'runs=', 1) if arg.startswith('run=') else arg for arg in sys.argv]


def extract_numeric_answer(text: str) -> Optional[float]:
    """Extract final numeric answer from model output."""
    # Look for patterns like "####" followed by number (GSM8K format)
    match = re.search(r'####\s*(-?\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    
    # Look for "The answer is" pattern
    match = re.search(r'(?:answer is|answer:|equals?)\s*\$?\s*(-?\d+\.?\d*)', text.lower())
    if match:
        return float(match.group(1))
    
    # Look for last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return float(numbers[-1])
    
    return None


def cluster_questions(questions: List[str], k: int = 8, method: str = "tfidf") -> List[int]:
    """
    Cluster questions using TF-IDF + k-means and return representative indices.
    
    Args:
        questions: List of question strings
        k: Number of clusters
        method: Clustering method ("tfidf" or "sbert")
    
    Returns:
        List of k representative question indices
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        embeddings = vectorizer.fit_transform(questions).toarray()
    else:
        # Fallback to TF-IDF if SBERT not available
        print(f"[cluster] Using TF-IDF (sbert not implemented)")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        embeddings = vectorizer.fit_transform(questions).toarray()
    
    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Select representative from each cluster (closest to centroid)
    representatives = []
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_embeddings = embeddings[cluster_indices]
        centroid = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        representatives.append(int(closest_idx))
    
    return representatives


def generate_paraphrases(question: str, model: OpenAIModel, num_paraphrases: int = 3) -> List[str]:
    """
    Generate meaning-preserving paraphrases of a question.
    
    Args:
        question: Original question
        model: Language model
        num_paraphrases: Number of paraphrases to generate
    
    Returns:
        List of paraphrased questions
    """
    prompt = f"""Paraphrase the following math word problem. Keep all quantities, numbers, and constraints exactly the same. Only change the wording.

Original: {question}

Generate {num_paraphrases} paraphrases, one per line:"""
    
    response = model.generate(prompt, temperature=0.7, max_tokens=512)
    
    # Parse paraphrases from response
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    paraphrases = []
    for line in lines[:num_paraphrases]:
        # Remove numbering like "1.", "2.", etc.
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
        if cleaned and len(cleaned) > 10:
            paraphrases.append(cleaned)
    
    # If we didn't get enough, repeat the original with slight modifications
    while len(paraphrases) < num_paraphrases:
        paraphrases.append(question)
    
    return paraphrases[:num_paraphrases]


def solve_question(question: str, model: OpenAIModel, temperature: float = 0.0) -> Tuple[str, Optional[float]]:
    """
    Solve a math word problem with chain-of-thought reasoning.
    
    Args:
        question: Math word problem
        model: Language model
        temperature: Sampling temperature
    
    Returns:
        (full_solution, numeric_answer)
    """
    prompt = f"""Solve this math word problem step by step. Show your reasoning and provide the final numeric answer.

Question: {question}

Solution:"""
    
    solution = model.generate(prompt, temperature=temperature, max_tokens=512)
    answer = extract_numeric_answer(solution)
    
    return solution, answer


def compute_self_consistency(
    question: str, 
    model: OpenAIModel, 
    num_samples: int = 5, 
    temperature: float = 0.7
) -> Tuple[float, Optional[float]]:
    """
    Compute self-consistency score by sampling multiple solutions.
    
    Args:
        question: Question to solve
        model: Language model
        num_samples: Number of stochastic samples
        temperature: Sampling temperature
    
    Returns:
        (self_consistency_score, majority_answer)
    """
    answers = []
    for _ in range(num_samples):
        _, answer = solve_question(question, model, temperature=temperature)
        if answer is not None:
            answers.append(answer)
    
    if not answers:
        return 0.0, None
    
    # Find majority answer
    answer_counts = Counter(answers)
    majority_answer, majority_count = answer_counts.most_common(1)[0]
    
    # Self-consistency score = majority fraction
    score = majority_count / len(answers)
    
    return score, majority_answer


def compute_paraphrase_consistency(
    question: str,
    model: OpenAIModel,
    num_paraphrases: int = 3,
    temperature: float = 0.3,
    majority_answer: Optional[float] = None
) -> float:
    """
    Compute paraphrase-consistency score.
    
    Args:
        question: Original question
        model: Language model
        num_paraphrases: Number of paraphrases to generate
        temperature: Solving temperature
        majority_answer: Majority answer from self-consistency (for comparison)
    
    Returns:
        Paraphrase consistency score
    """
    # Generate paraphrases
    paraphrases = generate_paraphrases(question, model, num_paraphrases)
    
    # Solve each paraphrase
    paraphrase_answers = []
    for para in paraphrases:
        _, answer = solve_question(para, model, temperature=temperature)
        if answer is not None:
            paraphrase_answers.append(answer)
    
    if not paraphrase_answers:
        return 0.0
    
    # If majority_answer is provided, compute fraction matching it
    if majority_answer is not None:
        matches = sum(1 for ans in paraphrase_answers if abs(ans - majority_answer) < 0.01)
        score = matches / len(paraphrases)
    else:
        # Otherwise, compute internal consistency
        answer_counts = Counter(paraphrase_answers)
        if answer_counts:
            majority_count = answer_counts.most_common(1)[0][1]
            score = majority_count / len(paraphrase_answers)
        else:
            score = 0.0
    
    return score


def build_demonstrations(
    train_data: List[Dict[str, Any]],
    model: OpenAIModel,
    cfg: DictConfig
) -> List[Dict[str, Any]]:
    """
    Build chain-of-thought demonstrations using PR-AutoCoT or CW-AutoCoT.
    
    Args:
        train_data: Training questions (used unlabeled)
        model: Language model
        cfg: Configuration
    
    Returns:
        List of demonstrations with reliability scores
    """
    method_name = cfg.method.name
    k = cfg.method.clustering.k
    
    print(f"[build_demos] Starting {method_name} with k={k} demonstrations")
    
    # Step 1: Diverse candidate selection
    questions = [item['question'] for item in train_data]
    representative_indices = cluster_questions(
        questions, 
        k=k, 
        method=cfg.method.clustering.vectorizer
    )
    
    print(f"[build_demos] Selected {len(representative_indices)} representative questions")
    
    # Step 2: Generate candidate demonstrations and score them
    demonstrations = []
    
    for idx, rep_idx in enumerate(representative_indices):
        question = train_data[rep_idx]['question']
        print(f"[build_demos] Processing demo {idx+1}/{len(representative_indices)}: {question[:60]}...")
        
        # Generate one candidate demo (greedy)
        solution, answer = solve_question(question, model, temperature=0.0)
        
        # Compute self-consistency score
        sc_score, majority_answer = compute_self_consistency(
            question,
            model,
            num_samples=cfg.method.self_consistency.num_samples,
            temperature=cfg.method.self_consistency.temperature
        )
        
        print(f"  Self-consistency: {sc_score:.3f}, majority answer: {majority_answer}")
        
        # Compute paraphrase-consistency score (if enabled)
        if method_name == "pr-autocot" and cfg.method.paraphrase_consistency.get('enabled', True):
            pc_score = compute_paraphrase_consistency(
                question,
                model,
                num_paraphrases=cfg.method.paraphrase_consistency.num_paraphrases,
                temperature=cfg.method.paraphrase_consistency.temperature,
                majority_answer=majority_answer
            )
            print(f"  Paraphrase-consistency: {pc_score:.3f}")
        else:
            pc_score = 1.0  # Not used in baseline
        
        # Compute reliability weight
        combining = cfg.method.reliability.combining
        if combining == "sqrt_product":
            reliability = math.sqrt(sc_score * pc_score)
        elif combining == "min":
            reliability = min(sc_score, pc_score)
        elif combining == "self_only":
            reliability = sc_score
        else:
            reliability = sc_score
        
        print(f"  Reliability weight: {reliability:.3f}")
        
        demonstrations.append({
            'question': question,
            'solution': solution,
            'answer': answer,
            'self_consistency': sc_score,
            'paraphrase_consistency': pc_score,
            'reliability': reliability
        })
    
    # Step 3: Filter by threshold and sort by reliability
    threshold = cfg.method.reliability.threshold
    demonstrations = [d for d in demonstrations if d['reliability'] >= threshold]
    demonstrations.sort(key=lambda d: d['reliability'], reverse=True)
    
    print(f"[build_demos] Kept {len(demonstrations)}/{len(representative_indices)} demos (threshold={threshold})")
    
    return demonstrations


def create_prompt_with_demos(demos: List[Dict[str, Any]], test_question: str) -> str:
    """Create prompt with demonstrations for test question."""
    prompt_parts = []
    
    # Add demonstrations
    for i, demo in enumerate(demos):
        prompt_parts.append(f"Example {i+1}:")
        prompt_parts.append(f"Question: {demo['question']}")
        prompt_parts.append(f"Solution: {demo['solution']}")
        prompt_parts.append("")
    
    # Add test question
    prompt_parts.append("Now solve this question:")
    prompt_parts.append(f"Question: {test_question}")
    prompt_parts.append("Solution:")
    
    return "\n".join(prompt_parts)


def run_inference(cfg: DictConfig) -> None:
    """Main inference pipeline."""
    
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow"
        )
        print(f"[inference] WandB initialized: {wandb.run.get_url()}")
    else:
        print(f"[inference] WandB disabled (mode={cfg.wandb.mode})")
    
    # Load data
    train_data, test_data = load_gsm8k_data(cfg)
    
    # Limit test data in sanity_check mode
    if cfg.mode == "sanity_check":
        test_data = test_data[:cfg.inference.sanity_check_samples]
        print(f"[inference] Sanity check mode: limited to {len(test_data)} test samples")
    
    # Initialize model
    model = OpenAIModel(cfg.model)
    
    # Build demonstrations
    demonstrations = build_demonstrations(train_data, model, cfg)
    
    print(f"[inference] Starting test inference on {len(test_data)} questions")
    
    # Run inference on test set
    correct = 0
    total = 0
    results = []
    
    for i, item in enumerate(test_data):
        question = item['question']
        true_answer = item['answer']
        
        # Create prompt with demonstrations
        prompt = create_prompt_with_demos(demonstrations, question)
        
        # Generate solution
        solution = model.generate(
            prompt,
            temperature=cfg.method.inference.temperature,
            max_tokens=512
        )
        predicted_answer = extract_numeric_answer(solution)
        
        # Check correctness
        is_correct = False
        if predicted_answer is not None and true_answer is not None:
            is_correct = abs(predicted_answer - true_answer) < 0.01
        
        if is_correct:
            correct += 1
        total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        if (i + 1) % 10 == 0 or i < 5:
            print(f"[inference] Sample {i+1}/{len(test_data)}: acc={accuracy:.3f} ({correct}/{total})")
        
        results.append({
            'question': question,
            'true_answer': true_answer,
            'predicted_answer': predicted_answer,
            'correct': is_correct
        })
        
        # Log to WandB
        if cfg.wandb.mode != "disabled":
            wandb.log({
                'step': i,
                'accuracy': accuracy,
                'correct': int(is_correct)
            })
    
    # Final metrics
    final_accuracy = correct / total if total > 0 else 0.0
    print(f"[inference] Final accuracy: {final_accuracy:.4f} ({correct}/{total})")
    
    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(results_dir / 'demonstrations.json', 'w') as f:
        json.dump(demonstrations, f, indent=2, default=str)
    
    # Log to WandB summary
    if cfg.wandb.mode != "disabled":
        wandb.summary['accuracy'] = final_accuracy
        wandb.summary['correct'] = correct
        wandb.summary['total'] = total
        wandb.summary['num_demonstrations'] = len(demonstrations)
    
    # Sanity validation
    if cfg.mode == "sanity_check":
        perform_sanity_validation(results, demonstrations)
    
    print(f"[inference] Results saved to {results_dir}")


def perform_sanity_validation(results: List[Dict[str, Any]], demonstrations: List[Dict[str, Any]]) -> None:
    """Perform sanity validation checks and emit verdict."""
    
    samples_processed = len(results)
    outputs_valid = sum(1 for r in results if r['predicted_answer'] is not None)
    correct_count = sum(1 for r in results if r['correct'])
    unique_predictions = len(set(r['predicted_answer'] for r in results if r['predicted_answer'] is not None))
    
    # Validation checks
    checks_passed = []
    checks_failed = []
    
    # Check 1: At least 5 samples processed
    if samples_processed >= 5:
        checks_passed.append("samples_processed")
    else:
        checks_failed.append(f"samples_processed={samples_processed}<5")
    
    # Check 2: All outputs are valid (not None)
    if outputs_valid == samples_processed:
        checks_passed.append("outputs_valid")
    else:
        checks_failed.append(f"outputs_valid={outputs_valid}/{samples_processed}")
    
    # Check 3: Outputs are not all identical
    if unique_predictions > 1:
        checks_passed.append("outputs_unique")
    else:
        checks_failed.append(f"unique_predictions={unique_predictions}")
    
    # Check 4: At least one correct prediction
    if correct_count > 0:
        checks_passed.append("at_least_one_correct")
    else:
        checks_failed.append(f"correct_count=0")
    
    # Check 5: Demonstrations were built
    if len(demonstrations) > 0:
        checks_passed.append("demonstrations_built")
    else:
        checks_failed.append("no_demonstrations")
    
    # Summary
    summary = {
        "samples": samples_processed,
        "outputs_valid": outputs_valid,
        "outputs_unique": unique_predictions,
        "correct": correct_count,
        "demonstrations": len(demonstrations)
    }
    
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
    
    # Verdict
    if checks_failed:
        reason = ";".join(checks_failed)
        print(f"SANITY_VALIDATION: FAIL reason={reason}")
    else:
        print("SANITY_VALIDATION: PASS")


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Entry point."""
    print(f"[inference] Starting {cfg.method.name} for run {cfg.run.run_id}")
    print(f"[inference] Mode: {cfg.mode}")
    print(f"[inference] Dataset: {cfg.dataset.name}")
    
    run_inference(cfg)
    
    print(f"[inference] Completed successfully")


if __name__ == "__main__":
    main()
