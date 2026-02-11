"""
Evaluation script for comparing PR-AutoCoT and baseline methods.
Fetches results from WandB and generates comparison metrics and figures.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import wandb


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Results directory path'
    )
    parser.add_argument(
        '--run_ids',
        type=str,
        required=True,
        help='JSON string list of run IDs to evaluate, e.g., \'["run-1", "run-2"]\''
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None,
        help='WandB entity (default: from env WANDB_ENTITY)'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='WandB project (default: from env WANDB_PROJECT)'
    )
    return parser.parse_args()


def fetch_wandb_run_data(entity: str, project: str, run_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch run data from WandB API.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run ID
    
    Returns:
        Dict with config, summary, and history
    """
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Fetch history (time series data)
        history = run.history()
        history_data = history.to_dict('records') if not history.empty else []
        
        return {
            'config': dict(run.config),
            'summary': dict(run.summary),
            'history': history_data,
            'name': run.name,
            'id': run.id
        }
    except Exception as e:
        print(f"[evaluate] Warning: Could not fetch WandB data for {run_id}: {e}")
        return None


def load_local_results(results_dir: Path, run_id: str) -> Optional[Dict[str, Any]]:
    """
    Load results from local files as fallback.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
    
    Returns:
        Dict with results and metrics
    """
    run_dir = results_dir / run_id
    
    if not run_dir.exists():
        return None
    
    results_file = run_dir / 'results.json'
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Compute metrics from results
    correct = sum(1 for r in results if r.get('correct', False))
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'summary': {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        },
        'results': results
    }


def create_accuracy_plot(run_data: Dict[str, Dict[str, Any]], results_dir: Path, run_id: str) -> str:
    """
    Create accuracy over time plot for a single run.
    
    Args:
        run_data: Run data with history
        results_dir: Results directory
        run_id: Run ID
    
    Returns:
        Path to saved figure
    """
    history = run_data.get('history', [])
    
    if not history:
        print(f"[evaluate] No history data for {run_id}, skipping plot")
        return None
    
    # Extract accuracy over steps
    steps = [h.get('step', i) for i, h in enumerate(history)]
    accuracies = [h.get('accuracy', 0) for h in history]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, accuracies, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Test Sample', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Accuracy Over Time: {run_id}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_dir = results_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'accuracy_over_time.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[evaluate] Saved accuracy plot: {output_path}")
    return str(output_path)


def create_comparison_plot(all_run_data: Dict[str, Dict[str, Any]], results_dir: Path) -> str:
    """
    Create comparison bar plot of final accuracies.
    
    Args:
        all_run_data: Dict mapping run_id to run data
        results_dir: Results directory
    
    Returns:
        Path to saved figure
    """
    run_ids = list(all_run_data.keys())
    accuracies = [all_run_data[rid]['summary'].get('accuracy', 0) for rid in run_ids]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71' if 'proposed' in rid else '#3498db' for rid in run_ids]
    bars = plt.bar(run_ids, accuracies, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Run ID', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Method Comparison: Final Accuracy', fontsize=14)
    plt.ylim(0, min(1.0, max(accuracies) * 1.2))
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_dir = results_dir / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'accuracy_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[evaluate] Saved comparison plot: {output_path}")
    return str(output_path)


def compute_aggregated_metrics(all_run_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregated metrics across all runs.
    
    Args:
        all_run_data: Dict mapping run_id to run data
    
    Returns:
        Aggregated metrics dict
    """
    metrics_by_run = {}
    
    for run_id, data in all_run_data.items():
        summary = data.get('summary', {})
        metrics_by_run[run_id] = {
            'accuracy': summary.get('accuracy', 0.0),
            'correct': summary.get('correct', 0),
            'total': summary.get('total', 0),
            'num_demonstrations': summary.get('num_demonstrations', 0)
        }
    
    # Identify proposed and baseline runs
    proposed_runs = [rid for rid in all_run_data.keys() if 'proposed' in rid]
    baseline_runs = [rid for rid in all_run_data.keys() if 'comparative' in rid]
    
    # Compute best proposed and best baseline
    best_proposed = None
    best_proposed_acc = 0.0
    if proposed_runs:
        for rid in proposed_runs:
            acc = metrics_by_run[rid]['accuracy']
            if acc > best_proposed_acc:
                best_proposed_acc = acc
                best_proposed = rid
    
    best_baseline = None
    best_baseline_acc = 0.0
    if baseline_runs:
        for rid in baseline_runs:
            acc = metrics_by_run[rid]['accuracy']
            if acc > best_baseline_acc:
                best_baseline_acc = acc
                best_baseline = rid
    
    # Compute gap
    gap = best_proposed_acc - best_baseline_acc if best_proposed and best_baseline else None
    
    aggregated = {
        'primary_metric': 'accuracy',
        'metrics': metrics_by_run,
        'best_proposed': best_proposed,
        'best_proposed_accuracy': best_proposed_acc,
        'best_baseline': best_baseline,
        'best_baseline_accuracy': best_baseline_acc,
        'gap': gap
    }
    
    return aggregated


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    # Parse run_ids from JSON string
    run_ids = json.loads(args.run_ids)
    print(f"[evaluate] Evaluating runs: {run_ids}")
    
    # Get WandB credentials
    wandb_entity = args.wandb_entity or os.environ.get('WANDB_ENTITY', 'airas')
    wandb_project = args.wandb_project or os.environ.get('WANDB_PROJECT', '2026-02-11')
    
    print(f"[evaluate] WandB: {wandb_entity}/{wandb_project}")
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data for each run
    all_run_data = {}
    generated_files = []
    
    for run_id in run_ids:
        print(f"[evaluate] Processing run: {run_id}")
        
        # Try WandB first
        run_data = fetch_wandb_run_data(wandb_entity, wandb_project, run_id)
        
        # Fallback to local results
        if run_data is None:
            print(f"[evaluate] Falling back to local results for {run_id}")
            run_data = load_local_results(results_dir, run_id)
        
        if run_data is None:
            print(f"[evaluate] Warning: No data found for {run_id}")
            continue
        
        all_run_data[run_id] = run_data
        
        # Export per-run metrics
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = run_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(run_data.get('summary', {}), f, indent=2)
        generated_files.append(str(metrics_file))
        print(f"[evaluate] Saved metrics: {metrics_file}")
        
        # Create per-run figures
        if 'history' in run_data and run_data['history']:
            fig_path = create_accuracy_plot(run_data, results_dir, run_id)
            if fig_path:
                generated_files.append(fig_path)
    
    if not all_run_data:
        print("[evaluate] Error: No run data available")
        sys.exit(1)
    
    # Compute aggregated metrics
    aggregated = compute_aggregated_metrics(all_run_data)
    
    # Save aggregated metrics
    comparison_dir = results_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated_file = comparison_dir / 'aggregated_metrics.json'
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    generated_files.append(str(aggregated_file))
    print(f"[evaluate] Saved aggregated metrics: {aggregated_file}")
    
    # Create comparison figures
    comparison_fig = create_comparison_plot(all_run_data, results_dir)
    if comparison_fig:
        generated_files.append(comparison_fig)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Primary metric: {aggregated['primary_metric']}")
    print(f"Best proposed: {aggregated['best_proposed']} (accuracy={aggregated['best_proposed_accuracy']:.4f})")
    print(f"Best baseline: {aggregated['best_baseline']} (accuracy={aggregated['best_baseline_accuracy']:.4f})")
    if aggregated['gap'] is not None:
        print(f"Gap: {aggregated['gap']:+.4f}")
    print("="*60)
    
    # Print all generated files
    print("\nGenerated files:")
    for path in generated_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
