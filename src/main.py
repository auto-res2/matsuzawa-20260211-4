"""
Main orchestrator for chain-of-thought prompt tuning experiments.
Invokes inference.py as a subprocess with mode-specific overrides.
"""
import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Orchestrate a single run_id for inference-only chain-of-thought experiments.
    
    Task type: Inference-only prompt tuning (no training).
    Applies mode-specific overrides and invokes inference.py.
    """
    
    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        # Sanity check mode: minimal execution
        OmegaConf.set_struct(cfg, False)
        cfg.wandb.mode = "disabled"
        cfg.inference.sanity_check_samples = 10  # Process 10 samples
        OmegaConf.set_struct(cfg, True)
        print(f"[main] Running in sanity_check mode: wandb disabled, limited to {cfg.inference.sanity_check_samples} samples")
    elif cfg.mode == "main":
        # Full execution
        print(f"[main] Running in main mode: full dataset, wandb online")
    else:
        print(f"[main] Running in {cfg.mode} mode")
    
    # Create results directory
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare command to invoke inference.py
    # Pass the entire config via Hydra command-line overrides
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.inference",
    ]
    
    # Add Hydra overrides from command line
    # We need to reconstruct the original command-line args
    import sys
    original_args = sys.argv[1:]  # Get original hydra args
    cmd.extend(original_args)
    
    print(f"[main] Invoking inference.py: {' '.join(cmd)}")
    
    # Invoke inference.py as subprocess
    result = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        check=False,  # Don't raise exception on non-zero exit
    )
    
    if result.returncode != 0:
        print(f"[main] inference.py exited with code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"[main] Completed run {cfg.run.run_id}")


if __name__ == "__main__":
    main()
