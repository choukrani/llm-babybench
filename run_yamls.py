import os
import subprocess
import glob
import argparse
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

def get_output_path_from_config(config_path):
    """Extract the base output path from a YAML config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get the output path from the config
        if 'output' in config and 'path' in config['output']:
            return Path(config['output']['path'])
        else:
            # Default path from pipeline.py
            return Path("results/results.json")
    except Exception as e:
        print(f"Error parsing config file {config_path}: {str(e)}")
        return None

def config_has_results(config_path):
    """Check if results already exist for this configuration."""
    output_path = get_output_path_from_config(config_path)
    
    if not output_path:
        # If we can't determine the output path, assume it hasn't been run
        return False
    
    # Check if any files match the pattern {stem}_*.{suffix}
    result_pattern = f"{output_path.stem}_*{output_path.suffix}"
    matching_files = list(output_path.parent.glob(result_pattern))
    
    # If we find any matching result files, this config has been run
    return len(matching_files) > 0

def run_config(config_path):
    """Run a single configuration file using runner/main.py"""
    # First check if this config has already been run
    if config_has_results(config_path):
        print(f"Skipping: {config_path} (results already exist)")
        return True, config_path, "Skipped (results exist)"
    
    try:
        start_time = time.time()
        print(f"Starting: {config_path}")
        
        # Execute the runner/main.py script with the config file
        result = subprocess.run(
            ["python3", "-m", "runner.main", "--config", config_path],
            capture_output=True,
            text=True,
            check=True  # Don't raise exception on non-zero exit
        )
        
        # Calculate execution time
        duration = time.time() - start_time
        
        # Check if the run was successful
        if result.returncode == 0:
            print(f"Completed: {config_path} (in {duration:.2f}s)")
            return True, config_path, None
        else:
            print(f"Failed: {config_path} (in {duration:.2f}s)")
            error_message = result.stderr.strip()
            print(f"Error: {error_message[:100]}..." if len(error_message) > 100 else f"Error: {error_message}")
            return False, config_path, error_message
            
    except Exception as e:
        print(f"Exception running {config_path}: {str(e)}")
        return False, config_path, str(e)

def main():
    parser = argparse.ArgumentParser(description="Run all configuration files in the configs directory.")
    parser.add_argument(
        # "--config_dir", type=str, default="configs", 
        "--config_dir", type=str, default="zinfinity_configs",
        help="Directory containing configuration YAML files (defaults to 'configs')."
    )
    parser.add_argument(
        "--max_workers", type=int, default=10,
        help="Maximum number of parallel worker threads."
    )
    parser.add_argument(
        "--force_rerun", action="store_true",
        help="Force rerun of all configs even if they have existing results."
    )
    args = parser.parse_args()
    
    # Find all YAML files in the configs directory
    config_paths = glob.glob(os.path.join(args.config_dir, "*.yaml"))
    
    if not config_paths:
        print(f"No configuration files found in {args.config_dir}")
        return
    
    print(f"Found {len(config_paths)} configuration files to process")
    
    # If we're not forcing reruns, filter out configs that already have results
    if not args.force_rerun:
        pending_configs = []
        skipped_configs = []
        
        for config_path in config_paths:
            if config_has_results(config_path):
                skipped_configs.append(config_path)
            else:
                pending_configs.append(config_path)
        
        print(f"Skipping {len(skipped_configs)} configs with existing results")
        print(f"Processing {len(pending_configs)} configs without results")
        
        config_paths = pending_configs
    
    if not config_paths:
        print("No configurations to run (all have existing results)")
        return
    
    # Statistics
    successful = []
    failed = []
    skipped = []
    
    # Run configurations in parallel
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(executor.map(run_config, config_paths))
    
    # Process results
    for success, config_path, error in results:
        if error == "Skipped (results exist)":
            skipped.append(config_path)
        elif success:
            successful.append(config_path)
        else:
            failed.append((config_path, error))
    
    # Print summary
    print("\n" + "="*80)
    print(f"Execution Summary:")
    print(f"  Total configurations: {len(config_paths) + len(skipped_configs if not args.force_rerun else [])}")
    print(f"  Skipped (existing results): {len(skipped) + len(skipped_configs if not args.force_rerun else [])}")
    print(f"  Successfully ran: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    # If there were failures, write them to a file
    if failed:
        # with open("failed_configs.txt", "w") as f:
        with open("zinfinity_failed_configs.txt", "w") as f:
            f.write("Failed configurations:\n")
            for config_path, error in failed:
                f.write(f"{config_path}: {error}\n")
        # print(f"  Failed configurations written to: failed_configs.txt")
        print(f"  Failed configurations written to: zinfinity_failed_configs.txt")
    
    print("="*80)

if __name__ == "__main__":
    main()
    