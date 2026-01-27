import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def analyze_results(root_dir="runs", output_dir="analysis_results"):
    print(f"Loading results from {root_dir}...")
    root = Path(root_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # 1. Load Data
    for run_dir in root.iterdir():
        if not run_dir.is_dir(): continue
        
        # Parse Run Name: SET-ID_attack[_budget]_seedS
        parts = run_dir.name.split('_')
        if len(parts) < 3: continue
        
        set_id = parts[0]
        seed_part = parts[-1]
        
        # Robust attack name parsing
        # Remove 'seedX' and potential budget suffix '10k', '20k' etc.
        name_parts = parts[1:-1]
        attack_name = "_".join(name_parts)
        
        # Clean up budget suffixes for AL attacks
        for suffix in ["_1k", "_10k", "_20k", "_50k", "_100k"]:
            if attack_name.endswith(suffix):
                attack_name = attack_name[:-len(suffix)]
                
        # Find latest timestamp
        timestamps = sorted([d for d in run_dir.iterdir() if d.is_dir()])
        if not timestamps: continue
        latest_run = timestamps[-1]
        
        # Iterate Seeds
        for seed_dir in latest_run.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"): continue
            
            summary_path = seed_dir / "summary.json"
            if not summary_path.exists(): continue
            
            with open(summary_path, 'r') as f:
                data = json.load(f)
            
            # Extract Metrics
            max_budget = int(data.get("max_budget", 20000))
            
            for cp_str, tracks in data.get("checkpoints", {}).items():
                if "track_a" not in tracks: continue
                metrics = tracks["track_a"]
                
                # Filter for AL attacks: only use the final checkpoint of each run
                # to ensure we use the fully-optimized result for that budget.
                # Heuristic: if run_dir has budget suffix, only take matching checkpoint.
                run_budget_suffix = f"_{int(cp_str)//1000}k"
                is_budget_run = run_dir.name.endswith(run_budget_suffix + f"_{seed_part}")
                
                # If it's an AL attack (has budget specific runs), enforce matching
                is_al = any(x in attack_name for x in ["activethief", "swiftthief", "cloudleak", "inversenet"])
                if is_al and not is_budget_run:
                    # This checkpoint is an intermediate point of a larger run, 
                    # OR a final point of a mismatched run.
                    # We only want: Run=10k -> Checkpoint=10k.
                    # If this run is "activethief_20k", we skip checkpoint 1000.
                    if int(cp_str) != max_budget:
                        continue

                results.append({
                    "Set": set_id,
                    "Attack": attack_name.upper(),
                    "Budget": int(cp_str),
                    "Seed": int(seed_part.replace("seed", "")),
                    "Accuracy": metrics["acc_gt"],
                    "Agreement": metrics["agreement"],
                    "KL": metrics.get("kl_mean"),
                    "L1": metrics.get("l1_mean")
                })

    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/master_results.csv", index=False)
    print(f"Loaded {len(df)} records. Saved to master_results.csv")

    # 2. Budget-wise Tables (Mean ± Std)
    unique_budgets = sorted(df["Budget"].unique())
    
    with open(f"{output_dir}/report_tables.md", "w") as report:
        report.write("# Benchmark Analysis Report\n\n")
        
        for b in unique_budgets:
            b_df = df[df["Budget"] == b]
            if b_df.empty: continue
            
            stats = b_df.groupby(["Set", "Attack"])[["Accuracy", "Agreement"]].agg(["mean", "std"]).reset_index()
            
            # Format
            for metric in ["Accuracy", "Agreement"]:
                mean_col = (metric, "mean")
                std_col = (metric, "std")
                stats[metric] = stats.apply(
                    lambda x: f"{x[mean_col]:.4f} ± {x[std_col]:.4f}" if pd.notnull(x[std_col]) else f"{x[mean_col]:.4f}", 
                    axis=1
                )
            
            # Pivot
            pivot = stats.pivot(index="Attack", columns="Set", values="Accuracy")
            pivot.columns = [f"{c[0]}" for c in pivot.columns] # Flatten headers
            
            print(f"Generating table for Budget {b}...")
            csv_path = f"{output_dir}/table_accuracy_{b}.csv"
            pivot.to_csv(csv_path)
            
            report.write(f"## Budget: {b}\n")
            report.write(pivot.to_markdown())
            report.write("\n\n")

    # 3. Learning Curve Plots
    print("Generating plots...")
    sns.set_theme(style="whitegrid")
    
    for set_id in df["Set"].unique():
        plt.figure(figsize=(10, 6))
        subset = df[df["Set"] == set_id]
        
        # Line plot with confidence interval
        sns.lineplot(
            data=subset, 
            x="Budget", 
            y="Accuracy", 
            hue="Attack", 
            style="Attack",
            markers=True, 
            dashes=False
        )
        
        plt.xscale("log")
        plt.title(f"Attack Performance on {set_id}")
        plt.ylabel("Accuracy (GT)")
        plt.xlabel("Query Budget (Log Scale)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plot_accuracy_{set_id}.png", dpi=300)
        plt.close()

    print(f"Analysis complete. Check {output_dir}/")

if __name__ == "__main__":
    analyze_results()
