import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def load_latest_results():
    """Load the most recent results JSON file"""
    results_dir = "results"
    json_files = glob.glob(os.path.join(results_dir, "detailed_results_*.json"))
    
    if not json_files:
        print(f"No results found in {results_dir}/")
        return None
    
    latest_file = max(json_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)
    
def plot_results(results):
    """Create side-by-side plots of BSF and ASF for each combination"""
    
    if not results:
        print("No results to plot")
        return
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Create a figure for each combination (2 subplots: BSF and ASF)
    for result in results:
        parent_sel = result['parent_selector']
        survival_sel = result['survival_selector']
        avg_bsf = np.array(result['avg_bsf'])
        avg_asf = np.array(result['avg_asf'])
        std_bsf = np.array(result['std_bsf'])
        generations = np.arange(1, len(avg_bsf) + 1)
        
        # Create figure with 2 subplots (1 row, 2 columns)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Parent: {parent_sel} | Survival: {survival_sel}', fontsize=14, fontweight='bold')
        
        # Plot 1: Best-So-Far (BSF)
        axes[0].plot(generations, avg_bsf, 'b-', linewidth=2, label='Avg BSF')
        axes[0].fill_between(generations, 
                             avg_bsf - std_bsf, 
                             avg_bsf + std_bsf, 
                             alpha=0.2, color='blue', label='±1 Std Dev')
        axes[0].set_xlabel('Generation', fontsize=11)
        axes[0].set_ylabel('Best Fitness (Distance)', fontsize=11)
        axes[0].set_title('Best-So-Far (BSF) Progression', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Average-So-Far (ASF)
        axes[1].plot(generations, avg_asf, 'r-', linewidth=2, label='Avg ASF')
        axes[1].set_xlabel('Generation', fontsize=11)
        axes[1].set_ylabel('Average Fitness (Distance)', fontsize=11)
        axes[1].set_title('Average-So-Far (ASF) Progression', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save figure
        filename = f"plots/{parent_sel}_vs_{survival_sel}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

def create_comparison_plot(results):
    """Create a comparison plot showing all combinations on same graph"""
    
    if not results:
        return
    
    # Create separate comparison plots for BSF and ASF
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Comparison of All Selection Schemes', fontsize=14, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, result in enumerate(results):
        parent_sel = result['parent_selector']
        survival_sel = result['survival_selector']
        avg_bsf = np.array(result['avg_bsf'])
        avg_asf = np.array(result['avg_asf'])
        generations = np.arange(1, len(avg_bsf) + 1)
        
        label = f"{parent_sel.replace('_', ' ')} + {survival_sel.replace('_', ' ')}"
        
        axes[0].plot(generations, avg_bsf, linewidth=2, color=colors[idx], label=label, marker='o', markersize=3)
        axes[1].plot(generations, avg_asf, linewidth=2, color=colors[idx], label=label, marker='o', markersize=3)
    
    # Configure BSF plot
    axes[0].set_xlabel('Generation', fontsize=11)
    axes[0].set_ylabel('Best Fitness (Distance)', fontsize=11)
    axes[0].set_title('Average BSF Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=9)
    
    # Configure ASF plot
    axes[1].set_xlabel('Generation', fontsize=11)
    axes[1].set_ylabel('Average Fitness (Distance)', fontsize=11)
    axes[1].set_title('Average ASF Comparison', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/all_combinations_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: plots/all_combinations_comparison.png")
    plt.close()

def main():
    print("Loading results...")
    results = load_latest_results()
    
    if results:
        print(f"\nGenerating plots for {len(results)} combinations...\n")
        plot_results(results)
        create_comparison_plot(results)
        print("\n✓ All plots generated successfully!")
        print("  Check the 'plots/' folder for your graphs")
    else:
        print("Please run q1.py first to generate results")

if __name__ == "__main__":
    main()
