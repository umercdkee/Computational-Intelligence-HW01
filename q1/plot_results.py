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
    """Create separate plots for BSF and ASF for each combination"""
    
    if not results:
        print("No results to plot")
        return
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Create separate figures for BSF and ASF
    for result in results:
        parent_sel = result['parent_selector']
        survival_sel = result['survival_selector']
        avg_bsf = np.array(result['avg_bsf'])
        avg_asf = np.array(result['avg_asf'])
        std_bsf = np.array(result['std_bsf'])
        generations = np.arange(1, len(avg_bsf) + 1)
        
        # Plot BSF
        fig = plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_bsf, 'b-', linewidth=2, label='Avg BSF')
        plt.fill_between(generations, 
                         avg_bsf - std_bsf, 
                         avg_bsf + std_bsf, 
                         alpha=0.2, color='blue', label='±1 Std Dev')
        plt.xlabel('Generation', fontsize=11)
        plt.ylabel('Best Fitness (Distance)', fontsize=11)
        plt.title(f'Best-So-Far (BSF) - Parent: {parent_sel} | Survival: {survival_sel}', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        filename_bsf = f"plots/{parent_sel}_vs_{survival_sel}_BSF.png"
        plt.savefig(filename_bsf, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filename_bsf}")
        plt.close()
        
        # Plot ASF
        fig = plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_asf, 'r-', linewidth=2, label='Avg ASF')
        plt.xlabel('Generation', fontsize=11)
        plt.ylabel('Average Fitness (Distance)', fontsize=11)
        plt.title(f'Average-So-Far (ASF) - Parent: {parent_sel} | Survival: {survival_sel}', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        filename_asf = f"plots/{parent_sel}_vs_{survival_sel}_ASF.png"
        plt.savefig(filename_asf, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filename_asf}")
        plt.close()

def create_comparison_plot(results):
    """Create separate comparison plots for BSF and ASF"""
    
    if not results:
        return
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # BSF Comparison
    fig = plt.figure(figsize=(12, 7))
    for idx, result in enumerate(results):
        parent_sel = result['parent_selector']
        survival_sel = result['survival_selector']
        avg_bsf = np.array(result['avg_bsf'])
        generations = np.arange(1, len(avg_bsf) + 1)
        
        label = f"{parent_sel.replace('_', ' ')} + {survival_sel.replace('_', ' ')}"
        plt.plot(generations, avg_bsf, linewidth=2, color=colors[idx], label=label, marker='o', markersize=3)
    
    plt.xlabel('Generation', fontsize=11)
    plt.ylabel('Best Fitness (Distance)', fontsize=11)
    plt.title('Comparison of All Selection Schemes - BSF', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/comparison_BSF.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: plots/comparison_BSF.png")
    plt.close()
    
    # ASF Comparison
    fig = plt.figure(figsize=(12, 7))
    for idx, result in enumerate(results):
        parent_sel = result['parent_selector']
        survival_sel = result['survival_selector']
        avg_asf = np.array(result['avg_asf'])
        generations = np.arange(1, len(avg_asf) + 1)
        
        label = f"{parent_sel.replace('_', ' ')} + {survival_sel.replace('_', ' ')}"
        plt.plot(generations, avg_asf, linewidth=2, color=colors[idx], label=label, marker='o', markersize=3)
    
    plt.xlabel('Generation', fontsize=11)
    plt.ylabel('Average Fitness (Distance)', fontsize=11)
    plt.title('Comparison of All Selection Schemes - ASF', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/comparison_ASF.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: plots/comparison_ASF.png")
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
