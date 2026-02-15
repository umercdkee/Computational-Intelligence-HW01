# Python file that will serve as implementation of EA algorithm for solving TSP on Qatar dataset

# Appendix – A
# EA Implementation:
# Your EA will perform the following cycle:
# • Step 0: Analyse the problem that you are addressing and decide your chromosome
# representation and fitness function.
# • Step 1: Initialize the population randomly or with potentially good solutions.
# • Step 2: Compute the fitness of each individual in the population.
# • Step 3: Select parents using a selection procedure.
# • Step 4: Create offspring by crossover and mutation operators.
# • Step 5: Compute the fitness of the new offspring.
# • Step 6: Select members of population to die using a selection procedure.
# • Step 7: Go to Step 2 until the termination criteria are met.


# Step 0: Chromosome: vector of num_cities, Fitness function: Distance between i to i+1.

import os
import sys
import random
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Need to do so it can locate BaseEA
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_class import BaseEA

class EA_TSP(BaseEA):
    def __init__(self, distance_matrix, population_size, mutation_rate, num_offspring,
                 parent_selector="rank_based", survival_selector="truncation"):
        super().__init__(population_size, minimize=True, mutation_rate=mutation_rate, num_offspring=num_offspring)
        self.num_cities = distance_matrix.shape[0]
        self.chromosomes = np.empty((self.population_size, self.num_cities), dtype=int)
        self.curr_fitness = np.zeros(self.population_size)
        self.distance_matrix = distance_matrix
        self.mutation_rate = mutation_rate
        self.num_offspring = num_offspring
        self.parent_selector = parent_selector
        self.survival_selector = survival_selector

    def _select_indices(self, scheme, k, pick_worst=False):
        """Call the named selection scheme. If pick_worst=True, selection
        favours high-fitness individuals (used to find who to kill in
        a minimisation problem)."""
        original = self.minimize
        if pick_worst:
            self.minimize = not original
        try:
            if scheme == "random":
                return self.random_selection(k)
            elif scheme == "truncation":
                return self.truncation_selection(k)
            elif scheme == "binary_tournament":
                return self.binary_tournament(k)
            elif scheme == "fitness_proportionate":
                return self.fitness_proportionate(k)
            elif scheme == "rank_based":
                return self.rank_based(k)
            else:
                raise ValueError(f"Unknown selection scheme: {scheme}")
        finally:
            self.minimize = original

    def calculate_fitness(self, x):
        fitness_val = 0
        for j in range(self.num_cities):
            src = x[j]
            dst = x[(j + 1) % self.num_cities]
            fitness_val += self.distance_matrix[src - 1][dst - 1]
        return fitness_val

    def initialize_population(self):
        for i in range(self.population_size):
            self.chromosomes[i] = [j for j in range(1, self.num_cities + 1)]
            random.shuffle(self.chromosomes[i])
            self.curr_fitness[i] = self.calculate_fitness(self.chromosomes[i])

    def create_offspring(self):
        all_children = []
        for _ in range(self.num_offspring):
            parents_idx = self._select_indices(self.parent_selector, 2)

            # Crossover
            left = np.random.randint(0,self.num_cities)
            right = np.random.randint(left,self.num_cities)

            new_child = np.zeros(self.num_cities, dtype=int)

            new_child[left:right+1] = self.chromosomes[parents_idx[0]][left:right+1]
            placed = set(new_child[left:right+1])  # track what's already placed

            ptr_new_child = (right + 1) % self.num_cities
            ptr_traverse = (right + 1) % self.num_cities

            while len(placed) < self.num_cities:
                city = self.chromosomes[parents_idx[1]][ptr_traverse]
                if city not in placed:
                    new_child[ptr_new_child] = city
                    placed.add(city)
                    ptr_new_child = (ptr_new_child + 1) % self.num_cities
                ptr_traverse = (ptr_traverse + 1) % self.num_cities
                
            # Mutation
            prob_mutation = np.random.uniform(0, 1)
            if prob_mutation <= self.mutation_rate:
                # Reverses the segment between two chosen indices
                l = random.randint(0, self.num_cities-1)
                r = random.randint(l, self.num_cities-1)
                new_child[l:r+1] = new_child[l:r+1][::-1]

            all_children.append(new_child)

        return all_children


    def select_to_kill(self):
        return self._select_indices(self.survival_selector, 1, pick_worst=True)[0]

    def run_loop(self, num_generations):
        bsf_history = []
        asf_history = []

        for gen in range(num_generations):
            new_children = self.create_offspring()

            for child in new_children:
                child_fitness = self.calculate_fitness(child)
                kill_idx = self.select_to_kill()

                if child_fitness < self.curr_fitness[kill_idx]:
                    self.chromosomes[kill_idx] = child
                    self.curr_fitness[kill_idx] = child_fitness

            curr_best = np.min(self.curr_fitness)
            curr_avg = np.mean(self.curr_fitness)
            bsf_history.append(curr_best)
            asf_history.append(curr_avg)

            if gen % 1000 == 0:
                print(f"Gen {gen}: Best Dist = {curr_best:.2f}")

        best_idx = np.argmin(self.curr_fitness)
        best_route = self.chromosomes[best_idx]
        best_distance = self.curr_fitness[best_idx]

        return best_route, best_distance, bsf_history, asf_history
    

def parse_tsp_file(filename):
    """Parse TSP file and extract coordinates"""
    coordinates = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    in_coord_section = False
    for line in lines:
        line = line.strip()
        
        if line == "NODE_COORD_SECTION":
            in_coord_section = True
            continue
        
        if line == "EOF":
            break
        
        if in_coord_section and line:
            parts = line.split()
            if len(parts) == 3:
                x, y = float(parts[1]), float(parts[2])
                coordinates.append((x, y))
    
    return np.array(coordinates)


def compute_distance_matrix(coordinates):
    """Compute Euclidean distance matrix from coordinates"""
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                distance_matrix[i][j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return distance_matrix


def main():
    # EA parameters
    population_size = 100
    num_generations = 3000
    mutation_rate = 0.25
    num_offspring = 125
    num_iterations = 10

    # Test combinations: (parent_selector, survival_selector)
    combinations = [
        ("fitness_proportionate", "truncation"),
        ("binary_tournament", "truncation"),
        ("truncation", "truncation"),
        ("random", "random"),
        ("rank_based", "random"),
    ]

    # Parse TSP file (do once)
    tsp_file = "qa194.tsp"
    coordinates = parse_tsp_file(tsp_file)
    num_cities = len(coordinates)
    distance_matrix = compute_distance_matrix(coordinates)

    print(f"Number of cities: {num_cities}")
    print(f"Running {len(combinations)} combinations × {num_iterations} iterations each\n")

    # Results storage
    all_results = []
    summary_data = []
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for parent_sel, survival_sel in combinations:
        print(f"\n{'='*60}")
        print(f"Testing: Parent={parent_sel}, Survival={survival_sel}")
        print(f"{'='*60}")
        
        bsf_runs = []
        asf_runs = []
        best_distances = []
        generations_run = []

        for iteration in range(num_iterations):
            print(f"  Iteration {iteration + 1}/{num_iterations}...", end=" ", flush=True)

            ea = EA_TSP(distance_matrix, population_size, mutation_rate, num_offspring,
                        parent_sel, survival_sel)
            ea.initialize_population()

            best_route, best_distance, bsf_history, asf_history = ea.run_loop(num_generations)
            
            bsf_runs.append(bsf_history)
            asf_runs.append(asf_history)
            best_distances.append(best_distance)
            generations_run.append(len(bsf_history))
            
            print(f"Best: {best_distance:.2f}")

        # Compute statistics
        min_len = min(len(run) for run in bsf_runs)
        avg_bsf = np.mean([run[:min_len] for run in bsf_runs], axis=0)
        avg_asf = np.mean([run[:min_len] for run in asf_runs], axis=0)
        std_bsf = np.std([run[:min_len] for run in bsf_runs], axis=0)
        
        # Summary statistics
        final_best = np.mean(best_distances)
        final_std = np.std(best_distances)
        avg_gens = np.mean(generations_run)

        summary_data.append({
            "Parent_Selector": parent_sel,
            "Survival_Selector": survival_sel,
            "Final_Avg_BSF": final_best,
            "Final_Std_BSF": final_std,
            "Avg_Generations": avg_gens,
            "Min_Best_Distance": np.min(best_distances),
            "Max_Best_Distance": np.max(best_distances),
        })

        # Save detailed histories
        all_results.append({
            "parent_selector": parent_sel,
            "survival_selector": survival_sel,
            "avg_bsf": avg_bsf.tolist(),
            "avg_asf": avg_asf.tolist(),
            "std_bsf": std_bsf.tolist(),
            "best_distances": best_distances,
            "generations_run": generations_run.copy(),
            "min_len": min_len,
        })

        print(f"\nResults:")
        print(f"  Final Avg BSF: {final_best:.2f} ± {final_std:.2f}")
        print(f"  Generations: {avg_gens:.0f}")

    # Save all detailed results as JSON
    json_file = f"results/detailed_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Detailed results saved to: {json_file}")

    # Save summary as CSV
    summary_df = pd.DataFrame(summary_data)
    csv_file = f"results/summary_{timestamp}.csv"
    summary_df.to_csv(csv_file, index=False)
    print(f"✓ Summary saved to: {csv_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
