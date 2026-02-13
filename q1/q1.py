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

# Need to do so it can locate BaseEA
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_class import BaseEA

class EA_TSP(BaseEA):
    def __init__(self, distance_matrix, population_size, mutation_rate, num_offspring):
        super().__init__(population_size, minimize=True, mutation_rate=mutation_rate, num_offspring=num_offspring)
        self.num_cities = distance_matrix.shape[0]
        self.chromosomes = np.empty((self.population_size, self.num_cities), dtype=int)
        self.curr_fitness = np.zeros(self.population_size)
        self.distance_matrix = distance_matrix
        self.mutation_rate = mutation_rate
        self.num_offspring = num_offspring

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
        parents_idx = self.rank_based(2)

        all_children = []
        for _ in range(self.num_offspring):

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
        return np.argmax(self.curr_fitness)

    def run_loop(self, num_generations, patience=15000):
        best_ever_fitness = float("inf")
        termination_count = 0

        for gen in range(num_generations):
            new_children = self.create_offspring()

            for child in new_children:
                child_fitness = self.calculate_fitness(child)
                kill_idx = self.select_to_kill()

                if child_fitness < self.curr_fitness[kill_idx]:
                    self.chromosomes[kill_idx] = child
                    self.curr_fitness[kill_idx] = child_fitness

            curr_best = np.min(self.curr_fitness)
            if curr_best < best_ever_fitness:
                best_ever_fitness = curr_best
                termination_count = 0
            else:
                termination_count += 1

            if termination_count >= patience:
                print(
                    f"Terminating at generation {gen} due to no improvement in {patience} generations."
                )
                break

            if gen % 1000 == 0:
                print(f"Gen {gen}: Best Dist = {curr_best:.2f}")

        best_idx = np.argmin(self.curr_fitness)
        best_route = self.chromosomes[best_idx]
        best_distance = self.curr_fitness[best_idx]

        return best_route, best_distance
    

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
    # Initialize EA
    population_size = 500
    num_generations = 500000
    mutation_rate = 0.1
    num_offspring = 2
    num_iterations = 5



    # Parse TSP file
    tsp_file = "qa194.tsp"
    coordinates = parse_tsp_file(tsp_file)
    num_cities = len(coordinates)
    
    print(f"Number of cities: {num_cities}")
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(coordinates)
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

    
        ea = EA_TSP(distance_matrix, population_size, mutation_rate, num_offspring)
        ea.initialize_population()
        
        print(f"Population size: {population_size}")
        print(f"Number of generations: {num_generations}")
        
        # Run EA
        best_route, best_distance = ea.run_loop(num_generations)
        
        print(f"\nBest route found: {best_route}")
        print(f"Best distance: {best_distance:.2f}")


if __name__ == "__main__":
    main()



        
