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

import random
import numpy as np

class EA_TSP:
    def __init__(self, distance_matrix, population_size):

        self.population_size = population_size
        self.num_cities = distance_matrix.shape[0]

        self.chromosomes = np.empty((self.population_size, self.num_cities), dtype=int)
        self.curr_fitness = np.zeros(self.population_size)
        
        self.distance_matrix = distance_matrix
        

    def initialize_population(self):

        for i in range(self.population_size):
            self.chromosomes[i] = [j for j in range(1, self.num_cities+1)]
            random.shuffle(self.chromosomes[i])

            for j in range(self.num_cities):
                src = self.chromosomes[i][j]
                dst = self.chromosomes[i][(j+1) % self.num_cities]

                self.curr_fitness[i] += self.distance_matrix[src-1][dst-1]

    def select_parents(self):
        # Rank based selection
        rank_indices = np.argsort(self.curr_fitness)

        total = (self.population_size*(self.population_size+1))/2

        probs = np.zeros(self.population_size)

        curr_rank = self.population_size
        for i in rank_indices:
            probs[i] = curr_rank / float(total)
            curr_rank -= 1

        rand_num = np.random.uniform(0,1,2)
        parents_idx = []
        for i in rand_num:
            left = 0
            right = 0

            for j in range(len(probs)):
                left = right
                right = left + probs[j]

                if i >= left and i <= right:
                    parents_idx.append(j)
                    break

        return parents_idx
    

    def create_offspring(self):
        parents_idx = self.select_parents()

        # Crossover
        left = np.random.randint(0,self.num_cities)
        right = np.random.randint(left,self.num_cities)

        new_child = np.zeros(self.num_cities, dtype=int)

        new_child[left:right+1] = self.chromosomes[parents_idx[0]][left:right+1]
        count = (right-left) + 1
        ptr_new_child = (right + 1) % self.num_cities
        ptr_traverse = (right + 1) % self.num_cities

        while True:
            if count == self.num_cities:
                break

            if self.chromosomes[parents_idx[1]][ptr_traverse] not in new_child:
                new_child[ptr_new_child] = self.chromosomes[parents_idx[1]][ptr_traverse]
                ptr_new_child = (ptr_new_child + 1) % self.num_cities
                ptr_traverse = (ptr_traverse + 1) % self.num_cities
            else:
                ptr_traverse = (ptr_traverse + 1) % self.num_cities

            if (ptr_traverse == right):
                break

        # Mutation
        random_nums = np.random.choice(a=self.num_cities, size=2, replace= False)

        new_child[random_nums[0]], new_child[random_nums[1]] = new_child[random_nums[1]], new_child[random_nums[0]]

        return new_child
    

    def calculate_fitness(self, x):
        fitness_val = 0
        for j in range(self.num_cities):
            src = x[j]
            dst = x[(j+1) % self.num_cities]

            fitness_val += self.distance_matrix[src-1][dst-1]

        return fitness_val
    

    def kill_members(self):
        # Rank based selection
        rank_indices = np.argsort(self.curr_fitness)[::1]

        total = (self.population_size*(self.population_size+1))/2
        probs = np.zeros(self.population_size)

        curr_rank = self.population_size
        for i in rank_indices:
            probs[i] = curr_rank / float(total)
            curr_rank -= 1

        rand_num = np.random.uniform(0,1,2)
        kill_user = []
        for i in rand_num:
            left = 0
            right = 0

            for j in range(len(probs)):
                left = right
                right = left + probs[j]

                if i >= left and i <= right:
                    kill_user.append(j)
                    break

        return kill_user


    def run_loop(self, num_generations):

        for gen in range(num_generations):
            new_child = self.create_offspring()
            new_child_fitness = self.calculate_fitness(new_child)

            kill_user = self.kill_members()

            self.chromosomes[kill_user[0]] = new_child
            self.curr_fitness[kill_user[0]] = new_child_fitness

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
    # Parse TSP file
    tsp_file = "qa194.tsp"
    coordinates = parse_tsp_file(tsp_file)
    num_cities = len(coordinates)
    
    print(f"Number of cities: {num_cities}")
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(coordinates)
    
    # Initialize EA
    population_size = 50
    num_generations = 500
    
    ea = EA_TSP(distance_matrix, population_size)
    ea.initialize_population()
    
    print(f"Population size: {population_size}")
    print(f"Number of generations: {num_generations}")
    
    # Run EA
    best_route, best_distance = ea.run_loop(num_generations)
    
    print(f"\nBest route found: {best_route}")
    print(f"Best distance: {best_distance:.2f}")


if __name__ == "__main__":
    main()



        
