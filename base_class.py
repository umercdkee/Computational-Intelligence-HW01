import numpy as np


class BaseEA:
	def __init__(self, population_size, minimize, mutation_rate, num_offspring):
		self.population_size = population_size
		self.minimize = minimize
		self.chromosomes = None
		self.curr_fitness = None
		self.mutation_rate = mutation_rate
		self.num_offspring = num_offspring

	# Problem-specific
	def initialize_population(self):
		pass

	def create_offspring(self):
		pass

	def calculate_fitness(self, chromosome):
		pass

	def select_to_kill(self):
		pass

	# selection operators
	def random_selection(self, k):
		return np.random.choice(np.arange(self.population_size), size=k, replace=False).tolist()

	def truncation_selection(self, k):
		sorted_indices = np.argsort(self.curr_fitness)
		if self.minimize:
			selected_indices = sorted_indices[:k]
		else:
			selected_indices = sorted_indices[::-1][:k]
		return selected_indices.tolist()

	def binary_tournament(self, k):
		selected_indices = []
		for _ in range(k):
			idx1, idx2 = np.random.choice(np.arange(self.population_size), size=2, replace=False)
			if self.minimize:
				winner = idx1 if self.curr_fitness[idx1] < self.curr_fitness[idx2] else idx2
			else:
				winner = idx1 if self.curr_fitness[idx1] > self.curr_fitness[idx2] else idx2
			selected_indices.append(winner)
		return selected_indices

	def fitness_proportionate(self, k):
		eps = 1e-9
		if self.minimize:
			scores = 1.0 / (self.curr_fitness + eps)
		else:
			scores = self.curr_fitness.copy()

		total = np.sum(scores)
		if total <= 0:
			return self.random_selection(k)

		probs = scores / total
		parents_idx = np.random.choice(
			np.arange(self.population_size),
			size=k,
			p=probs,
			replace=False,
		)
		return parents_idx.tolist()

	def rank_based(self, k):
		rank_indices = np.argsort(self.curr_fitness)
		if not self.minimize:
			rank_indices = rank_indices[::-1]

		total = (self.population_size * (self.population_size + 1)) / 2.0
		probs = np.zeros(self.population_size)

		curr_rank = self.population_size
		for i in rank_indices:
			probs[i] = curr_rank / total
			curr_rank -= 1

		rand_num = np.random.uniform(0, 1, size=k)
		parents_idx = []
		for val in rand_num:
			left = 0.0
			right = 0.0
			for j in range(len(probs)):
				left = right
				right = left + probs[j]
				if val >= left and val <= right:
					parents_idx.append(j)
					break

		return parents_idx

	# Common EA loop
	def run_loop(self, num_generations, patience=15000):
		best_ever_fitness = float("inf") if self.minimize else float("-inf")
		termination_count = 0

		for gen in range(num_generations):
			new_child = self.create_offspring()
			new_child_fitness = self.calculate_fitness(new_child)

			kill_idx = self.select_to_kill()

			if self.minimize:
				should_replace = new_child_fitness < self.curr_fitness[kill_idx]
				curr_best = np.min(self.curr_fitness)
			else:
				should_replace = new_child_fitness > self.curr_fitness[kill_idx]
				curr_best = np.max(self.curr_fitness)

			if should_replace:
				self.chromosomes[kill_idx] = new_child
				self.curr_fitness[kill_idx] = new_child_fitness

			improved = curr_best < best_ever_fitness if self.minimize else curr_best > best_ever_fitness
			if improved:
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
				label = "Best Dist" if self.minimize else "Best Fitness"
				print(f"Gen {gen}: {label} = {curr_best:.2f}")

		best_idx = np.argmin(self.curr_fitness) if self.minimize else np.argmax(self.curr_fitness)
		best_route = self.chromosomes[best_idx]
		best_value = self.curr_fitness[best_idx]

		return best_route, best_value
