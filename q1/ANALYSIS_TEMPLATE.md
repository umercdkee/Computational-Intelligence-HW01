# CS 451 - Assignment 1: Evolutionary Algorithm for TSP
## Analysis Report

**Team Members:** [Add your names]  
**Date:** [Add submission date]

---

## 1. Problem Formulation

### 1.1 Problem Definition
Travelling Salesman Problem (TSP) with 194 cities from Qatar dataset. The goal is to find the shortest route visiting each city exactly once and returning to the origin.

### 1.2 Chromosome Representation
- **Type:** Permutation-based (ordered list of cities)
- **Format:** Array of integers from 1 to 194
- **Example:** [1, 5, 23, 12, ..., 194]
- **Rationale:** Direct representation of tour order enables straightforward distance calculation and intuitive genetic operators

### 1.3 Fitness Function
```
Fitness = Total Distance of Tour
         = Σ distance(city[i], city[i+1]) for i in 0..192
         + distance(city[193], city[0])
         
Objective: Minimize fitness (shorter distance is better)
```

### 1.4 Genetic Operators

#### Crossover: Order-1 (OX) Crossover
- Selects random segment from parent 1
- Fills remaining cities maintaining parent 2's order
- Preserves good solution patterns while introducing variation

#### Mutation: Reverse Segment Mutation
- Randomly selects two indices
- Reverses the segment between them
- Probability: [mutation_rate from your run]
- Maintains valid permutation structure

---

## 2. Implementation Details

### 2.1 Selection Schemes Tested
1. **Fitness Proportionate Selection (FPS)** - Probability based on fitness rank
2. **Binary Tournament** - Random pairwise tournaments, winner selected
3. **Truncation** - Deterministic top-k selection
4. **Random** - Uniformly random selection
5. **Rank-Based (RBS)** - Probability based on ranking

### 2.2 Experimental Setup
- **Population Size:** 100
- **Offspring per Generation:** 125
- **Maximum Generations:** 10,000
- **Early Stopping (Patience):** 2,000 generations without improvement
- **Number of Runs:** 10 (different random seeds)
- **Mutation Rate:** 0.25

---

## 3. Results and Analysis

### 3.1 Summary Statistics
[The summary_YYYYMMDD_HHMMSS.csv file contains all quantitative results]

**Key Metrics:**
- Final Average BSF (Best-So-Far distance)
- Standard Deviation (variance across 10 runs)
- Average Generations to Convergence
- Best/Worst solutions found

### 3.2 Graph Analysis

#### Best-So-Far (BSF) Plots
- Show convergence trajectory
- Lower curves indicate better performance
- Steeper initial drops = faster improvement
- Plateaus indicate convergence

#### Average-So-Far (ASF) Plots
- Show population quality over time
- Measures diversity and selection pressure
- Gap between ASF and BSF indicates population variance

### 3.3 Scheme Performance Comparison

**Best Performing Combination:**
- Parent Selection: [e.g., Binary Tournament]
- Survival Selection: [e.g., Truncation]
- Final Avg BSF: [distance value]
- Analysis: [Why did this perform best?]

**Worst Performing Combination:**
- Parent Selection: [e.g., Random]
- Survival Selection: [e.g., Random]
- Final Avg BSF: [distance value]
- Analysis: [Why did this perform poorly?]

---

## 4. Key Findings and Insights

### 4.1 Selection Schemes Impact

#### Parent Selection (Selection Pressures)
- **Fitness Proportionate:** [Your observations]
  - Strengths: 
  - Weaknesses:

- **Rank-Based:** [Your observations]
  - Strengths:
  - Weaknesses:

- **Binary Tournament:** [Your observations]
  - Strengths:
  - Weaknesses:

- **Truncation:** [Your observations]
  - Strengths:
  - Weaknesses:

- **Random:** [Your observations]
  - Strengths:
  - Weaknesses:

### 4.2 Convergence Patterns
- [Describe how different combinations converge]
- [Early convergence vs slow improvement]
- [Premature convergence observations]

### 4.3 Solution Quality
- [Compare distances to known optimum (9352)]
- [Compare to previous best achieved (9573)]
- [Your best solution distance]
- [Gap analysis and feasibility]

### 4.4 Computational Efficiency
- [Average runtime per combination]
- [Computational complexity observations]
- [Impact of population and offspring parameters]

---

## 5. Conclusions and Recommendations

### 5.1 Best Configuration
Based on our experiments, the following configuration works best for TSP:
- **Parent Selection:** [Your choice]
- **Survival Selection:** [Your choice]
- **Justification:** [Why this combination is optimal]

### 5.2 Trade-offs Observed
- Solution Quality vs Computation Time
- Population Diversity vs Convergence Speed
- Exploration vs Exploitation

### 5.3 Future Improvements
- [Advanced operators (e.g., 2-opt local search)]
- [Adaptive mutation rates]
- [Hybrid algorithms combining EA with other metaheuristics]
- [Parameter auto-tuning]

### 5.4 General Insights
- The impact of selection pressure on convergence
- Population diversity effects on solution quality
- Early stopping effectiveness with patience parameter

---

## 6. References
- Dataset: http://www.math.uwaterloo.ca/tsp/world/countries.html
- Assignment Description: CS 451 Spring 2026

---

## Appendix: Additional Observations

### A1. Generation Statistics
[Include observations about number of generations each method took]

### A2. Variance Analysis
[Discuss which schemes had more consistent results across 10 runs]

### A3. Visual Pattern Analysis
[Describe patterns visible in the plots]

