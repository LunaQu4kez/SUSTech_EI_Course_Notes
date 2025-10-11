import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict
import random


class EvolutionaryAlgorithm:
    def __init__(self, 
                 population_size: int, 
                 gene_length: int,
                 bounds: List[Tuple[int, int]],
                 objective_function: Callable,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 selection_method: str = "tournament",
                 tournament_size: int = 3):
        """
        Evolutionary Algorithm initialization
        
        Args:
            population_size: Population size
            gene_length: Gene length (dimension of solution vector)
            bounds: Boundaries for each dimension [(min, max), ...]
            objective_function: Objective function to optimize
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
            selection_method: Selection method ("tournament", "roulette")
            tournament_size: Tournament size for tournament selection
        """
        self.population_size = population_size
        self.gene_length = gene_length
        self.bounds = bounds
        self.objective_function = objective_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size       
        self.population = self.initialize_population()
        self.best_individual = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
    def initialize_population(self) -> np.ndarray:
        population = np.zeros((self.population_size, self.gene_length), dtype=int)        
        for i in range(self.population_size):
            for j in range(self.gene_length):
                lower_bound, upper_bound = self.bounds[j]
                population[i, j] = random.randint(lower_bound, upper_bound)                
        return population
    
    def evaluate_fitness(self, individual: np.ndarray) -> float:
        return self.objective_function(individual)
    
    def evaluate_population(self) -> List[float]:
        fitness_values = []
        for individual in self.population:
            fitness = self.evaluate_fitness(individual)
            fitness_values.append(fitness)
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual.copy()               
        return fitness_values
    
    def tournament_selection(self, fitness_values: List[float]) -> int:
        tournament_indices = random.sample(range(self.population_size), self.tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitness)]
        return winner_index
    
    def roulette_selection(self, fitness_values: List[float]) -> int:
        max_fitness = max(fitness_values)
        adjusted_fitness = [max_fitness - f + 1e-10 for f in fitness_values]
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]       
        return random.choices(range(self.population_size), weights=probabilities)[0]
    
    def selection(self, fitness_values: List[float]) -> int:
        if self.selection_method == "tournament":
            return self.tournament_selection(fitness_values)
        elif self.selection_method == "roulette":
            return self.roulette_selection(fitness_values)
        else:
            raise ValueError("No this method")
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()                    
        child1 = parent1.copy()
        child2 = parent2.copy()
        crossover_point = random.randint(1, self.gene_length - 1)        
        child1[:crossover_point] = parent1[:crossover_point]
        child1[crossover_point:] = parent2[crossover_point:]        
        child2[:crossover_point] = parent2[:crossover_point]
        child2[crossover_point:] = parent1[crossover_point:]
        for i in range(self.gene_length):
            lower_bound, upper_bound = self.bounds[i]
            child1[i] = np.clip(child1[i], lower_bound, upper_bound)
            child2[i] = np.clip(child2[i], lower_bound, upper_bound)        
        return child1, child2
    
    def mutation(self, individual: np.ndarray) -> np.ndarray:
        mutated_individual = individual.copy()       
        for i in range(self.gene_length):
            if random.random() < self.mutation_rate:
                lower_bound, upper_bound = self.bounds[i]
                mutated_individual[i] = random.randint(lower_bound, upper_bound)        
        return mutated_individual
    
    def create_new_generation(self, fitness_values: List[float]) -> np.ndarray:
        new_population = []
        elite_index = np.argmin(fitness_values)
        new_population.append(self.population[elite_index].copy())        
        while len(new_population) < self.population_size:
            parent1_idx = self.selection(fitness_values)
            parent2_idx = self.selection(fitness_values)            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)           
            new_population.extend([child1, child2])           
        if len(new_population) > self.population_size:
            new_population = new_population[:self.population_size]            
        return np.array(new_population)
    
    def run(self, generations: int, verbose: bool = True) -> Dict:
        self.fitness_history = []        
        for generation in range(generations):
            fitness_values = self.evaluate_population()
            current_best_fitness = min(fitness_values)
            self.fitness_history.append(current_best_fitness)            
            if verbose and generation % 200 == 0:
                print(f"Generation {generation}: Best Fitness = {current_best_fitness:.6f}")
            self.population = self.create_new_generation(fitness_values)            
        final_fitness_values = self.evaluate_population()       
        if verbose:
            print(f"\nFinal Best Fitness: {self.best_fitness:.10f}")
            print(f"Best Individual: {self.best_individual}")        
        return {
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual,
            'fitness_history': self.fitness_history,
            'final_population': self.population
        }
    
    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Evolutionary Algorithm Convergence')
        plt.grid(True)
        plt.yscale('log')
        plt.show()


# ============= #
# Test Function #
# ============= #

def f1(x):
    return np.sum(x**2)

def f2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def f3(x):
    return np.sum(np.cumsum(x)**2)

def f4(x):
    return np.max(np.abs(x))

def f5(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def f6(x):
    return np.sum((x + 0.5)**2)

def f7(x):
    x = x / 100
    n = len(x)
    return np.sum(np.arange(1, n+1) * x**4) + random.random()

def f8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def f9(x):
    x = x / 100
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def f10(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

def f11(x):
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return sum_part - prod_part + 1

def f12(x):
    n = len(x)
    y = 1 + 0.25 * (x + 1)
    term1 = 10 * np.sin(np.pi * y[0])**2
    term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
    term3 = (y[-1] - 1)**2
    return (np.pi / n) * (term1 + term2 + term3)

def f13(x):
    term1 = np.sin(3 * np.pi * x[0])**2
    term2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    return 0.1 * (term1 + term2 + term3)

def f14(x):
    x = x / 1000
    a = np.array([[-32, -16, 0, 16, 32] * 5,
                  [-32, -16, 0, 16, 32] * 5])
    sum_val = 0
    for j in range(25):
        denominator = j + np.sum((x - a[:, j])**6)
        sum_val += 1 / denominator
    return 1 / (0.002 + sum_val)

def f15(x):
    a = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
    b = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    sum_val = 0
    for i in range(11):
        numerator = x[0] * (b[i]**2 + b[i] * x[1])
        denominator = b[i]**2 + b[i] * x[2] + x[3]
        sum_val += (a[i] - numerator / denominator)**2
    return sum_val


def use(test_num):
    DIMENSION = 30
    POPULATION_SIZE = 200
    GENERATIONS = 5000

    funcs = ["", f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15]
    bounds = ["", 
              [(-100, 100)], [(-100, 100)], [(-10, 10)], [(-100, 100)], [(-100, 100)],
              [(-30, 30)], [(-128, 128)], [(-500, 500)], [(-512, 512)], [(-32, 32)],
              [(-600, 600)], [(-50, 50)], [(-50, 50)], [(-65536, 65536)], [(-5, 5)]]
    bound = bounds[test_num] * DIMENSION
    
    if test_num == 14:
        DIMENSION = 2
    if test_num == 15:
        DIMENSION = 4
    
    print("EA Test Func No.", test_num)
    print(f"Population Size: {POPULATION_SIZE}, Gene Length: {DIMENSION}")
    
    ea = EvolutionaryAlgorithm(
        population_size=POPULATION_SIZE,
        gene_length=DIMENSION,
        bounds=bound,
        objective_function=funcs[test_num],
        mutation_rate=0.1,
        crossover_rate=0.8,
        selection_method="tournament",
        tournament_size=3
    )
    
    results = ea.run(generations=GENERATIONS, verbose=True)
    #ea.plot_convergence()    
    return results


def compare_parameters(test_num):
    DIMENSION = 30
    GENERATIONS = 500
    funcs = ["", f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15]
    bounds = ["", 
              [(-100, 100)], [(-100, 100)], [(-10, 10)], [(-100, 100)], [(-100, 100)],
              [(-30, 30)], [(-128, 128)], [(-500, 500)], [(-512, 512)], [(-32, 32)],
              [(-600, 600)], [(-50, 50)], [(-50, 50)], [(-65536, 65536)], [(-5, 5)]]
    bound = bounds[test_num] * DIMENSION
    if test_num == 14:
        DIMENSION = 2
    if test_num == 15:
        DIMENSION = 4
    population_sizes = [30, 50, 100]    
    plt.figure(figsize=(12, 8))    
    for pop_size in population_sizes:
        print(f"Test Population Sizes: {pop_size}")       
        ea = EvolutionaryAlgorithm(
            population_size=pop_size,
            gene_length=DIMENSION,
            bounds=bound,
            objective_function=funcs[test_num],
            mutation_rate=0.1,
            crossover_rate=0.9
        )        
        results = ea.run(generations=GENERATIONS, verbose=False)
        plt.plot(results['fitness_history'], label=f'Population Size = {pop_size}')    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Effect of Population Size on Convergence')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.show()


def process_data(result, test_num):
    best_fitness = result["best_fitness"]
    fitness_history = result["fitness_history"]
    idx = 0
    for i in range(len(fitness_history)):
        if fitness_history[i] == best_fitness:
            idx = i + 1
            break
    with open(f"../data/{test_num}.txt", 'a') as f:
        f.write(f"{best_fitness} {idx}\n")


if __name__ == "__main__":
    for test_num in range(1, 16):
        for _ in range(20):
            result = use(test_num)
            process_data(result, test_num)

