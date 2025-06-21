import random
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Distance matrix for cities A–H
cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
distance_matrix = {
    'A': [0, 20, 30, 25, 12, 33, 44, 57],
    'B': [22, 0, 19, 20, 20, 29, 43, 45],
    'C': [28, 19, 0, 17, 38, 48, 55, 60],
    'D': [25, 20, 19, 0, 28, 35, 40, 55],
    'E': [12, 18, 34, 25, 0, 21, 30, 40],
    'F': [35, 25, 45, 30, 20, 0, 25, 39],
    'G': [47, 39, 50, 35, 28, 20, 0, 28],
    'H': [60, 38, 54, 50, 33, 40, 25, 0],
}

def calculate_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        from_city = cities.index(route[i])
        to_city = cities.index(route[i + 1])
        distance += distance_matrix[route[i]][to_city]
    return distance

def fitness(route):
    return -calculate_distance(route)  # minimize distance, so use negative

def initialize_population(size):
    inner_cities = cities[1:]  # B–H
    population = []
    for _ in range(size):
        middle = random.sample(inner_cities, len(inner_cities))
        chromosome = ['A'] + middle + ['A']
        population.append(chromosome)
    return population

def roulette_select(population, fitnesses):
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choices(population, k=len(population))

    cum_probs = []
    cumulative = 0
    for fit in fitnesses:
        cumulative += fit
        cum_probs.append(cumulative)

    selected = []
    for _ in range(len(population)):
        r = random.uniform(0, total_fitness)
        for i, cp in enumerate(cum_probs):
            if r <= cp:
                selected.append(population[i])
                break
    return selected

def order_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(1, len(parent1) - 1), 2))
    middle1 = parent1[start:end]
    middle2 = parent2[start:end]

    def fill(child_middle, other_parent):
        child = ['A']
        child += child_middle
        rest = [city for city in other_parent if city not in child_middle and city != 'A']
        child[1:start] = rest[:start-1]
        child[end:-1] = rest[start-1:]
        child += ['A']
        return child

    return fill(middle1, parent2), fill(middle2, parent1)

def mutate(route, mutation_rate=0.25):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(1, len(route) - 1), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def ga_tsp(population_size=40, generations=40, crossover_prob=0.8, mutation_prob=0.25, threshold=0.05):
    population = initialize_population(population_size)
    logs = {
        "generations": [],
        "best_fitnesses_per_generation": [],
        "best_individuals_per_generation": [],
    }
    table = PrettyTable()
    table.field_names = ["Generation", "Best Route", "Best Fitness"]

    best_fit = None
    prev_fit = None

    for gen in range(generations):
        fitnesses = [fitness(p) for p in population]
        best_index = fitnesses.index(max(fitnesses))
        best_fit = fitnesses[best_index]
        best_route = population[best_index]

        logs["generations"].append(gen)
        logs["best_fitnesses_per_generation"].append(best_fit)
        logs["best_individuals_per_generation"].append(best_route)
        table.add_row([gen, best_route, round(-best_fit, 2)])  # distance is -fitness

        if prev_fit is not None and abs(best_fit - prev_fit) <= threshold:
            break
        prev_fit = best_fit

        selected = roulette_select(population, [f - min(fitnesses) + 1 for f in fitnesses])  # normalize

        next_gen = []
        while len(next_gen) < population_size:
            p1, p2 = random.sample(selected, 2)
            if random.random() < crossover_prob:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            c1 = mutate(c1, mutation_prob)
            c2 = mutate(c2, mutation_prob)
            next_gen.extend([c1, c2])

        population = next_gen[:population_size]

    # Final result
    print("Optimal route:", logs["best_individuals_per_generation"][-1])
    print("Minimum distance:", -logs["best_fitnesses_per_generation"][-1])
    print(table)

    plt.figure(figsize=(10, 6))
    plt.plot(logs["generations"], [-f for f in logs["best_fitnesses_per_generation"]], marker='o')
    plt.title(f'GA TSP: Best Distance vs Generation (Pop={population_size})')
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'tsp_output_pop{population_size}.png')
    plt.close()

    return logs

ga_tsp(population_size=60)