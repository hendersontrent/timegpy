import random
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import zscore
from .generate_random_tree import generate_random_tree
from .evaluate_tree import evaluate_tree
from .mutation import point_mutation
from .mutation import subtree_mutation
from .mutation import hoist_mutation
from .fitness import compute_eta_squared
from .calculate_program_size import calculate_program_size
from .parsers import tree_to_feature_string
from .checks import ensure_has_X_t
from .crossover import crossover

def tsgp(
    X,
    y,
    pop_size=100,
    n_generations=5,
    fitness_threshold=0.95,
    p_point_mutation=0.01,
    p_subtree_mutation=0.01,
    p_hoist_mutation=0.01,
    p_crossover=0.9,
    p_exponent=0.3,
    tournament_size=20,
    use_parsimony=True,
    auto_parsimony=True,
    parsimony_coefficient=0.001,
    verbose=False,
    max_depth=8,
    max_lag=8,
    max_exponent=5,
    seed=123,
    n_generation_improve=1,
    z_score=True
):
    
    """
    Evolve a genetic programming algorithm to find informative time-average features for time-series classification.
    Args:
        X (array): ID by time matrix containing time-series data
        y (array): vector of class labels for each row of X
        pop_size (int): size of each population. Defaults to 100
        n_generations (int): maximum number of generations. Defaults to 5
        fitness_threshold (float): objective function value which if equalled or exceeded, will terminate the algorithm. Defaults to 0.95
        p_point_mutation (float): probability of point mutation occurring. Defaults to 0.01
        p_subtree_mutation (float): probability of subtree mutation occurring. Defaults to 0.01
        p_hoist_mutation (float): probability of hoist mutation occurring. Defaults to 0.01
        p_crossover (float): probability of crossover occurring. Defaults to 0.9
        p_exponent (float): probability of a time lag being exponentiated. Defaults to 0.3
        tournament_size (int): size of each tournament to find a suitable parent. Defaults to 20
        use_parsimony (bool): whether to use parsimony-adjusted fitness instead of raw fitness. Defaults to True
        auto_parsimony (bool): whether to calculate generational parsimony coefficients dynamically. Defaults to True
        parsimony_coefficient (float): if auto_parsimony = False, this static coefficient for parsimony will be applied to all generations. Defaults to 0.001
        verbose (bool): whether to print updates of algorithm progress. Defaults to False
        max_depth (int): maximum number of time-lag terms allowed in a single feature expression. Defaults to 8
        max_lag (int): maximum time-lag allowed in a single feature expression. Defaults to 8
        max_exponent (int): maximum exponent allowed. Defaults to 5
        seed (int): fixes Python's random seed for reproducibility. Defaults to 123
        n_generation_improve (int): number of generations of no fitness improvement before algorithm terminates early. Defaults to 1
        z_score (bool): whether to z-score input data X. Defaults to True

    Returns:
        Input data X, class label vector y, fitness results for every generation, and best individual time-average feature.
    """

    def tournament_selection(pop, fitnesses):
        selected = random.sample(list(zip(pop, fitnesses)), tournament_size)
        selected = [item for item in selected if not np.isnan(item[1])]
        if not selected:
            return deepcopy(random.choice(pop))
        return deepcopy(max(selected, key=lambda x: x[1])[0])

    random.seed(seed)
    generation_data = []
    best_fitness = -np.inf
    no_improve_counter = 0

    if z_score:
        X = zscore(X, axis=1, nan_policy='omit')

    population = [generate_random_tree(max_depth, p_exponent, max_lag, max_exponent) for _ in range(pop_size)]

    for gen in range(n_generations):
        fitness_scores = []
        program_sizes = []

        for i, program in enumerate(population):
            feature = evaluate_tree(program, X)
            if feature is None or np.isnan(feature).all():
                fitness = np.nan
            else:
                try:
                    fitness = compute_eta_squared(feature, y)
                except:
                    fitness = np.nan

            size = calculate_program_size(program)
            fitness_scores.append(fitness)
            program_sizes.append(size)

        if use_parsimony:
            clean_fitness, clean_sizes, clean_programs = [], [], []
            for f, s, p in zip(fitness_scores, program_sizes, population):
                if not np.isnan(f):
                    clean_fitness.append(f)
                    clean_sizes.append(s)
                    clean_programs.append(p)

            fitness_np = np.array(clean_fitness, dtype=float)
            sizes_np = np.array(clean_sizes, dtype=float)

            if auto_parsimony and len(sizes_np) > 1 and np.var(sizes_np) > 0:
                try:
                    cov = np.cov(fitness_np, sizes_np)[0, 1]
                    parsimony = abs(cov / np.var(sizes_np))
                except:
                    parsimony = parsimony_coefficient
            else:
                parsimony = parsimony_coefficient

            fitness_parsimony = fitness_np - parsimony * sizes_np
            population = clean_programs
            program_sizes = clean_sizes
            fitness_scores = clean_fitness
        else:
            fitness_parsimony = fitness_scores

        for idx, (prog, fit, fit_adj, size) in enumerate(zip(population, fitness_scores, fitness_parsimony, program_sizes)):
            generation_data.append({
                'generation': gen,
                'individual': idx,
                'expression': tree_to_feature_string(prog),
                'program_size': size,
                'fitness': fit,
                'fitness_parsimony': fit_adj
            })

        # Identify best of current generation
        gen_best_idx = np.nanargmax(fitness_parsimony)
        gen_best_fit = fitness_parsimony[gen_best_idx]

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if verbose:
            print(f"Generation {gen}: best fitness (parsimony): {best_fitness:.4f}")
        if best_fitness >= fitness_threshold:
            if verbose:
                print("Stopping early: fitness threshold reached.")
            break
        if no_improve_counter >= n_generation_improve:
            if verbose:
                print("Stopping early: no improvement in best fitness.")
            break

        # Generate next generation
        remaining_prob = 1.0 - (p_point_mutation + p_subtree_mutation + p_hoist_mutation + p_crossover)
        operations = ['point', 'subtree', 'hoist', 'crossover', 'clone']
        weights = [p_point_mutation, p_subtree_mutation, p_hoist_mutation, p_crossover, remaining_prob]

        new_population = []
        for _ in range(pop_size):
            parent1 = tournament_selection(population, fitness_parsimony)
            operation = random.choices(operations, weights=weights)[0]

            if operation == 'point':
                child = point_mutation(parent1, max_lag, max_exponent)
            elif operation == 'subtree':
                child = subtree_mutation(parent1, max_depth, p_exponent, max_lag, max_exponent)
            elif operation == 'hoist':
                child = hoist_mutation(parent1)
                if not ensure_has_X_t(child):
                    child = deepcopy(parent1)
            elif operation == 'crossover':
                parent2 = tournament_selection(population, fitness_parsimony)
                child = crossover(parent1, parent2)
                if not ensure_has_X_t(child):
                    child = deepcopy(parent1)
            else:  # 'clone'
                child = deepcopy(parent1)

            new_population.append(child)

        population = new_population

    df_all = pd.DataFrame(generation_data)

    # Resolve ties by earliest appearance
    best_value = df_all['fitness_parsimony'].max() if use_parsimony else df_all['fitness'].max()
    best_candidates = df_all[df_all['fitness_parsimony' if use_parsimony else 'fitness'] == best_value]
    best_info = best_candidates.sort_values(by=['generation', 'individual']).iloc[[0]].copy()

    return X, y, df_all, best_info