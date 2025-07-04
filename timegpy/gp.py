import random
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import zscore
from multiprocessing import Pool
from functools import partial
from .generate_random_tree import generate_random_tree
from .mutation import point_mutation
from .mutation import subtree_mutation
from .mutation import hoist_mutation
from .parsers import tree_to_feature_string
from .checks import ensure_has_X_t
from .crossover import crossover
from .evaluate_program_wrapper import evaluate_program

def evolve(
    X,
    y,
    pop_size=1000,
    n_generations=20,
    fitness_threshold=1.0,
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
    max_lag_terms=8,
    max_lag=8,
    max_exponent=5,
    const_range=(-1.0, 1.0),
    p_const=0.1,
    p_unary=0.1,
    unary_set=['sin', 'cos', 'tan'],
    seed=123,
    n_generation_improve=2,
    z_score=True,
    n_procs=1
):
    """
    Evolve a genetic programming algorithm to find informative time-average features for time-series classification.

    Args:
        X (array): ID by time matrix containing time-series data
        y (array): vector of class labels for each row of X
        pop_size (int): size of each population. Defaults to 1000
        n_generations (int): maximum number of generations. Defaults to 20
        fitness_threshold (float): objective function value which if equalled or exceeded, will terminate the algorithm. Defaults to 1.0 for no stopping
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
        max_lag_terms (int): maximum number of lag/constant terms allowed in a single feature expression. Defaults to 8
        max_lag (int): maximum time-lag allowed in a single feature expression. Defaults to 8
        max_exponent (int): maximum exponent allowed. Defaults to 5
        const_range (tuple): either a tuple of floats for the range of values a constant term can take, or None for no constants. Defaults to (-1.0, 1.0)
        p_const (int): probability of a given leaf node being a constant versus a time lag. Defaults to 0.1
        p_unary (float): probability of applying a unary trigonometric operator to a term. Defaults to 0.1
        unary_set (list): allowed unary operators. Defaults to ['sin', 'cos', 'tan']
        seed (int): fixes Python's random seed for reproducibility. Defaults to 123
        n_generation_improve (int): number of generations of no fitness improvement before algorithm terminates early. Defaults to 1
        z_score (bool): whether to z-score input data X. Defaults to True
        n_procs (int): number of processes to use if parallel processing is desired. Defaults to 1 for serial processing

    Returns:
        Data frame of fitness results for every generation and a data frame of the best individual time-average feature for easy identification
    """

    #------------- Check arguments -------------

    total_mutation_prob = p_point_mutation + p_subtree_mutation + p_hoist_mutation + p_crossover
    if total_mutation_prob >= 1.0:
        raise ValueError("The sum of mutation and crossover probabilities must be less than 1.")

    if not (0 < parsimony_coefficient < 1):
        raise ValueError("parsimony_coefficient must be between 0 and 1.")

    if not (0 < p_const < 1):
        raise ValueError("p_const must be between 0 and 1.")

    if not (0 < fitness_threshold <= 1):
        raise ValueError("fitness_threshold must be between 0 (exclusive) and 1 (inclusive).")

    if tournament_size >= pop_size:
        raise ValueError("tournament_size must be smaller than pop_size.")

    if unary_set is None or len(unary_set) == 0:
        p_unary = 0.0
    else:
        allowed_unary = {'sin', 'cos', 'tan'}
        if not set(unary_set).issubset(allowed_unary):
            raise ValueError("unary_set can only contain 'sin', 'cos', 'tan'")
        
    #------------- Define functions -------------

    def tournament_selection(pop, fitnesses, exclude_idx=None):
        indices = list(range(len(pop)))
        if exclude_idx is not None:
            indices.remove(exclude_idx)
        tournament_indices = random.sample(indices, min(tournament_size, len(indices)))
        selected = [(pop[i], fitnesses[i]) for i in tournament_indices if not np.isnan(fitnesses[i])]
        if not selected:
            return deepcopy(random.choice(pop))
        return deepcopy(max(selected, key=lambda x: x[1])[0])
    
    #------------- Run core algorithm -------------

    random.seed(seed)
    generation_data = []
    best_fitness = -np.inf
    no_improve_counter = 0

    if z_score:
        X = zscore(X, axis=1, nan_policy='omit')

    population = [
        generate_random_tree(
            max_lag_terms=max_lag_terms,
            prob_exponent=p_exponent,
            max_lag=max_lag,
            max_exponent=max_exponent,
            const_range=const_range,
            p_const=p_const,
            p_unary=p_unary,
            unary_set=unary_set,
            force_X_t=True
        ) for _ in range(pop_size)
    ]

    for gen in range(n_generations):
        print(f"Evolving generation {gen}")

        if n_procs > 1:
            with Pool(processes=n_procs) as pool:
                results = pool.map(partial(evaluate_program, X=X, y=y, z_score=False), population)
        else:
            results = [evaluate_program(prog, X, y, z_score=False) for prog in population]

        fitness_scores, program_sizes = zip(*results)
        #print(fitness_scores)

        if use_parsimony:
            clean_data = [(f, s, p) for f, s, p in zip(fitness_scores, program_sizes, population) if not np.isnan(f)]
            if not clean_data:
                break
            clean_fitness, clean_sizes, clean_programs = zip(*clean_data)
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
            population = list(clean_programs)
            program_sizes = list(clean_sizes)
            fitness_scores = list(clean_fitness)
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

        remaining_prob = 1.0 - (p_point_mutation + p_subtree_mutation + p_hoist_mutation + p_crossover)
        operations = ['point', 'subtree', 'hoist', 'crossover', 'clone']
        weights = [p_point_mutation, p_subtree_mutation, p_hoist_mutation, p_crossover, remaining_prob]

        new_population = []
        for idx in range(pop_size):
            parent1_idx = random.randint(0, len(population) - 1)
            parent1 = deepcopy(population[parent1_idx])
            operation = random.choices(operations, weights=weights)[0]

            if operation == 'point':
                child = point_mutation(parent1, max_lag=max_lag, max_exponent=max_exponent, const_range=const_range)
            elif operation == 'subtree':
                child = subtree_mutation(
                    parent1,
                    max_lag_terms=max_lag_terms,
                    prob_exponent=p_exponent,
                    max_lag=max_lag,
                    max_exponent=max_exponent,
                    const_range=const_range,
                    p_const=p_const,
                    p_unary=p_unary,
                    unary_set=unary_set
                )
            elif operation == 'hoist':
                child = hoist_mutation(parent1)
                if not ensure_has_X_t(child):
                    child = deepcopy(parent1)
            elif operation == 'crossover':
                parent2 = tournament_selection(population, fitness_parsimony, exclude_idx=parent1_idx)
                child = crossover(parent1, parent2)
                if not ensure_has_X_t(child):
                    child = deepcopy(parent1)
            else:
                child = deepcopy(parent1)

            new_population.append(child)

        population = new_population

    df_all = pd.DataFrame(generation_data)
    best_value = df_all['fitness_parsimony'].max() if use_parsimony else df_all['fitness'].max()
    best_candidates = df_all[df_all['fitness_parsimony' if use_parsimony else 'fitness'] == best_value]
    best_info = best_candidates.sort_values(by=['generation', 'individual']).iloc[[0]].copy()

    return df_all, best_info
