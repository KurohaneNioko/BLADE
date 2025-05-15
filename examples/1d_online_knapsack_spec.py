"""Evolves a heuristic for the 1D Online Knapsack Problem.

Instruction:
- Improve the `solve` function iteratively based on previous versions.
- Make small, logical changes, focusing on effectively using the provided item
  and historical information (including lists of previously accepted items).
- Aim for concise and efficient Python code.
- The `solve` function should return 1 to accept the item, 0 to reject,
  assuming the item fits the knapsack's remaining capacity. Consider if using
  aggregate information derived from history lists might be simpler or more
  efficient than iterating over full lists in some cases.

`solve` Function Quick Reference:
  - Args: item_weight(float), item_profit(float), remaining_capacity(float),
          total_capacity(float), accepted_items_weights(list[float]),
          accepted_items_profits(list[float]), num_items_seen(int)
  - Returns: int (1 to accept, 0 to reject)
  - Assumption: Called only if the item fits remaining capacity. Focus on desirability.
"""

import numpy as np
import blade

# --- The Function to Evolve ---
@blade.evolve
def solve(
    item_weight: float,
    item_profit: float,
    remaining_capacity: float,
    total_capacity: float,
    accepted_items_weights: list[float],
    accepted_items_profits: list[float],
    num_items_seen: int
) -> int:
    """Decides whether to accept the current item based on desirability.
  
    This function is called ONLY IF the current item fits in the knapsack
    (i.e., item_weight <= remaining_capacity).
  
    Args:
      item_weight: Weight of the current item arriving.
      item_profit: Profit of the current item arriving.
      remaining_capacity: Current available capacity in the knapsack BEFORE
                        considering this item.
      total_capacity: The original total capacity of the knapsack.
      accepted_items_weights: A list of the weights of items already accepted
                              into the knapsack, in acceptance order.
      accepted_items_profits: A list of the profits of items already accepted
                              into the knapsack, in acceptance order.
      num_items_seen: The total number of items encountered *before* this one
                    (0-indexed count of items processed).
  
    Returns:
      int: 1 if the item should be accepted into the knapsack based on its
           desirability and the current context/history, 0 otherwise.
    """
    # Baseline heuristic: Accept if the item has positive profit density.
    # This version ignores history lists for simplicity as a starting point.
    # It assumes the fit check was done externally.
  
    # Handle division by zero for weightless items.
    if item_weight > 1e-6:
        density = item_profit / item_weight
    elif item_profit > 0:
        density = 1e18 # Treat as very high density if profit > 0 and weight is ~0
    else:
        density = 0.0
  
    # Simple decision rule for the baseline: accept if density > 0
    if density > 0:
        return 1
    else:
        return 0

# --- The Solver (Simulates Online Process using the evolved solve) ---
# This part is needed for evaluation but is NOT part of the code evolved by the LLM.
def solve(capacity: float, weights: np.ndarray, profits: np.ndarray) -> float:
    """
    Simulates the online knapsack filling process using the evolved solve function.
  
    Args:
      capacity: Total knapsack capacity.
      weights: Numpy array of item weights in arrival order.
      profits: Numpy array of item profits in arrival order.
  
    Returns:
      Total profit achieved by the heuristic solve function.
    """
    remaining_capacity = float(capacity)
    total_profit = 0.0
    accepted_items_weights = []
    accepted_items_profits = []
    num_items = len(weights)
  
    weights_np = np.asarray(weights, dtype=float)
    profits_np = np.asarray(profits, dtype=float)
  
    for i in range(num_items):
        w_i = weights_np[i]
        p_i = profits_np[i]
        num_items_seen = i
    
        if w_i <= remaining_capacity + 1e-9:
            decision = solve(
                w_i,
                p_i,
                remaining_capacity,
                capacity,
                accepted_items_weights,
                accepted_items_profits,
                num_items_seen
            )
            if decision == 1:
               total_profit += p_i
               remaining_capacity -= w_i
               accepted_items_weights.append(w_i)
               accepted_items_profits.append(p_i)
  
    return total_profit

# --- The Evaluator (Receives data dict, calls solve) ---
# This part is needed for evaluation but is NOT part of the code evolved by the LLM.
@blade.run
def evaluate(instance_data: dict[str, any]) -> float:
    """
    Evaluates the current 'solve' heuristic on a single knapsack instance
    by averaging its performance over multiple random shuffles of the item order.
  
    Args:
      instance_data: A dictionary containing 'capacity', 'weights' (list/np.ndarray),
                     'profits' (list/np.ndarray), and 'optimal_profit'.
  
    Returns:
      The average ratio of the profit achieved by the heuristic to the known
      optimal profit, averaged over several random shuffles of the item order.
      Returns 0.0 on error or invalid data.
    """
    import itertools, math
    N_THRESHOLD_FOR_PERMUTATIONS = 10
    MAX_SAMPLES_WHEN_SHUFFLING = 3628800 # 10! Number of random shuffles to average over per instance. Adjust as needed.
                     # A value between 10-25 is often a reasonable trade-off.

    try:
        capacity = float(instance_data['capacity'])
        # Ensure we work with numpy arrays and keep the originals safe
        original_weights = np.array(instance_data['weights'], dtype=float)
        original_profits = np.array(instance_data['profits'], dtype=float)
        # We don't need optimal_selection here, only the optimal profit value
        optimal_profit = float(instance_data['optimal_profit'])
    
        # --- Basic Input Validation ---
        if len(original_weights) != len(original_profits):
            print("Error: Mismatch between number of weights and profits.")
            return 0.0
        if capacity < 0:
            print(f"Warning: Negative capacity encountered: {capacity}.")
            # Depending on problem definition, negative capacity might be invalid
            return 0.0
        # Handle non-profitable instances or zero optimal profit to avoid division by zero
        if optimal_profit <= 1e-9:
            # If the best possible profit is zero (or negative), any non-negative
            # heuristic profit achieves the "optimal" ratio of 1.0 (or more, capped at 1.0).
            # However, running solve might still be needed if negative profits are possible
            # and the heuristic could achieve negative profit. Let's run solve once.
            if len(original_weights) == 0: return 1.0 # Empty instance is trivially optimal
            heuristic_profit = solve(capacity, original_weights, original_profits)
            return 1.0 if heuristic_profit >= -1e9 else 0.0 # Return 1 if non-negative profit achieved
    
        num_items = len(original_weights)
        if num_items == 0:
            # Empty instance case already handled by optimal_profit check, but good to be explicit
            return 1.0
    
        # --- Shuffling and Evaluation Loop ---
        ratios = []
        indices = np.arange(num_items) # Indices to shuffle [0, 1, ..., n-1]
        if num_items <= N_THRESHOLD_FOR_PERMUTATIONS:
            # --- Evaluate All Permutations ---
            num_perms_actual = math.factorial(num_items) # Calculate actual number for logging
            for current_indices_tuple in itertools.permutations(indices):
                permuted_weights = original_weights[list(current_indices_tuple)]
                permuted_profits = original_profits[list(current_indices_tuple)]
    
                heuristic_profit = solve(capacity, permuted_weights, permuted_profits)
                ratio = heuristic_profit / optimal_profit
                ratios.append(max(0.0, ratio))
    
        else:
            # --- Fallback to Random Sampling (up to 10! samples) ---
            # Determine the actual number of samples needed: min(n!, MAX_SAMPLES_WHEN_SHUFFLING)
            # However, since n > 10, n! will always be > 10!, so we just use MAX_SAMPLES_WHEN_SHUFFLING
            num_samples_to_run = MAX_SAMPLES_WHEN_SHUFFLING
            for _ in range(num_samples_to_run):
                np.random.shuffle(indices) # Shuffle in-place
                shuffled_weights = original_weights[indices]
                shuffled_profits = original_profits[indices]
    
                heuristic_profit = solve(capacity, shuffled_weights, shuffled_profits)
                ratio = heuristic_profit / optimal_profit
                ratios.append(max(0.0, ratio))
    
        # --- Aggregate Results ---
        if not ratios: # Should only happen if NUM_SHUFFLES is 0
            return 0.0
    
        average_ratio = np.mean(ratios)
        return average_ratio

    except KeyError as e:
        print(f"Error: Missing key '{e}' in instance_data dictionary.")
        return 0.0
    except Exception as e:
        # Catch other potential errors during processing/solving
        print(f"Error during evaluation with instance data: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return 0.0