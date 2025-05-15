"""Instruction:
Evolve a Python function `solve` for online monotone submodular maximization
with a cardinality constraint `k`. Items arrive sequentially from a stream.
The goal is to select a subset `S` of `k` items that maximizes a given
submodular function `f(S)`. The specific submodular function `f` (e.g., based
on kernel methods or coverage) is provided via an object during evaluation.

The `solve` function is only called when |S|=k, for each arriving item, and must decide whether
to discard it or propose replacing an existing item.

`solve` Function Signature:
  - Args:
    - current_item (Any): The arriving item (e.g., np.ndarray features, Dict data).
    - current_solution_items (List[Any]): Items currently in the solution set S.
    - k (int): Cardinality constraint.
    - submodular_func (object): Provides methods like `peek_replace_gain(idx, item)` 
      to query hypothetical gains without changing the function's internal state.
  - Returns:
    - Tuple[int, Optional[int]]: (action, target_replace_index)
      - action: 0=Discard, 1=Replace.
      - target_replace_index: Index in `current_solution_items` to replace if action=1, else None.
Attention: DO NOT build functions to calculate gain! To get the gain, just call `submodular_func.peek_replace_gain(i, current_item)` for REPLACE item at index `i`, with current item.
Guidance:
- Base decisions only on the function arguments. No access to future items or full history.
- Use the `peek_` methods of `submodular_func` to evaluate potential changes.
- Aim for a policy that performs well on average across different `k`.
- Improve upon the simple baseline provided. Consider comparing gains, estimating item
  contributions, or other heuristics achievable via the `peek_` methods.
"""

import numpy as np
import math
import random
import time
from typing import List, Tuple, Optional, Dict, Any, Set
import blade

@blade.evolve
def solve(
    current_item: Any,
    current_solution_items: List[Any],
    k: int,
    submodular_func: Any
) -> Tuple[int, Optional[int]]:
    """
    Decides whether to discard (0), or propose replacing (1) the current item.
    Baseline: Replace if best swap gain positive.
    """
    current_size = len(current_solution_items)
    best_replace_gain = -float('inf')
    best_replace_idx = -1
    for i in range(k):
        try:
            current_replace_gain = submodular_func.peek_replace_gain(i, current_item)
            if current_replace_gain > best_replace_gain:
                best_replace_gain = current_replace_gain
                best_replace_idx = i
        except Exception as e:
            # print(f"Error during peek_replace_gain for index {i}: {e}. Skipping index.")
            continue # Skip this potential replacement if peek fails
    if best_replace_gain > 1e-9:
        if best_replace_idx == -1: return 0, None # Safety
        return 1, best_replace_idx
    else:
        return 0, None


# --- The Evaluator ---
@blade.run
def evaluate(instance_data) -> float:
    """
    Evaluates the 'solve' heuristic by comparing to offline greedy.
    """
    import math
    import random
    import time
    from typing import List, Tuple, Optional, Dict, Any, Set
    # --- Submodular Function Instantiation Helper ---
    def get_submodular_function(name: str, k: int, dimension: int, data_X: List[Any] = None):
        """Instantiates the correct submodular function based on name."""
        if name == 'GaussianKernel':
            return LogDetGaussianKernel(k=k, dimension=dimension, kappa=1.0)
        elif name == 'LaplacianKernel':
            return LogDetLaplacianKernel(k=k, dimension=dimension, kappa=10.0, sigma=1.0)
        elif name == 'TweetText':
            return TweetTextCoverage(k=k)
        else:
            raise ValueError(f"Unknown submodular function name: {name}")

    # --- Online Algorithm Simulator ---
    def run_online_algorithm(
        X: List[Any],
        k: int,
        solve_func: callable,
        submodular_func: Any
    ) -> float:
        """Simulates the online arrival process."""
        n_items = len(X)
        submodular_func.reset()
        for i in range(n_items):
            current_item = X[i]
            current_solution_copy = submodular_func.get_solution() # Pass copy
            # Wrap solve call in try-except as evolved code might fail
            try:
                current_size = len(current_solution_copy)
                if current_size < k:
                    try:
                        gain_add_only = submodular_func.peek_marginal_gain(current_item)
                    except Exception as e:
                        # print(f"Error during peek_marginal_gain: {e}. Discarding item.")
                        gain_add_only = -float('inf') # Treat error as very bad gain
                    if gain_add_only > 1e-9: 
                        action, target_replace_index = 2, None
                    else: 
                        action, target_replace_index = 0, None
                else:
                    action, target_replace_index = solve_func(
                        current_item=current_item,
                        current_solution_items=current_solution_copy,
                        k=k,
                        submodular_func=submodular_func
                    )
            except Exception as e:
                # print(f"Error executing solve function: {e}. Discarding item.")
                action, target_replace_index = 0, None # Default to discard on error
    
            current_actual_size = len(submodular_func.get_solution())
    
            # Wrap update calls in try-except as well
            try:
                if action == 2 and current_actual_size < k:
                    submodular_func.update(action_item=current_item, index_to_replace=None)
                elif action == 1 and current_actual_size == k and target_replace_index is not None:
                    if 0 <= target_replace_index < k:
                         submodular_func.update(action_item=current_item, index_to_replace=target_replace_index)
                # else: pass # Discard
            except Exception as e:
                 # print(f"Error during submodular_func.update: {e}. State might be inconsistent.")
                 # If update fails, the score might become unreliable. Maybe return error score?
                 # For now, just continue the simulation.
                 pass
    
        final_value = submodular_func.get_value()
        # Ensure final value is numeric, return low score if not
        if not isinstance(final_value, (int, float)) or not np.isfinite(final_value):
            print(f"Warning: Final submodular value is non-numeric or non-finite ({final_value}).")
            return -1e18
        return final_value

    # --- Helper Function: Offline Greedy Algorithm ---
    def offline_greedy_maximization(
        X: List[Any],
        k: int,
        submodular_func_instance: Any
        ) -> float:
        """Performs standard offline greedy algorithm."""
        n_items = len(X)
        if k <= 0: return 0.0
        k = min(k, n_items)
    
        submodular_func_instance.reset()
        selected_indices = set()
    
        for _ in range(k):
            best_gain = -float('inf')
            best_item_index = -1
            # Note: Iterating through all remaining items can be slow for large N.
            # Consider optimizations if this becomes a bottleneck (e.g. lazy greedy),
            # but standard greedy is fine for correctness baseline.
            for i in range(n_items):
                if i not in selected_indices:
                    marginal_gain = submodular_func_instance.peek_marginal_gain(X[i])
                    if marginal_gain > best_gain:
                        best_gain = marginal_gain
                        best_item_index = i
    
            if best_item_index != -1 and best_gain > -1e18: # Use very small threshold
                 submodular_func_instance.update(X[best_item_index], index_to_replace=None)
                 selected_indices.add(best_item_index)
            else: break
        return submodular_func_instance.get_value()

    # ======= submodular functions =======
    # --- Gaussian Kernel Log-Determinant ---
    class LogDetGaussianKernel:
        """
        Python implementation for f(S) = log(det(I + kappa * Ms)) with Gaussian kernel.
        Applicable to ForestCover, CreditCardFraud, KDDCup99.
        """
        def __init__(self, k: int, dimension: int, kappa: float = 1.0):
            if dimension <= 0:
                raise ValueError("Dimension must be positive.")
            # Parameter 'l' in C++ corresponds to sigma 'σ'
            # Paper uses sigma = d / (2*sqrt(d)) = sqrt(d) / 2 for kernel selection task
            self.sigma = math.sqrt(dimension) / 2.0 if dimension > 0 else 1.0
            self.kappa = kappa
            self.k = k
            self.dimension = dimension
            self._inv_sigma_sq = 1.0 / (self.sigma**2) if self.sigma > 1e-9 else float('inf')
            self.reset()
            # For identifying items if needed (optional, assuming indices are enough)
            # self._item_hashes = {} # Maps item hash to index in solution
    
        def reset(self):
            """Resets the internal state."""
            # K_S = Ms + (1/kappa)*I
            self._L = np.zeros((0, 0), dtype=np.float64) # Cholesky of K_S
            self._solution_items = [] # List of np.ndarray feature vectors
            self._current_f_val = 0.0
            self.query_count = 0
    
        def _gaussian_kernel(self, item1: np.ndarray, item2: np.ndarray) -> float:
            """Computes the Gaussian kernel value."""
            # Efficiently compute squared Euclidean distance
            # Avoid large intermediate arrays if possible, though np.dot is optimized
            diff = item1 - item2
            sq_dist = np.dot(diff, diff)
            # Handle potential overflow in exp for large distances / small sigma
            arg = -0.5 * sq_dist * self._inv_sigma_sq
            if arg < -700: # exp(-700) is close to 0
                 return 0.0
            return math.exp(arg)
    
        def _compute_kernel_vector(self, item: np.ndarray) -> np.ndarray:
            """Computes the vector K(item, S)."""
            if not self._solution_items:
                return np.array([], dtype=np.float64)
            # Vectorized computation is faster for larger solutions
            # Expand dims for broadcasting
            item_exp = np.expand_dims(item, axis=0) # Shape (1, D)
            solution_arr = np.array(self._solution_items) # Shape (current_size, D)
            diff = item_exp - solution_arr # Shape (current_size, D)
            sq_dist = np.sum(diff * diff, axis=1) # Shape (current_size,)
            # Handle exp argument limits
            args = -0.5 * sq_dist * self._inv_sigma_sq
            # Clip args to prevent overflow/underflow issues in exp
            args = np.clip(args, -700, 700)
            return np.exp(args) # k(item, S)
    
        def get_value(self) -> float:
            return self._current_f_val
    
        def get_solution(self) -> List[np.ndarray]:
            return list(self._solution_items)
    
        def peek_marginal_gain(self, item: np.ndarray) -> float:
            """Calculates f(S U {item}) - f(S) without changing state."""
            self.query_count += 1
            current_size = len(self._solution_items)
    
            if current_size == 0:
                return 0.5 * math.log(1.0 + self.kappa) # k(item,item) = 1
    
            k_vec = self._compute_kernel_vector(item) # k(item, S)
            k_ii = 1.0 # k(item, item)
    
            try:
                # Solve L*z = k_vec for z
                z = np.linalg.solve(self._L, k_vec)
            except np.linalg.LinAlgError:
                 # print("Warning: Singular L in peek_marginal_gain.")
                 return -1e9 # Indicate error
    
            kappa_z_sq = self.kappa * np.dot(z, z)
            # Target is (k_ii + 1/kappa) = 1 + 1/kappa
            d_squared_target = (k_ii + (1.0 / self.kappa)) - np.dot(z, z)
    
            if d_squared_target <= 1e-12:
                 # This implies adding the item would make the matrix nearly singular or non-PD
                 # The gain comes from log(d_squared_target * kappa) relative to log(1/kappa) term implicitly included
                 # print(f"Warning: d_squared_target {d_squared_target:.2e} <= 0 in peek_marginal_gain.")
                 return -1e9 # Effectively zero or negative gain
    
            # Gain = 0.5 * log( det(K_S U {i}) / det(K_S) )
            # det(K_S U {i}) = det(K_S) * d_squared_target
            # Gain = 0.5 * log(d_squared_target) + constant ? NO
            # Gain = f(S U {i}) - f(S)
            # f(S) = 0.5 * (k log k + log det(K_S))
            # f(S U {i}) = 0.5 * ((k+1) log k + log det(K_S U {i}))
            #            = 0.5 * ((k+1) log k + log(det(K_S) * d_squared_target))
            #            = 0.5 * ((k+1) log k + log det(K_S) + log d_squared_target)
            # Gain = 0.5 * (log kappa + log d_squared_target)
            marginal_gain = 0.5 * (math.log(self.kappa) + math.log(d_squared_target))
    
            if np.isnan(marginal_gain): return -1e9
            return max(-1e9, marginal_gain)
    
        def peek_replace_gain(self, index_to_remove: int, item_to_add: np.ndarray) -> float:
            r"""Calculates gain f((S \ {S[idx]}) U {item}) - f(S) without changing state."""
            self.query_count += 1 # Count as one complex query
    
            if not (0 <= index_to_remove < len(self._solution_items)):
                raise IndexError(f"Invalid index_to_remove: {index_to_remove}")
    
            # Simulate remove/add by rebuilding K_S' (less efficient but robust)
            temp_solution = [item for i, item in enumerate(self._solution_items) if i != index_to_remove]
            temp_solution.append(item_to_add)
            k_prime = len(temp_solution) # Should = k
    
            # Calculate f(S') using slogdet for stability
            if k_prime == 0:
                f_new = 0.0
            else:
                try:
                    K_S_prime_base = np.identity(k_prime)
                    M_S_prime = np.zeros((k_prime, k_prime))
                    for i in range(k_prime):
                        for j in range(i, k_prime):
                            kernel_val = self._gaussian_kernel(temp_solution[i], temp_solution[j])
                            M_S_prime[i, j] = kernel_val
                            if i != j: M_S_prime[j, i] = kernel_val
                    K_S_prime = M_S_prime + K_S_prime_base * (1.0 / self.kappa)
                    sign, logdet_K_S_prime = np.linalg.slogdet(K_S_prime)
    
                    if sign <= 0: f_new = -1e18
                    else: f_new = 0.5 * (k_prime * math.log(self.kappa) + logdet_K_S_prime)
                    if np.isnan(f_new) or np.isinf(f_new): f_new = -1e18
                except Exception as e: # Catch potential errors during calculation
                    # print(f"Error during peek_replace_gain K_S' calc: {e}")
                    # TODO: mis calling 
                    f_new = -1e18
    
            return f_new - self._current_f_val
    
        def update(self, action_item: np.ndarray, index_to_replace: Optional[int] = None) -> float:
            """Updates the solution set S and the internal state. Returns the new f(S) value."""
            current_size = len(self._solution_items)
    
            if index_to_replace is None: # Append
                if current_size >= self.k: raise ValueError("Solution full, cannot append.")
                # --- Efficient Cholesky Update ---
                k_vec = self._compute_kernel_vector(action_item)
                k_ii = 1.0
                if current_size == 0:
                    d_squared = k_ii + (1.0 / self.kappa)
                    if d_squared <= 1e-12: raise ValueError("Cannot add item, d^2 <= 0.")
                    self._L = np.array([[math.sqrt(d_squared)]], dtype=np.float64)
                else:
                    try: z = np.linalg.solve(self._L, k_vec)
                    except np.linalg.LinAlgError: return self._current_f_val # Failed update
                    d_squared = (k_ii + (1.0 / self.kappa)) - np.dot(z, z)
                    if d_squared <= 1e-12: return self._current_f_val # Failed update
                    d = math.sqrt(d_squared)
                    new_L = np.zeros((current_size + 1, current_size + 1), dtype=np.float64)
                    new_L[:current_size, :current_size] = self._L
                    new_L[current_size, :current_size] = z
                    new_L[current_size, current_size] = d
                    self._L = new_L
                self._solution_items.append(action_item)
            else: # Replace
                 if not (0 <= index_to_replace < current_size): raise IndexError("Invalid index.")
                 # --- Rebuild K_S and L (less efficient but robust) ---
                 self._solution_items[index_to_replace] = action_item
                 k_current = len(self._solution_items)
                 try:
                     K_S_base = np.identity(k_current)
                     M_S = np.zeros((k_current, k_current))
                     for i in range(k_current):
                         for j in range(i, k_current):
                             kernel_val = self._gaussian_kernel(self._solution_items[i], self._solution_items[j])
                             M_S[i,j] = kernel_val
                             if i != j: M_S[j,i] = kernel_val
                     K_S = M_S + K_S_base * (1.0 / self.kappa)
                     self._L = np.linalg.cholesky(K_S)
                 except np.linalg.LinAlgError:
                     raise RuntimeError("Matrix non-PD during replacement update.")
    
            # --- Update cached f(S) value ---
            current_k = len(self._solution_items)
            diag_L = np.diag(self._L)
            if np.any(diag_L <= 1e-12): self._current_f_val = -1e18
            else:
                logdet_K_S = 2 * np.sum(np.log(diag_L))
                self._current_f_val = 0.5 * (current_k * math.log(self.kappa) + logdet_K_S)
                if np.isnan(self._current_f_val) or np.isinf(self._current_f_val): self._current_f_val = -1e18
            return self._current_f_val
    
    # --- Laplacian Kernel Log-Determinant ---
    
    class LogDetLaplacianKernel(LogDetGaussianKernel): # Inherit basic structure
        """
        Python implementation for f(S) = log(det(I + kappa * Ms)) with Laplacian kernel.
        Applicable to YouTube dataset.
        """
        def __init__(self, k: int, dimension: int, kappa: float = 10.0, sigma: float = 1.0):
            # Override parent init
            if dimension <= 0: raise ValueError("Dimension must be positive.")
            self.sigma = sigma # Use fixed sigma=1 for Laplacian from C++
            self.kappa = kappa # Use fixed kappa=10 from C++
            self.k = k
            self.dimension = dimension
            self._inv_sigma = 1.0 / self.sigma if self.sigma > 1e-9 else float('inf')
            self.reset()
    
        def _laplacian_kernel(self, item1: np.ndarray, item2: np.ndarray) -> float:
            """Computes the Laplacian kernel value."""
            diff = item1 - item2
            # L1 norm (Manhattan distance) often used, but C++ used L2 distance in exp
            # Let's assume C++ meant exp(-||x-y||_2 / sigma) based on formula structure
            l2_dist = np.linalg.norm(diff)
            # arg = -l2_dist * self._inv_sigma
            arg = -l2_dist # C++ code used exp(-distance), sigma=1 implicitly? Let's match C++
            if arg < -700: return 0.0
            return math.exp(arg)
    
        def _compute_kernel_vector(self, item: np.ndarray) -> np.ndarray:
            """Computes the vector K(item, S) using Laplacian kernel."""
            if not self._solution_items:
                return np.array([], dtype=np.float64)
            # Vectorized computation might be complex for L2 norm then exp individually
            # Loop implementation matching C++ more closely:
            vec = np.zeros(len(self._solution_items), dtype=np.float64)
            for i, s_item in enumerate(self._solution_items):
                 vec[i] = self._laplacian_kernel(item, s_item)
            return vec # k(item, S)
    
        def peek_replace_gain(self, index_to_remove: int, item_to_add: np.ndarray) -> float:
            """Override to use Laplacian kernel in rebuild."""
            self.query_count += 1
            if not (0 <= index_to_remove < len(self._solution_items)): raise IndexError("Invalid index")
            temp_solution = [item for i, item in enumerate(self._solution_items) if i != index_to_remove]
            temp_solution.append(item_to_add)
            k_prime = len(temp_solution)
            if k_prime == 0: f_new = 0.0
            else:
                try:
                    K_S_prime_base = np.identity(k_prime)
                    M_S_prime = np.zeros((k_prime, k_prime))
                    for i in range(k_prime):
                        for j in range(i, k_prime):
                            # Use Laplacian kernel here
                            kernel_val = self._laplacian_kernel(temp_solution[i], temp_solution[j])
                            M_S_prime[i, j] = kernel_val
                            if i != j: M_S_prime[j, i] = kernel_val
                    K_S_prime = M_S_prime + K_S_prime_base * (1.0 / self.kappa)
                    sign, logdet_K_S_prime = np.linalg.slogdet(K_S_prime)
                    if sign <= 0: f_new = -1e18
                    else: f_new = 0.5 * (k_prime * math.log(self.kappa) + logdet_K_S_prime)
                    if np.isnan(f_new) or np.isinf(f_new): f_new = -1e18
                except Exception as e:
                    # print(f"Error during peek_replace_gain K_S' calc (Lap): {e}")
                    # TODO: mis calling in kernel [unsupported operand type(s) for -: 'float' and 'NoneType']
                    f_new = -1e18
            return f_new - self._current_f_val
    
        def update(self, action_item: np.ndarray, index_to_replace: Optional[int] = None) -> float:
            """Override to use Laplacian kernel in rebuild on replace."""
            current_size = len(self._solution_items)
            if index_to_replace is None: # Append - use parent method logic
                 super().update(action_item, index_to_replace)
            else: # Replace - rebuild using Laplacian
                 if not (0 <= index_to_replace < current_size): raise IndexError("Invalid index.")
                 self._solution_items[index_to_replace] = action_item
                 k_current = len(self._solution_items)
                 try:
                     K_S_base = np.identity(k_current)
                     M_S = np.zeros((k_current, k_current))
                     for i in range(k_current):
                         for j in range(i, k_current):
                             # Use Laplacian kernel here
                             kernel_val = self._laplacian_kernel(self._solution_items[i], self._solution_items[j])
                             M_S[i,j] = kernel_val
                             if i != j: M_S[j,i] = kernel_val
                     K_S = M_S + K_S_base * (1.0 / self.kappa)
                     self._L = np.linalg.cholesky(K_S)
                 except np.linalg.LinAlgError:
                     raise RuntimeError("Matrix non-PD during replacement update (Lap).")
                 # --- Update cached f(S) value (same formula) ---
                 diag_L = np.diag(self._L)
                 if np.any(diag_L <= 1e-12): self._current_f_val = -1e18
                 else:
                     logdet_K_S = 2 * np.sum(np.log(diag_L))
                     self._current_f_val = 0.5 * (k_current * math.log(self.kappa) + logdet_K_S)
                     if np.isnan(self._current_f_val) or np.isinf(self._current_f_val): self._current_f_val = -1e18
            return self._current_f_val
    
    
    # --- Tweet Text Coverage ---
    
    class TweetTextCoverage:
        """
        Python implementation for f(S) = sum_w sqrt(sum_{e in S} val(w, e)).
        Applicable to Twitter dataset.
        Expects items to be dicts with 'words': List[str] and 'retweets': float.
        """
        def __init__(self, k: int, dimension: int = 0, **kwargs): # Dimension unused
            self.k = k
            # Internal state: map of word -> total score (retweets) in solution
            self._word_scores_in_solution: Dict[str, float] = {}
            self._solution_items = [] # Store the original dicts
            self._current_f_val = 0.0
            self.query_count = 0
    
        def reset(self):
            self._word_scores_in_solution = {}
            self._solution_items = []
            self._current_f_val = 0.0
            self.query_count = 0
    
        def _calculate_f_from_scores(self, word_scores: Dict[str, float]) -> float:
            """Calculates f value from a word score dictionary."""
            if not word_scores:
                return 0.0
            total_sqrt_sum = sum(math.sqrt(score) for score in word_scores.values() if score > 0)
            return total_sqrt_sum
    
        def get_value(self) -> float:
            return self._current_f_val
    
        def get_solution(self) -> List[Dict[str, Any]]:
            return list(self._solution_items) # Return copy of list of dicts
    
        def peek_marginal_gain(self, item: Dict[str, Any]) -> float:
            """Calculates f(S U {item}) - f(S) without changing state."""
            self.query_count += 1
            if not item or 'words' not in item or 'retweets' not in item: return -1e9
            retweets = item['retweets']
            if retweets <= 0: return 0.0 # Adding item with no score adds no value
    
            # Calculate change in f value efficiently
            gain = 0.0
            for word in item['words']:
                current_score = self._word_scores_in_solution.get(word, 0.0)
                # Change = sqrt(current + new) - sqrt(current)
                gain += math.sqrt(current_score + retweets) - math.sqrt(current_score)
    
            return max(-1e9, gain) # Allow small negative FP errors
    
    
        def peek_replace_gain(self, index_to_remove: int, item_to_add: Dict[str, Any]) -> float:
            r"""Calculates gain f((S \ {S[idx]}) U {item}) - f(S) without changing state."""
            self.query_count += 1 # Count as one complex query
    
            if not (0 <= index_to_remove < len(self._solution_items)): raise IndexError("Invalid index")
            if not item_to_add or 'words' not in item_to_add or 'retweets' not in item_to_add: return -1e9
    
            item_being_removed = self._solution_items[index_to_remove]
            retweets_removed = item_being_removed.get('retweets', 0.0)
            retweets_added = item_to_add['retweets']
    
            # Calculate gain by summing changes for affected words
            gain = 0.0
            affected_words: Set[str] = set(item_being_removed.get('words', [])) | set(item_to_add['words'])
    
            for word in affected_words:
                current_score = self._word_scores_in_solution.get(word, 0.0)
                score_after_remove = max(0.0, current_score - retweets_removed) if word in item_being_removed.get('words', []) else current_score
                score_final = score_after_remove + retweets_added if word in item_to_add['words'] else score_after_remove
    
                # Change = sqrt(score_final) - sqrt(current_score)
                # Avoid sqrt(negative) if score becomes < 0 due to FP errors, though unlikely here
                sqrt_current = math.sqrt(current_score) if current_score > 0 else 0.0
                sqrt_final = math.sqrt(score_final) if score_final > 0 else 0.0
                gain += (sqrt_final - sqrt_current)
    
            return max(-1e9, gain)
    
    
        def update(self, action_item: Dict[str, Any], index_to_replace: Optional[int] = None) -> float:
            """Updates the solution set S and internal state. Returns the new f(S) value."""
            current_size = len(self._solution_items)
    
            # --- Add the new item's contribution ---
            new_retweets = action_item.get('retweets', 0.0)
            new_words = action_item.get('words', [])
            if new_retweets > 0:
                 for word in new_words:
                     self._word_scores_in_solution[word] = self._word_scores_in_solution.get(word, 0.0) + new_retweets
    
            if index_to_replace is None: # Append
                if current_size >= self.k: raise ValueError("Solution full.")
                self._solution_items.append(action_item)
            else: # Replace
                 if not (0 <= index_to_replace < current_size): raise IndexError("Invalid index.")
                 item_to_remove = self._solution_items[index_to_replace]
                 removed_retweets = item_to_remove.get('retweets', 0.0)
                 removed_words = item_to_remove.get('words', [])
    
                 # --- Remove the old item's contribution ---
                 if removed_retweets > 0:
                     for word in removed_words:
                         new_score = self._word_scores_in_solution.get(word, 0.0) - removed_retweets
                         if new_score <= 1e-9: # Remove entry if score becomes zero or negative
                             self._word_scores_in_solution.pop(word, None)
                         else:
                             self._word_scores_in_solution[word] = new_score
    
                 self._solution_items[index_to_replace] = action_item # Put new item in place
    
            # --- Recalculate total f(S) ---
            self._current_f_val = self._calculate_f_from_scores(self._word_scores_in_solution)
            return self._current_f_val

    # ======= START =======
    # KS_TO_EVALUATE = [10, 20, 30, 40, 50]
    KS_TO_EVALUATE = [10, 20] # speed
    all_k_ratios = []
    MAX_EVAL_ITEMS = 700    # speed
    total_start_time = time.time()

    try:
        # 'X' contains the features or tweet dicts. 'y' is not needed.
        instance_data = instance_data.item()
        # print(instance_data)
        X_full = instance_data['X']
        func_name = instance_data['submodular_func_name']
        n_items_full = len(X_full)

        if n_items_full == 0: return 0.0 # Return ratio of 0 if no data

        # --- Subsampling Logic ---
        if n_items_full > MAX_EVAL_ITEMS and func_name != 'TweetText':
            # print(f"Info: Subsampling {MAX_EVAL_ITEMS} items from {n_items_full} for evaluation.")
            # Ensure consistent sampling *within* an evaluation call, but random across calls
            # Use random.sample for lists, or np.random.choice for indices if X is array
            if isinstance(X_full, np.ndarray):
                 indices = np.random.choice(n_items_full, MAX_EVAL_ITEMS, replace=False)
                 X_eval = X_full[indices]
            elif isinstance(X_full, list):
                 # If X is list of dicts (like Twitter), sample the list
                 X_eval = random.sample(X_full, MAX_EVAL_ITEMS)
            else:
                 print("Warning: Unknown type for X_full, attempting list sampling.")
                 try:
                     X_eval = random.sample(list(X_full), MAX_EVAL_ITEMS)
                 except:
                     print("Error during sampling. Using full dataset (might be slow).")
                     X_eval = X_full # Fallback
            n_items_eval = len(X_eval)
        else:
            # Use the full dataset if it's small enough
            X_eval = X_full
            n_items_eval = n_items_full
        # --- End Subsampling Logic ---
        
        dimension = 0
        if n_items_eval > 0:
             if func_name in ['GaussianKernel', 'LaplacianKernel']:
                 if isinstance(X_eval[0], np.ndarray): dimension = X_eval[0].shape[0]
                 else: raise ValueError("Vector data expected for kernel functions")

        # print(f"Eval '{func_name}', N_eval={n_items_eval}{f', D={dimension}' if dimension>0 else ''}")

        for k in KS_TO_EVALUATE:
             k_start_time = time.time()
             # print(f"--- k = {k} ---") # Reduce verbosity
             current_k = min(k, n_items_eval)
             if current_k <= 0: continue

             online_f = get_submodular_function(func_name, current_k, dimension, X_eval)
             greedy_f = get_submodular_function(func_name, current_k, dimension, X_eval)

             f_val_online = run_online_algorithm(X_eval, current_k, solve, online_f)
             f_val_greedy = offline_greedy_maximization(X_eval, current_k, greedy_f)

             ratio = 0.0
             if f_val_greedy > 1e-9:
                  ratio = f_val_online / f_val_greedy
             elif f_val_online > 1e-9: # Online got >0, greedy got 0
                  ratio = 2.0 # Assign positive outcome
             else: # Both near zero
                  ratio = 1.0 # Achieved the same (zero)

             # Clamp ratio between 0 and (say) 2.0 (1.1) to handle potential float issues
             ratio = max(0.0, min(ratio, 2.0))
             all_k_ratios.append(ratio)
             # print(f"  k={k}: Online={f_val_online:.3f}, Greedy={f_val_greedy:.3f}, Ratio={ratio:.3f} ({(time.time()-k_start_time):.2f}s)")
             # print(f"  k={k}: Online={f_val_online}, Greedy={f_val_greedy}, Ratio={ratio} ({(time.time()-k_start_time):.2f}s)")

        if not all_k_ratios: average_ratio = 0.0
        else: average_ratio = np.mean(all_k_ratios)

        # print(f"Eval '{func_name}', Avg Ratio (k={KS_TO_EVALUATE}, N_eval={n_items_eval}): {average_ratio:.4f} (Total T: {(time.time()-total_start_time):.2f}s)")
        return average_ratio

    except KeyError as e:
        print(f"Error: Missing key '{e}'.")
        return 0.0 # Low score for setup errors
    except MemoryError:
       print(f"Error: MemoryError.")
       return 0.0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed debug
        return 0.0 # Low score for runtime errors

if __name__ == "__main__":
    pass