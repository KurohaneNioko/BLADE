# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A codes database that implements the evolutionary algorithm."""
import pathlib
import pickle
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
import select
import random
from typing import Any, Iterable, Tuple, List
import multiprocessing as mp
from multiprocessing import Queue, Pipe
from multiprocessing.connection import Connection 

from absl import logging
import numpy as np
import scipy

from blade import code_manipulation
from blade import config as config_lib

Signature = tuple[float, ...]
ScoresPerTest = List[float] #Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
  """Returns the tempered softmax of 1D finite `logits`."""
  if not np.all(np.isfinite(logits)):
    non_finites = set(logits[~np.isfinite(logits)])
    raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
  if not np.issubdtype(logits.dtype, np.floating):
    logits = np.array(logits, dtype=np.float32)
  logits += 1e-9

  result = scipy.special.softmax(logits / temperature, axis=-1)
  # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
  index = np.argmax(result)
  result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
  return result

def _scores_to_distribution(scores: np.ndarray | list[float],
                            temperature: float = 1.0,
                            smoothing_epsilon: float = 1e-9) -> np.ndarray:
  """Converts a vector of scores into a probability distribution using softmax."""
  scores_np = np.array(scores, dtype=np.float32)
  if scores_np.size == 0: # Handle empty scores
    return np.array([], dtype=np.float32)
  # Use softmax for normalization (handles negative scores, emphasizes larger ones)
  dist = _softmax(scores_np, temperature=temperature)
  # Add epsilon smoothing to prevent zeros for KL calculation
  dist += smoothing_epsilon
  dist /= np.sum(dist) # Re-normalize
  return dist

# Helper function for KL divergence
def _calculate_kl_divergence(dist_p: np.ndarray, dist_q: np.ndarray) -> float:
  """Calculates KL divergence D_KL(P || Q).
  Assumes P and Q are probability distributions (non-negative, sum to 1).
  Uses scipy.stats.entropy.
  """
  if dist_p.size == 0 or dist_q.size == 0 or dist_p.shape != dist_q.shape:
    # Return 0 divergence if inputs are invalid/mismatched (no basis for comparison)
    # print("Warning: Invalid inputs for KL divergence calculation.")
    return 0.0  
  # Ensure inputs are valid probability distributions
  if not np.all(dist_p >= 0) or not np.all(dist_q >= 0):
    # print("Warning: KL divergence inputs contain negative values.")
    return 0.0 # Or handle differently? Should not happen with softmax/smoothing.
  # if not np.isclose(np.sum(dist_p), 1.0) or not np.isclose(np.sum(dist_q), 1.0):
    # print(f"Warning: KL divergence inputs do not sum to 1. Sums: {np.sum(dist_p)}, {np.sum(dist_q)}")
    # Attempt to re-normalize defensively, though this indicates an upstream issue
  dist_p += 1e-9
  dist_q += 1e-9
  dist_p /= np.sum(dist_p)
  dist_q /= np.sum(dist_q)
  # scipy.stats.entropy calculates D_KL(pk || qk)
  kl_div = scipy.stats.entropy(dist_p, dist_q)
  # print("KL: ", dist_p, dist_q, kl_div)

  # Handle potential inf result if q has zeros where p doesn't (should be mitigated by smoothing)
  if np.isinf(kl_div):
    # print("Warning: KL divergence is infinite. Capping.")
    return 10.0 # Arbitrary large number, adjust as needed
  # Handle potential NaN results (e.g. if inputs were somehow invalid despite checks)
  if np.isnan(kl_div):
    # print("Warning: KL divergence resulted in NaN. Returning 0.")
    return 0.0
  return kl_div

def _reduce_score(scores_per_test: ScoresPerTest) -> float:
  """Reduces per-test scores into a single score.
  Used before register a code. Now store multiple score for each code,
  and reduced score is the sum of scores from all tests.
  """
  return np.sum(scores_per_test)
  # return scores_per_test[list(scores_per_test.keys())[-1]]


def _get_signature(scores_per_test: ScoresPerTest, precision=5) -> Signature:
  """Represents test scores as a canonical signature."""
  if type(scores_per_test[0]) == float:
    return f"{np.mean(scores_per_test):.{precision}f}"
    # return tuple(list(map(lambda x: f"{x:.4f}", scores_per_test)))
  else:
    return tuple(scores_per_test)
  # return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
  """A prompt produced by the CodesDatabase, to be sent to Generators.

  Attributes:
    code: The prompt, ending with the header of the function to be completed.
    version_generated: The function to be completed is `_v{version_generated}`.
    island_id: Identifier of the island that produced the implementations
       included in the prompt. Used to direct the newly generated implementation
       into the same island.
  """
  code: str
  version_generated: int
  island_id: int


class CodesDatabase(mp.Process):
  """A collection of codes, organized as islands."""

  def __init__(
      self,
      config: config_lib.CodesDatabaseConfig,
      template: code_manipulation.Code,
      function_to_evolve: str,
      # for multiproc
      save_code_queue: Queue,
      send_prompt_pipes: List[Connection],
      log_path,
      # for KL divergence
      identifier: str = "",
      signature_precision: int = 5
  ) -> None:
    super().__init__()
    self._db_config: config_lib.CodesDatabaseConfig = config
    self._template: code_manipulation.Code = template
    self._function_to_evolve: str = function_to_evolve
    self._log_path = log_path
    # multiprocess
    self._save_code_queue: Queue = save_code_queue
    self._send_prompt_pipes: List[Connection] = send_prompt_pipes
    self._exit_flag = mp.Event()

    self._signature_precision = signature_precision
    # Initialize empty islands.
    self._islands: list[Island] = []
    for _ in range(config.num_islands):
      self._islands.append(
          Island(template, function_to_evolve, config.functions_per_prompt,
                 config.cluster_selecting_temperature_init,
                 config.cluster_selecting_temperature_period,
                 config.kl_divergence_weight,
                 self._signature_precision
                ))
    self._best_score_per_island: list[float] = (
        [-float('inf')] * config.num_islands)
    self._best_code_per_island: list[code_manipulation.Function | None] = (
        [None] * config.num_islands)
    self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
        [None] * config.num_islands)

    self._last_reset_time: float = time.time()
    self._code_counter = 0
    self._backups_done = 0
    self.identifier = identifier

    # flag indicate to reset islands
    # 0 means no reset,
    # 1: first record the score threshold (less/leq will be reset)
    #    now fetch prompts from lower half score, then code changed to 2
    # 2: mutation is going
    # when prompts run out, reset using mark
    self._reset_flag = 0
    self._last_n_chance_before_reset = config.last_n_chance_before_reset

  def get_best_codes_per_island(self) -> Iterable[Tuple[code_manipulation.Function | None]]:
    return sorted(zip(self._best_code_per_island, self._best_score_per_island), key=lambda t: t[1], reverse=True)

  def save(self, file):
    """Save database to a file"""
    data = {}
    keys = ["_islands", "_best_score_per_island", "_best_code_per_island", "_best_scores_per_test_per_island"]
    for key in keys:
      data[key] = getattr(self, key)
    pickle.dump(data, file)

  def load(self, file):
    """Load previously saved database"""
    data = pickle.load(file)
    for key in data.keys():
      setattr(self, key, data[key])

  def backup(self):
    filename = f"code_db_{self._function_to_evolve}_{self.identifier}_{self._backups_done}.pickle"
    # p = pathlib.Path(self._db_config.backup_folder)
    p = pathlib.Path(self._log_path) / "backup"
    if not p.exists():
      p.mkdir(parents=True, exist_ok=True)
    filepath = p / filename
    logging.info(f"Saving backup to {filepath}.")

    with open(filepath, mode="wb") as f:
      self.save(f)
    self._backups_done += 1

  def get_prompt(self, island_id=None, override_function_num=None) -> Prompt:
    """Returns a prompt containing implementations from one chosen island."""
    if island_id is None:
      island_id = np.random.randint(len(self._islands))
    code, version_generated = self._islands[island_id].get_prompt(use_kl_div=self._db_config.kl_divergence_weight>0, override_function_num=override_function_num)
    return Prompt(code, version_generated, island_id)

  def _register_code_in_island(
      self,
      code: code_manipulation.Function,
      island_id: int,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `code` in the specified island."""
    self._islands[island_id].register_code(code, scores_per_test)
    score = _reduce_score(scores_per_test)
    if score > self._best_score_per_island[island_id]:
      self._best_code_per_island[island_id] = code
      self._best_scores_per_test_per_island[island_id] = copy.deepcopy(scores_per_test)
      self._best_score_per_island[island_id] = score
      logging.info('Best score of island %d increased to %s', island_id, score)

  def register_code(
      self,
      code: code_manipulation.Function,
      island_id: int | None,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `code` in the database."""
    # In an asynchronous implementation we should consider the possibility of
    # registering a code on an island that had been reset after the prompt
    # was generated. Leaving that out here for simplicity.
    if island_id is None:
      # This is a code added at the beginning, so adding it to all islands.
      for island_id in range(len(self._islands)):
        self._register_code_in_island(code, island_id, scores_per_test)
    else:
      self._register_code_in_island(code, island_id, scores_per_test)

    # Check whether it is time to reset an island.
    if (time.time() - self._last_reset_time > self._db_config.reset_period) and self._reset_flag == 0:
      # self._last_reset_time = time.time()
      # self.reset_islands()
      self._reset_flag = 1

    # Backup every N iterations
    if self._code_counter >= 0:
      self._code_counter += 1
      if self._code_counter > self._db_config.backup_period:
        self._code_counter = 0
        self.backup()

  def get_keep_and_reset_island_ids(self):
    """Resets the weaker half of islands."""
    # We sort best scores after adding minor noise to break ties.
    indices_sorted_by_score: np.ndarray = np.argsort(
        self._best_score_per_island +
        np.random.randn(len(self._best_score_per_island)) * 1e-6)
    num_islands_to_reset = self._db_config.num_islands // 2
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    return keep_islands_ids, reset_islands_ids
    
  def reset_islands(self, score_threshold, keep_islands_ids) -> None:
    """Reset based on the threshold"""
    for island_id in range(len(self._islands)):
      if self._best_score_per_island[island_id] >= score_threshold:
        continue
      self._islands[island_id] = Island(
          self._template,
          self._function_to_evolve,
          self._db_config.functions_per_prompt,
          self._db_config.cluster_selecting_temperature_init,
          self._db_config.cluster_selecting_temperature_period,
          self._db_config.kl_divergence_weight,
          self._signature_precision
      )
      self._best_score_per_island[island_id] = -float('inf')
      founder_island_id = np.random.choice(keep_islands_ids)
      founder = self._best_code_per_island[founder_island_id]
      founder_scores = self._best_scores_per_test_per_island[founder_island_id]
      self._register_code_in_island(founder, island_id, founder_scores)
    self._last_reset_time = time.time()
    self._reset_flag = 0
    logging.info("Database reset!")
  
  def stop(self):
    self._exit_flag.set()

  def run(self):
    """Run DB in multiprocessing"""
    keep_islands_ids, reset_islands_ids = None, None 
    keep_island_score_threshold = None
    prompts_for_mutation = None
    while not self._exit_flag.is_set():
      # finish saving tasks first
      while not self._save_code_queue.empty(): 
        code, island_id, scores_per_test = self._save_code_queue.get()
        self.register_code(code, island_id, scores_per_test)
      
      # check reset_flag!
      if self._reset_flag == 1:
        keep_islands_ids, reset_islands_ids = self.get_keep_and_reset_island_ids()
        keep_island_score_threshold = self._best_score_per_island[keep_islands_ids[-1]]
        prompts_for_mutation = []
        for _ in range(self._last_n_chance_before_reset):
          for island_id in reset_islands_ids:
            prompts_for_mutation.append(self.get_prompt(island_id, override_function_num=1))
        self._reset_flag = 2
      
      # now generate prompts
      try:
        readable, _, _ = select.select(self._send_prompt_pipes, [], [], 0.1)
        for pipe in readable:
          if pipe.recv()  == "p": # Generator send "p" to fetch prompt
            if self._reset_flag == 2:
              if prompts_for_mutation:  # 确保列表不为空
                prompts_for_mutation_idx = random.randrange(len(prompts_for_mutation))
                current_prompt = prompts_for_mutation.pop(prompts_for_mutation_idx)
              else:
                # exit the mutation process
                self.reset_islands(keep_island_score_threshold, keep_islands_ids)
                keep_islands_ids, reset_islands_ids = None, None 
                keep_island_score_threshold = None
                prompts_for_mutation = None
                current_prompt = self.get_prompt()
            else:
              current_prompt = self.get_prompt()
            pipe.send(current_prompt)
      except KeyboardInterrupt:
        self.stop()
      if self._exit_flag.is_set():
        print(f"Database 进程收到退出信号，完成backup后退出")
        self.backup()
        break
        

class Island:
  """A sub-population of the codes database."""

  def __init__(
      self,
      template: code_manipulation.Code,
      function_to_evolve: str,
      functions_per_prompt: int,
      cluster_selecting_temperature_init: float,
      cluster_selecting_temperature_period: int,
      kl_divergence_weight: float = 0.75,
      signature_precision: int = 5
  ) -> None:
    self._template: code_manipulation.Code = template
    self._function_to_evolve: str = function_to_evolve
    self._functions_per_prompt: int = functions_per_prompt
    self._cluster_selecting_temperature_init = cluster_selecting_temperature_init
    self._cluster_selecting_temperature_period = (cluster_selecting_temperature_period)

    self._clusters: dict[Signature, Cluster] = {}
    self._num_codes: int = 0

    self._kl_divergence_weight = kl_divergence_weight
    self._signature_precision = signature_precision

  def register_code(
      self,
      code: code_manipulation.Function,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Stores a code on this island, in its appropriate cluster."""
    signature = _get_signature(scores_per_test, self._signature_precision)
    if signature not in self._clusters:
      score = _reduce_score(scores_per_test)
      self._clusters[signature] = Cluster(score, code, scores_per_test)
    else:
      self._clusters[signature].register_code(code, scores_per_test)
    self._num_codes += 1

  def get_prompt(self, use_kl_div=False, override_function_num=None) -> tuple[str, int]:
    """Constructs a prompt containing functions from this island."""
    signatures = list(self._clusters.keys())
    cluster_scores = np.array(
        [self._clusters[signature].score for signature in signatures])

    # Convert scores to probabilities using softmax with temperature schedule.
    period = self._cluster_selecting_temperature_period
    temperature = self._cluster_selecting_temperature_init * (
        1 - (self._num_codes % period) / period)
    probabilities = _softmax(cluster_scores, temperature)

    # At the beginning of an experiment when we have few clusters, place fewercodes into the prompt.
    functions_per_prompt = min(len(self._clusters), 
        (override_function_num if override_function_num is not None else self._functions_per_prompt))

    implementations = []
    scores = []
    idx = np.random.choice(
        len(signatures), size=functions_per_prompt, p=probabilities)
    chosen_signatures = [signatures[i] for i in idx]
    if not use_kl_div or functions_per_prompt==1: # or self._functions_per_prompt!=2:
      for signature in chosen_signatures:
        cluster = self._clusters[signature]
        chosen_code, chosen_scores = cluster.select_code()
        implementations.append(chosen_code)
        scores.append(cluster.score)
    else:
      # modified: use KL div to get the second prompt
      first_code_scores_per_test = None
      for i, signature in enumerate(chosen_signatures):
        cluster = self._clusters[signature]
        if i == 0:
          code, first_code_scores_per_test = cluster.select_code()
        else:
          code, code_scores = cluster.select_code(
                    reference_scores_per_test=first_code_scores_per_test,
                    kl_divergence_weight=self._kl_divergence_weight)
        implementations.append(code)
        scores.append(cluster.score)
      
    indices = np.argsort(scores)
    sorted_implementations = [implementations[i] for i in indices]
    version_generated = len(sorted_implementations) + 1
    return self._generate_prompt(sorted_implementations), version_generated
    
  def _generate_prompt(
      self,
      implementations: Sequence[code_manipulation.Function]) -> str:
    """Creates a prompt containing a sequence of function `implementations`."""
    implementations = copy.deepcopy(implementations)  # We will mutate these.

    # Format the names and docstrings of functions to be included in the prompt.
    versioned_functions: list[code_manipulation.Function] = []
    for i, implementation in enumerate(implementations):
      new_function_name = f'{self._function_to_evolve}_v{i}'
      implementation.name = new_function_name
      # Update the docstring for all subsequent functions after `_v0`.
      if i >= 1:
        implementation.docstring = (
            f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
      # If the function is recursive, replace calls to itself with its new name.
      implementation = code_manipulation.rename_function_calls(
          str(implementation), self._function_to_evolve, new_function_name)
      versioned_functions.append(
          code_manipulation.text_to_function(implementation))

    # Create the header of the function to be generated by the LLM.
    next_version = len(implementations)
    new_function_name = f'{self._function_to_evolve}_v{next_version}'
    header = dataclasses.replace(
        implementations[-1],
        name=new_function_name,
        body='',
        docstring=('Improved version of '
                   f'`{self._function_to_evolve}_v{next_version - 1}`.'),
    )
    versioned_functions.append(header)

    # Replace functions in the template with the list constructed here.
    prompt = dataclasses.replace(self._template, functions=versioned_functions)
    return str(prompt)


class Cluster:
  """A cluster of codes on the same island and with the same Signature."""

  def __init__(self, score: float, implementation: code_manipulation.Function, score_per_test):
    self._score = score
    self._scores_per_test = [score_per_test]
    self._codes: list[code_manipulation.Function] = [implementation]
    self._lengths: list[int] = [len(str(implementation))]

  @property
  def score(self) -> float:
    """Reduced score of the signature that this cluster represents."""
    return self._score

  def register_code(self, code: code_manipulation.Function, scores_per_test) -> None:
    """Adds `code` to the cluster."""
    # print(scores_per_test)
    self._codes.append(code)
    self._scores_per_test.append(scores_per_test)
    self._lengths.append(len(str(code)))

  def select_code(self, reference_scores_per_test: np.ndarray | None = None, 
                     kl_divergence_weight: float = 0.0 # Default to 0 (no KL influence)
                    ) -> tuple[code_manipulation.Function, np.ndarray]:
    """
    select a code from the cluster.
    If reference_scores_per_test is provided, uses KL divergence D_KL(candidate || reference)
    to increase the probability of selecting codes with different score profiles,
    combined with the base probability favouring shorter codes.

    Returns:
        tuple[Function, np.ndarray]: The selected code and its scores_per_test vector.
    """
    
    normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
        max(self._lengths) + 1e-6)
    base_logits = -normalized_lengths
    num_codes = len(self._lengths)
    # Use KL
    if reference_scores_per_test is not None and kl_divergence_weight > 0:
      # print(reference_scores_per_test)
      dist_ref = _scores_to_distribution(reference_scores_per_test)
      # print(dist_ref)
      if dist_ref.size > 0: # Proceed only if reference distribution is valid
        kl_divergences = []
        valid_indices_for_kl = [] # Keep track of codes where KL could be calculated
        for i in range(num_codes):
          candidate_scores = self._scores_per_test[i]
          # Check if candidate scores have the same dimension as reference
          # if candidate_scores.shape == dist_ref.shape:
          dist_candidate = _scores_to_distribution(candidate_scores)
          # Calculate D_KL(dist_candidate || dist_ref)
          kl_div = _calculate_kl_divergence(dist_candidate, dist_ref)
          kl_divergences.append(kl_div)
          valid_indices_for_kl.append(i)
          # print(kl_div)
        # print(kl_divergences)
        if valid_indices_for_kl: # If KL was calculated for at least one code
          # --- 3. Combine base logits and KL divergence ---
          combined_logits = np.array(base_logits, dtype=np.float64) # Use float64 for precision
          kl_array = np.array(kl_divergences, dtype=np.float64)
          # print(kl_array)
          # Add weighted KL divergence only to the valid indices
          # Higher KL -> Higher logit -> Higher probability
          combined_logits[valid_indices_for_kl] += kl_divergence_weight * kl_array
          # Ensure numerical stability before final softmax
          combined_logits -= np.max(combined_logits)
          # --- 4. Calculate final probabilities ---
          final_probabilities = _softmax(combined_logits, temperature=1.0) # T=1 for combined logits
          final_probabilities /= np.sum(final_probabilities) # Ensure sum to 1
        else: # KL divergence calculation failed for all candidates (e.g. shape mismatch everywhere)
          # Fallback to base probabilities
          final_probabilities = _softmax(base_logits, temperature=1.0)
      else: # Reference distribution was invalid
        # Fallback to base probabilities
        final_probabilities = _softmax(base_logits, temperature=1.0)
    else: # Reference distribution was invalid
      # Fallback to base probabilities
      final_probabilities = _softmax(base_logits, temperature=1.0)
    # ==========================
    # final_probabilities safety control
    probabilities_ok = False
    reason = "Unknown validation failure" # Default reason
    if num_codes > 0 and final_probabilities is not None and len(final_probabilities) == num_codes:
        # Check 1: Ensure all probabilities are finite (not NaN or Inf)
        if np.all(np.isfinite(final_probabilities)):
            # Check 2: Ensure all probabilities are non-negative (allow tiny FP errors)
            if np.all(final_probabilities >= -1e-9):
                # Clip tiny negatives to zero before checking sum
                final_probabilities = np.maximum(final_probabilities, 0)
                prob_sum = np.sum(final_probabilities)
    
                # Check 3: Ensure the sum is positive and normalize if needed
                if prob_sum > 1e-9: # Check if sum is meaningfully positive
                    # Normalize if not already close to 1
                    if not np.isclose(prob_sum, 1.0):
                        # print(f"Info: Renormalizing probabilities (sum={prob_sum})") # Optional info
                        final_probabilities /= prob_sum
    
                    # Final verification that sum is now 1 (within tolerance)
                    if np.isclose(np.sum(final_probabilities), 1.0):
                        probabilities_ok = True # All checks passed!
                    else:
                        reason = f"sum failed to normalize to 1 ({np.sum(final_probabilities)})"
                else:
                    # If sum is zero/negative after clipping, can't normalize -> uniform
                    reason = f"sum of non-negative probabilities is zero or near-zero ({prob_sum})"
            else:
                # Found negative values beyond tolerance
                neg_indices = np.where(final_probabilities < -1e-9)[0]
                reason = f"contains negative values at indices {neg_indices}"
        else:
            # Found non-finite values
            nan_indices = np.where(np.isnan(final_probabilities))[0]
            inf_indices = np.where(np.isinf(final_probabilities))[0]
            reason = f"contains NaN at {nan_indices} and/or Inf at {inf_indices}"
    elif num_codes <= 0:
         reason = "cluster has zero codes" # Should likely be caught earlier
    else:
         reason = f"probability vector is None or length mismatch ({len(final_probabilities) if final_probabilities is not None else 'None'} vs {num_codes})"
    
    
    # Apply fallback to uniform distribution if any check failed
    if not probabilities_ok:
        print(f"Warning: Final probabilities invalid ({reason}). Falling back to uniform distribution.")
        if num_codes > 0:
            final_probabilities = np.ones(num_codes, dtype=float) / num_codes
        # else:
        # This state should ideally be prevented before reaching here
        # If we get here, make probabilities empty to likely cause controlled error later or handle explicitly
        # final_probabilities = np.array([], dtype=float)
        # Or raise error immediately:
        # raise ValueError(f"Cannot select, invalid probability state for {num_codes} codes: {reason}")
    
    
    # --- 5. Select code using final probabilities ---
    # Now, 'final_probabilities' should be safe to use with np.random.choice,
    # provided num_codes > 0.
    # Add a check for the zero code case if it wasn't handled earlier
    if num_codes <= 0:
         raise ValueError("Attempting to select from a cluster with zero codes.")
    # post choosing
    chosen_index = np.random.choice(len(self._codes), p=final_probabilities)
    chosen_code = self._codes[chosen_index]
    chosen_scores = self._scores_per_test[chosen_index]
    return chosen_code, chosen_scores # np.random.choice(self._codes, p=probabilities)
