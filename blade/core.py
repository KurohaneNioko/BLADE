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

"""A single-threaded implementation of the BLADE pipeline."""
import logging
import time
import signal
import multiprocessing as mp

from blade import code_manipulation


def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(
      code_manipulation.yield_decorated(specification, 'blade', 'run'))
  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@blade.run`.')
  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'blade', 'evolve'))
  if len(evolve_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@blade.evolve`.')
  return evolve_functions[0], run_functions[0]


def run(generators, database, iterations: int = -1):
  """Launches a BLADE experiment."""

  try:
    # This loop can be executed in parallel on remote generator machines. As each
    # generator enters an infinite loop, without parallelization only the first
    # generator will do any work.
    
    while iterations != 0:
      for i, s in enumerate(generators):
        print("Generator:", i)
        s.generate()
      if iterations > 0:
        iterations -= 1
  except KeyboardInterrupt:
    logging.info("Keyboard interrupt. Stopping.")
  database.backup()


def run_mp(generators, database, iterations: int = -1):
  """Launches a Blade experiment via multiprocessing."""
    
  processes = []    # 0-x: generator
  database.daemon = True
  # processes.append(database)
  database.start()
  for i, s in enumerate(generators):
    print("generator:", i)
    s.start()
    time.sleep(27)
  
  def signal_handler(sig, frame):
    print('\n Ctrl+C catched, shut down elegantly...')
    for p in processes:
        p.stop()
    for p in processes:
        p.join(timeout=10)
    time.sleep(1)    
    database.stop()
    time.sleep(1)
    database.join()
    print("All procs stopped. Exit.")
    sys.exit(0)
  signal.signal(signal.SIGINT, signal_handler)
  
  while True:
    time.sleep(0.2)
