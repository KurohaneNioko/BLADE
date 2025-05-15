import json
import logging
import os
import pathlib
import pickle
import time
import signal
import multiprocessing as mp
import numpy as np
import click
from dotenv import load_dotenv

from blade import config, core, sandbox, generator, codes_database, code_manipulation, evaluator

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(format='%(asctime)s %(message)s', level=LOGLEVEL)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
# for name in logging.root.manager.loggerDict:
#   print(name)


def get_all_subclasses(cls):
  all_subclasses = []

  for subclass in cls.__subclasses__():
    all_subclasses.append(subclass)
    all_subclasses.extend(get_all_subclasses(subclass))

  return all_subclasses


SANDBOX_TYPES = get_all_subclasses(sandbox.DummySandbox) + [sandbox.DummySandbox]
SANDBOX_NAMES = [c.__name__ for c in SANDBOX_TYPES]


def parse_input(filename_or_data: str):
  if len(filename_or_data) == 0:
    raise Exception("No input data specified")
  paths = [p.strip() for p in filename_or_data.split(',')]
  loaded_data = []
  for filepath in paths:
    p = pathlib.Path(filepath)
    if not p.exists():
      # Choose whether to raise error or just warn and skip
      print(f"Warning: Input file not found, skipping: {filepath}")
      continue
    # if not p.name.endswith(".json"):
    #   print(f"Warning: Input file is not a JSON file, skipping: {filepath}")
    #   continue
    try:
      if p.name.endswith(".json"):
        with open(filepath, 'r') as f:
          data = json.load(f)
          # Basic validation (keys checked more thoroughly in evaluate)
          # if not isinstance(data, dict) or not all(k in data for k in ['capacity', 'weights', 'profits', 'optimal_profit']):
          #      print(f"Warning: JSON data in {filepath} lacks required keys or is not a dictionary, skipping.")
          #      continue
          # Optional: Pre-convert lists to numpy arrays here if desired
          # data['weights'] = np.array(data['weights'], dtype=float)
          # data['profits'] = np.array(data['profits'], dtype=float)
          # Note: Keeping as lists is fine, evaluate handles np.array conversion
      elif p.name.endswith(".npy"):
        data = np.load(filepath, allow_pickle=True)
      loaded_data.append(data)
    except json.JSONDecodeError:
      print(f"Warning: Error decoding JSON from {filepath}, skipping.")
      continue
    except Exception as e:
      print(f"Warning: Unexpected error loading {filepath}: {e}, skipping.")
      continue
  if len(loaded_data) != 0:
    return loaded_data
  # p = pathlib.Path(filename_or_data)
  # if p.exists():
  #   if p.name.endswith(".json"):
  #     return json.load(open(filename_or_data, "r"))
  #   if p.name.endswith(".pickle"):
  #     return pickle.load(open(filename_or_data, "rb"))
  #   raise Exception("Unknown file format or filename")

  # INPUT data is pure numbers
  if len(loaded_data) == 0:
    if "," not in filename_or_data:
      data = [filename_or_data]
    else:
      data = filename_or_data.split(",")
    if filename_or_data[0].isnumeric():
      f = int if data[0].isdecimal() else float
      data = [f(v) for v in data]
    return data

@click.group()
@click.pass_context
def main(ctx):
  pass


@main.command()
@click.argument("spec_file", type=click.File("r"))
@click.argument('inputs')
@click.option('--model_jsons', default="gpt-3.5-turbo-instruct", help='LLM model json metadata')
@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing code database')
@click.option('--iterations', default=-1, type=click.INT, help='Max iterations per generator')
@click.option('--generators', default=15, type=click.INT, help='Generator')
@click.option('--max_cpu_count', default=64, type=click.INT, help='total cpu count')
@click.option('--exp_name', default="", help='name of the problem')
@click.option('--validation_num', default=5, type=click.INT, help='num of first X datasets as valation, rest for test')
@click.option('--brainstorming_times_per_prompt', default=3, type=click.INT, help='brainstorming times per prompt')
@click.option('--signature_precision', default=3, type=click.INT, help='0.876 / 0.8765 / 0.87654 as sign, precision=3,4,5')
@click.option('--sandbox_type', default="ContainerSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
def run(spec_file, inputs, model_jsons, output_path, load_backup, iterations, generators, max_cpu_count, exp_name, validation_num, brainstorming_times_per_prompt, signature_precision,
sandbox_type):
  """ Execute function-search algorithm:

\b
  SPEC_FILE is a python module that provides the basis of the LLM prompt as
            well as the evaluation metric.
            See examples/cap_set_spec.py for an example.\n
\b
  INPUTS    input filename ending in .json or .pickle, or a comma-separated
            input data. The files are expected contain a list with at least
            one element. Elements shall be passed to the solve() method
            one by one. Examples
              8
              8,9,10
              ./examples/cap_set_input_data.json
"""
  # print(inputs)
  # Load environment variables from .env file.
  #
  # Using OpenAI APIs with 'llm' package requires setting the variable
  # OPENAI_API_KEY=sk-...
  # See 'llm' package on how to use other providers.
  load_dotenv()
  
  timestamp = str(int(time.time()))
  timestamp += f"-{exp_name}"
  log_path = pathlib.Path(output_path) / timestamp
  if not log_path.exists():
    log_path.mkdir(parents=True)
    logging.info(f"Writing logs to {log_path}")
  file_logger = logging.getLogger('result')
  f_handler = logging.FileHandler(log_path / 'result.log')
  f_handler.setLevel(LOGLEVEL)
  file_logger.propagate = False
  f_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
  file_logger.addHandler(f_handler)


  # mp
  save_code_queue = mp.Queue()    # evaluator ---queue---> db
  # send_prompt_pipes = [mp.Pipe() for _ in range(generators)]
  prompt_pipes_db_end = []
  prompt_pipes_generator_end = []
  for _ in range(generators):
    db_end, generator_end = mp.Pipe()
    prompt_pipes_db_end.append(db_end)
    prompt_pipes_generator_end.append(generator_end)

  model_json_filenames = model_jsons.split(",")
  model_items = []
  for filepath in model_json_filenames:
    with open(filepath, 'r') as f:
      model_items.append(json.load(f))
  # print(model_items)

  specification = spec_file.read()
  function_to_evolve, function_to_run = core._extract_function_names(specification)
  template = code_manipulation.text_to_code(specification)

  conf = config.Config()
  print(conf, conf.codes_database)
  database = codes_database.CodesDatabase(
    conf.codes_database, template, function_to_evolve, log_path=log_path, identifier=timestamp,
    save_code_queue=save_code_queue, send_prompt_pipes=prompt_pipes_db_end, signature_precision=signature_precision)
  if load_backup:
    database.load(load_backup)
    print("backup loaded")

  inputs = parse_input(inputs)


    
  q_generator2eval = mp.Queue(maxsize=max_cpu_count*4)    # generator --> q --> eval
  sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)
  # evaluators = [evaluator.Evaluator(
  #   # database,
  #   sandbox_class(base_path=log_path),
  #   template,
  #   function_to_evolve,
  #   function_to_run,
  #   inputs,
  #   send_code_queue=save_code_queue,
  #   timeout_seconds=30,
  # # ) for _ in range(conf.num_evaluators)]
  # ) for _ in range(generators)]
  evaluater_pool = evaluator.EvaluatorPool(
    sandbox_class,
    log_path,
    template,
    function_to_evolve,
    function_to_run,
    inputs,
    validation_num,
    init_subproc_num=generators,
    receive_code_queue=q_generator2eval,
    send_code_queue=save_code_queue,
    timeout_seconds=30,max_cpu_count=max_cpu_count, file_logger=file_logger)

  # We send the initial implementation to be analysed by one of the evaluators.
  if not load_backup:
    initial = template.get_function(function_to_evolve).body
    evaluater_pool.save_init_to_db(initial, island_id=None, version_generated=None, database=database)
    time.sleep(2)
    assert len(database._islands[0]._clusters) > 0, ("Initial analysis failed. Make sure that Sandbox works! "
                                                   "See e.g. the error files under sandbox data.")
  
  generators = [generator.Generator(
      # evaluators[i], 
      q_generator2eval,
      model_items,
      samples_per_prompt=2,
      brainstorming_times_per_prompt=brainstorming_times_per_prompt, 
      recv_prompt_pipe=prompt_pipes_generator_end[i],
      log_path=log_path, generator_id=i) for i in range(generators)]

  # core.run(generators, database, iterations)
  # core.run_mp(generators, database, iterations)
  # copy core.run_mp here!
  """Launches a BLADE experiment via multiprocessing."""
  # processes = []    # 0-x: generator
  database.daemon = True
  database.start()
  evaluater_pool.start()
  for i, s in enumerate(generators):
    print("Generator:", i)
    s.start()
    time.sleep(i+1)
  def signal_handler(sig, frame):
    print('\n Ctrl+C Catch, shut down elegantly...')
    for s in generators:
        s.stop()
    # stop generator: generator will end after put.
    time.sleep(30)
    evaluater_pool.stop()    # work until no item in q_generator2eval
    time.sleep(8)
    for eval_proc in evaluater_pool._evaluator_procs:
        eval_proc.join()
    print("eval finished.")
    time.sleep(3)
    database.stop()
    time.sleep(1)
    database.join()
    for s in generators:
        s.join()
    print("All procs stopped. Exit.")
    sys.exit(0)
  signal.signal(signal.SIGINT, signal_handler)
  while True:
    time.sleep(2)
    for s in generators:
      if not s.is_alive():
        # a generator died!
        s.start()
        time.sleep(2)


@main.command()
@click.argument("db_file", type=click.File("rb"))
def ls(db_file):
  """List codes from a stored database (usually in data/backups/ )"""
  conf = config.Config(num_evaluators=1)

  # A bit silly way to list codes. This probably does not work if config has changed any way
  database = codes_database.CodesDatabase(conf.codes_database, None, "", identifier="",
  save_code_queue=None, send_prompt_pipes=None, log_path="./data")
  database.load(db_file)

  codes = database.get_best_codes_per_island()
  print(f"Found {len(codes)} codes")
  for i, (code, score) in enumerate(codes):
    print(f"{i}: Code with score {score}")
    print(code)
    print("\n")
  while True:
    input_scores = input('Input scores like "0.888 0.919 0.585", to get the code with specific score.').strip()
    if input_scores[0] == '"': input_scores = input_scores[1:]
    if input_scores[-1] == '"': input_scores = input_scores[:-1]
    for island in database._islands:
      for _signature, cluster in island._clusters:
        for idx, scores in enumerate(cluster._scores_per_test):
          scores_str = " ".join(list(map(lambda x: f"{x:.4f}", scores))).strip()
          if score_str == input_scores:
            code = cluster._codes[idx]
            print(f"Found! score={np.mean(scores)}")
            print(code)
            print('\n')
        

if __name__ == '__main__':
  main()
