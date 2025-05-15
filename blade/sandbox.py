import logging
import json
import hashlib
import ast
import os
import pathlib
import sys
import time
from typing import Any

import cloudpickle

CONTAINER_MAIN = (pathlib.Path(__file__).parent / "container" / "container_main.py").absolute()

IMAGE_NAME = "blade_sandbox"


class DummySandbox:
  """Base class for Sandboxes that execute the generated code.

  Note: this base class executes the code but does not offer any sandboxing!!!
  It should be only used in unit testing or debugging, and not with real LLM
  unless the host environment is in some kind of sandbox itself.
  Even in sandboxed host, the executed code could theoretically affect later executions.
  """

  sandboxes = 0

  def __init__(self, **kwargs):
    self.id = DummySandbox.sandboxes

    DummySandbox.sandboxes += 1

  def run(
          self,
          program: str,
          function_to_run: str,
          test_input,
          timeout_seconds: int,
  ) -> tuple[Any, bool]:
    """Returns `function_to_run(test_input)` and whether execution succeeded."""

    # The same "program" seems to be now repeatedly parsed using AST and then compiled.
    # This could probably be simplified quite a bit.
    namespace = DummySandbox.compile_code(program)
    return namespace[function_to_run](test_input)

  @staticmethod
  def compile_code(program: str):
    namespace = {}

    parsed_code = ast.parse(program)
    compiled_code = compile(parsed_code, filename="<ast>", mode="exec")
    exec(compiled_code, namespace)
    return namespace


class ExternalProcessSandbox(DummySandbox):
  """Sandbox that executes the code in a separate Python process in the same host.

  Note: This does not provide real safety and should be only used in an environment where the host process is
  in some kind of safe sandbox itself (i.e., a container).
  This kind of sandbox merely makes it more probable that single invalid call does not break the whole
  blade algorithm. It might be easier to set up and thus nice environment to tune the prompts and other code.
  """

  def __init__(self, base_path: pathlib.Path, timeout_secs: int = 30, python_path: str = "python"):
    super(ExternalProcessSandbox, self).__init__()

    self.output_path = pathlib.Path(base_path) / f"sandbox{self.id}"
    self.timeout_secs = timeout_secs
    self.python_path = python_path
    self.call_count = 0

    self.input_path = self.output_path / "inputs"
    for p in [self.output_path, self.input_path]:
      if not p.exists():
        p.mkdir(parents=True)

  def _exec(self, call_data_path: pathlib.Path, input_path: pathlib.Path, error_file_path: pathlib.Path):
    """Use podman/docker to execute python in a container.
    - The main.py shall execute the LLM generated method from prog.pickle file providing
      input.pickle as the input for the method.
    - main.py writes the output of the method into output.pickle.
    Everything except the /workspace folder will be read-only so that the environment remains good
    for future runs.
    """
    prog_path = call_data_path / "prog.pickle"
    output_file = call_data_path / "output.pickle"
    cmd = (f"{self.python_path} {CONTAINER_MAIN} {prog_path} {input_path} {output_file}"
           f"  2> {error_file_path}")
    # print(cmd)
    logging.debug(f"Executing: {cmd}")
    return os.system(cmd)

  def run(
          self,
          program: str,
          function_to_run: str,
          test_input,
          timeout_seconds: int,
  ) -> tuple[Any, bool]:

    call_data_folder = (self.output_path / f"call{self.call_count}").absolute()
    if not call_data_folder.exists():
      call_data_folder.mkdir()

    try:
      input_hash = hash(test_input)
    except TypeError:
      try:
        # Use separators=(',', ':') for the most compact representation without spaces
        canonical_string = json.dumps(test_input, sort_keys=True, separators=(',', ':'))
        # Create a SHA256 hash of the UTF-8 encoded string
        # Using hexdigest() gives a standard hexadecimal string suitable for filenames
        input_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
      except TypeError:    # last resort
        object_id = id(test_input)
        timestamp_ns = time.time_ns() # Nanosecond timestamp
        # Combine them into a string
        unique_string_source = f"{object_id}-{timestamp_ns}"
        # Hash this combined string for a fixed-length, cleaner ID
        input_hash = hashlib.sha256(unique_string_source.encode('utf-8')).hexdigest()

    # print(input_hash)
    input_path = (self.input_path / f"{input_hash}.pickle").absolute()
    # print(input_path)
    if not input_path.exists():
      with open(input_path, "wb") as f:
        cloudpickle.dump(test_input, f)
    try:
      namespace = DummySandbox.compile_code(program)

      prog_file = (call_data_folder / f"prog.pickle").absolute()
      with open(prog_file, "wb+") as f:
        cloudpickle.dump(namespace[function_to_run], f)

      error_file = self.output_path / f"stderr_{self.call_count}.log"

      retcode = self._exec(call_data_folder, input_path, error_file)
      self.call_count += 1
      # print(call_data_folder, input_path)

      if retcode != 0:
        print("err in sandbox running python file.")
        self._save_diagnostics(program, call_data_folder)
        return None, False

      output_file = call_data_folder / f"output.pickle"
      with open(output_file, "rb") as f:
        out = cloudpickle.load(f)
        return out, True
    except Exception as e:
      # print(e, flush=True)
      logging.debug(f"Could not execute code: {e}")
    self._save_diagnostics(program, call_data_folder)
    return None, False

  @staticmethod
  def _save_diagnostics(program: str, output_path: pathlib.Path):
    filepath = output_path / "program.py"
    logging.debug(f"Writing program to {filepath}")
    with open(filepath, "w+") as f:
      f.write(program)
