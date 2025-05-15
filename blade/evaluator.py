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

"""Class for evaluating codes proposed by the Generator."""
import ast
import re
import time
from collections.abc import Sequence
import copy
from typing import Any, Tuple
from absl import logging
import threading
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Pipe
from multiprocessing.connection import Connection 
from queue import Empty

from blade import code_manipulation
from blade import codes_database
from blade import sandbox

"""
  Regex to find all methods named 'solve_vX'.
  With each match, start from the 'def solve_vX(' and continue until there's a new line with any of
  - a new 'def'
  - ` or ' or # without indentation
"""
# modified: return type can be *int* or float !
# version number can be very large!
METHOD_MATCHER = re.compile(r"def solve_v\d+\([\s\S]*?\)\s*->\s*(?:float|int):(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
# METHOD_MATCHER = re.compile(
#     r"(def solve_v\d+\([\s\S]*?\)\s*->\s*(?:float|int):[\r\n]+)"  # 匹配函数定义
#     r"((?:^[ \t]+(?!def |#|`).*[\r\n]+)*)"  # 函数体内容（带缩进且不遇到特定终止条件）
#     r"(?=^[ \t]*$|^[ \t]*(?:def |#|@|`))",  # 前瞻断言确保终止条件
#     flags=re.MULTILINE | re.DOTALL)
METHOD_NAME_MATCHER = re.compile(r"solve_v\d+")
METHOD_MATCHER_NO_VX = re.compile(r"def solve\([\s\S]*?\)\s*->\s*(?:float|int):(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
# METHOD_MATCHER_NO_VX = re.compile(
#     r"def solve\([\s\S]*?\)\s*->\s*(?:float|int):[\r\n]+"  # 匹配函数定义
#     r"((?:^[ \t]+(?!def |#|`).*[\r\n]+)*)"  # 函数体内容（带缩进且不遇到特定终止条件）
#     r"(?=^[ \t]*$|^[ \t]*(?:def |#|@|`))",  # 前瞻断言确保终止条件
#     flags=re.MULTILINE | re.DOTALL)
METHOD_NAME_MATCHER_NO_VX = re.compile(r"solve")

class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _find_method_implementation(generated_code: str) -> Tuple[str, str]:
  """Find the last 'def solve_vX()' method from generated code.

  Return the code and the name of the method.
  """
  # modified: match solve no _vX if there is no _vX
  matches = METHOD_MATCHER.findall(generated_code)
  # print(len(matches))
  if __name__ == "__main__":
    print(f"_find_method_implementation: matches, len={len(matches)}")
    print(matches)
  if matches:
    # return "", ""
    # title = matches[0]
    last_match = matches[-1]
    # print(matches)
    # print(title)
    # print(last_match)
    name = METHOD_NAME_MATCHER.search(last_match).group()
    return last_match, name
  matches = METHOD_MATCHER_NO_VX.findall(generated_code)
  if __name__ == "__main__":
    print(f"_find_method_implementation: matches, len={len(matches)}")
    print(matches)
  if not matches:    # TODO: simply extract contents in first ```python to ```
    extract_content =  generated_code.split("```python")[1].split("```")[0]
    extract_content = extract_content.split("@blade.evolve")[-1].strip()
    # print(extract_content)
    name = extract_content.split('def ', 1)[-1].split('(', 1)[0].strip()
    # print(name)
    # name = METHOD_NAME_MATCHER.search(extract_content).group()
    return extract_content, name
  last_match = matches[-1]
  name = METHOD_NAME_MATCHER_NO_VX.search(last_match).group()
  return last_match, name

# def _find_method_implementation(generated_code: str) -> Tuple[str, str]:
#   """Find the last 'def solve_vX()' method from generated code.

#   Return the code and the name of the method.
#   """
#   matches = METHOD_MATCHER.findall(generated_code)
#   if not matches:
#     return "", ""
#   last_match = matches[-1]
#   name = METHOD_NAME_MATCHER.search(last_match).group()
#   return last_match, name
  

def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''
  if not type(generated_code) is str:
    generated_code = str(generated_code)

  method_name = "fake_function_header"
  # Check is the response only a continuation for our prompt or full method implementation with header
  # modified: support no _vX
  if "def solve_v" in generated_code or "def solve" in generated_code:
    code, method_name = _find_method_implementation(generated_code)
  else:
    code = f'def {method_name}():\n{generated_code}'
  if __name__ == "__main__":
    print("code")
    print(code)
    print("============")
  # print(method_name)
  # print("============")
    
  # Finally parse the code to make sure it's valid Python
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  visitor = _FunctionLineVisitor(method_name)
  visitor.visit(tree)
  # maybe the function head is multiple lines
  # modified: find the first line with `-> int` or `-> float`
  return_type_pattern = re.compile(r"->\s*(?:float|int)\s*:")
  lines = code.splitlines()
  # 查找第一个包含返回类型声明的行
  start_line = 1
  for idx, line in enumerate(lines):
    if return_type_pattern.search(line):
      start_line = idx + 1  # 函数体从下一行开始
      break
  body_lines = lines[start_line:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'


def _sample_to_code(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Code,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str, int]:
  """Returns the compiled generated function and the full runnable code."""
  warning_tag = 0
  body = _trim_function_body(generated_code)
  if len(body) < 7:
    warning_tag = 1
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  code = copy.deepcopy(template)
  evolved_function = code.get_function(function_to_evolve)
  evolved_function.body = body
  # print(body)
  # print("============")
  # print(evolved_function)
  # print("============")
  # print(code)
  # print("============")
  # print("============")
  # exit(-9)
  return evolved_function, str(code), warning_tag



def _calls_ancestor(code: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(code):
    # In `code` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False

# ==============================================
# POOL!
# ==============================================

class EvaluatorPool:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      # database: codes_database.CodesDatabase,
      sbox_type,
      sbox_log_path,
      template: code_manipulation.Code,
      function_to_evolve: str,
      function_to_run: str,
      inputs: Sequence[Any],
      validation_num: int,
      # from generator
      receive_code_queue: Queue,
      # to DB
      send_code_queue: Queue,
      init_subproc_num: int = 8,
      timeout_seconds: int = 30,
      max_cpu_count: int = 64,
      file_logger = None
      
  ):
    # self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self._timeout_seconds = timeout_seconds
    # self._sandbox = sbox
    self._sbox_type = sbox_type
    self._sbox_log_path = sbox_log_path
    self._sboxes = []
    self._valadation_num = validation_num
    # Queue for saving code
    self._receive_code_queue = receive_code_queue
    self._send_code_queue = send_code_queue
    # process pool
    self._evaluator_procs = []
    self._init_subproc_num = init_subproc_num
    self._max_subproc_num = max(int(1.5*init_subproc_num*2), max_cpu_count)  #int(1.25*mp.cpu_count())
    self._idx_add_worker = -1
    # monitor thread
    self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
    self._active = mp.Value('b', True)
    self._scale_up_lock = threading.Lock()

    self.file_logger = file_logger
    

  def save_init_to_db(self, 
    sample: str,
    island_id: int | None,
    version_generated: int | None,
    database
  ) -> None:
    # print("INIT DB")
    sandbox = self._sbox_type(base_path=self._sbox_log_path)
    
    new_function, code, warning_tag = _sample_to_code(
    sample, version_generated, self._template, self._function_to_evolve)
    scores_per_test = []  # order same as _inputs
    for current_input in self._inputs[:self._valadation_num]:
      # print(current_input, type(current_input))
      test_output, runs_ok = sandbox.run(
          code, self._function_to_run, current_input, self._timeout_seconds)
      # print('data run', test_output)
      if (runs_ok and not _calls_ancestor(code, self._function_to_evolve)
          and test_output is not None):
        if not isinstance(test_output, (int, float)):
          raise ValueError('@function.run did not return an int/float score.')
        logging.info(f"ISLAND {island_id}, test_output: {test_output}")
        scores_per_test.append(float(test_output))
    if scores_per_test:
      database.register_code(new_function, island_id, scores_per_test)

  def start(self):
      # 初始启动 min_workers
      for _ in range(self._init_subproc_num):
          self._add_worker()
      self._monitor_thread.start()
      logging.info(f"[EvaluatorPool] evaluator process max num {self._max_subproc_num}")
    
  def _add_worker(self):
      with self._scale_up_lock:
          if len(self._evaluator_procs) >= self._max_subproc_num:
              return
          # ready sbox for each proc
          self._idx_add_worker += 1
          sandbox = self._sbox_type(base_path=self._sbox_log_path)
          self._sboxes.append(sandbox)
          p = mp.Process(target=self._evaluator_loop, args=(self._idx_add_worker,))
          p.start()
          self._evaluator_procs.append(p)
          logging.info(f"[EvaluatorPool] Add new evaluator process PID {p.pid}, current evaluator num {len(self._evaluator_procs)}")
    
  def _monitor_loop(self):
      while self._active.value:
          time.sleep(7)
          qsize = self._receive_code_queue.qsize()
          current_workers = len(self._evaluator_procs)
          # 动态扩容逻辑
          multiplier = 1
          if qsize > current_workers*multiplier and current_workers < self._max_subproc_num:
              workers_needed = min(
                  (qsize // multiplier) - current_workers,
                  self._max_subproc_num - current_workers
              )
              for _ in range(workers_needed):
                  self._add_worker()

  def _evaluator_loop(self, idx):
      while True:
          try:    # 智能获取任务：阻塞式获取，但每1秒检查进程状态
              task_tuple = self._receive_code_queue.get(block=True, timeout=1)
              # check sbox
              self.analyse(*task_tuple, idx)
              # self.result_queue.put(result)
          except Empty:
              # 此处可添加优雅退出逻辑
              if not self._active.value:
                  break
          except Exception as e:
              logging.info(f"Evaluator error: {e}")

    
  def shutdown(self):
      self._active.value = False
      # for p in self.workers:
      #     p.terminate()
    

  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
      generator_id: int,
      code_name: str,
      sbox_idx: int
  ) -> None:
    """Compiles the sample into a code and executes it on test inputs."""
    sandbox = self._sboxes[sbox_idx]
    new_function, code, warning_tag = _sample_to_code(
        sample, version_generated, self._template, self._function_to_evolve)
    if __name__ == "__main__":
      print("new_function, code")
      print(new_function, code)

    if warning_tag == 1:
        logging.info(f"Gen {generator_id} {code_name} Lexical Error!")
        return
    
    scores_per_test = []  # order same as _inputs
    for current_input in self._inputs:
      test_output, runs_ok = sandbox.run(
          code, self._function_to_run, current_input, self._timeout_seconds)
      if __name__ == "__main__":
        print(code)
        print(self._function_to_run)
        print(runs_ok)
        print(_calls_ancestor(code, self._function_to_evolve))
        print(test_output)
      if (runs_ok and not _calls_ancestor(code, self._function_to_evolve)
          and test_output is not None):
        if not isinstance(test_output, (int, float)):
          raise ValueError('@function.run did not return an int/float score.')
        # logging.info(f"ISLAND {island_id}, test_output: {test_output}")
        # print(f"ISLAND {island_id}, test_output: {test_output}")
        scores_per_test.append(float(test_output))
        if len(scores_per_test) == self._valadation_num and sum(scores_per_test) < 1e-3:
            break    # do not waste time on rubbish codes
    if scores_per_test:
      output_s = f"G{generator_id} Is{island_id}:"
      for r in scores_per_test[:min(self._valadation_num, len(scores_per_test))]:
        output_s = output_s + f" {r:.4f}"
      output_s += f"-> avg: {np.mean(scores_per_test[:min(self._valadation_num, len(scores_per_test))]):.4f}"
      if self._valadation_num < len(scores_per_test):
        output_s += " | test: "
        for r in scores_per_test[self._valadation_num:]:
          output_s = output_s + f"{r:.4f} "
      output_s += f"| {code_name}"
      logging.info(output_s)
      if self.file_logger != None:
        self.file_logger.info(output_s)
      if all([_<=1e-9 for _ in scores_per_test]):
        logging.info(f"Gen {generator_id} {code_name} Runtime ERROR!")
        return  # do not save the all 0 codee
    if scores_per_test:
      self._send_code_queue.put((new_function, island_id, scores_per_test[:self._valadation_num]))
    
