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

"""Class for selecting new programs."""
from collections.abc import Collection, Sequence

import asyncio
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    AsyncRetrying 
)
import random
import numpy as np
import multiprocessing as mp
from typing import Any, Iterable, Tuple, List
import openai
from openai import OpenAI, AsyncOpenAI
from multiprocessing import Queue, Pipe
from multiprocessing.connection import Connection 

from funsearch import evaluator
from funsearch import programs_database

class BrainstormLLMs:
  """Use 2 LLMs for brainstorming """  
  def __init__(self, samples_per_prompt: int, 
               brainstorming_times_per_prompt: int, 
               model_items, 
               generator_id,
               generator_alive,
               q_generator2eval,
               log_path=None) -> None:
    self._generator_id=generator_id
    self._samples_per_prompt = samples_per_prompt
    self._brainstorming_times_per_prompt = brainstorming_times_per_prompt
    assert len(model_items) == 2, ("Now only support 2 LLMs.")
    # model_items: [{"api_key": XXX, "base_url": "yyy"}, ...]
    self.prompt_count = 0
    self.log_path = log_path
    if not self.log_path.exists():
      self.log_path.mkdir(parents=True)

    self.clients = [AsyncOpenAI(api_key=item["api_key"], base_url=item["base_url"]) for item in model_items]
    self.model_names = [item["model_name"] for item in model_items]
    # generation config
    self.generation_params = {}
    self.generation_params['top_p'] = 0.95
    self.generation_params['temperature'] = 1.02
    self.generation_params['presence_penalty'] = 0.88
    self.generation_params['max_tokens'] = 2048
    self.system_prompt = {"role": "system", "content": "You are a heuristic algorithm designer."}
    # mp
    self._q_generator2eval = q_generator2eval
    self._active = generator_alive #mp.Value('b', True)
    
  def fit_brainstorming_prompt(self, response_from_another):
    return f"Here are algorithmic designs from other models for the problem.\nOne algorithm from a model: ````{response_from_another}````\nYou are free to design a new version of the heuristic algorithm to solve the problem."

  @retry(
    retry=retry_if_exception_type((openai.APIError, openai.APITimeoutError, openai.InternalServerError)),
    wait=wait_exponential(multiplier=2, min=3, max=38),
    # stop=stop_after_attempt(12)
  )
  async def _chat_llm(self, client_idx, prompt, messages):
    messages.append({"role": "user", "content": prompt})
    response = await self.clients[client_idx].chat.completions.create(
        model=self.model_names[client_idx],
        messages=messages,
        **self.generation_params,
    )
    response = response.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    return response, messages
    
  async def _generate_code(self, prompt: str, evaluator, island_id, version_generated, generator_id, evaluated_index=-1) -> str:
    """Returns a predicted continuation of `prompt`."""
    responses = []
    history_dialogs = [[{"role": "system", "content": "You are a heuristic algorithm designer. It is OK to import numpy as np and use other libraries at the beginning, but please put all the auxiliary python code you write *inside the solve function*, as they will be *only* used by the solve function and only the code in the solve function will be checked and used for evaluation."},] for _ in range(len(self.model_names))]
    evaluated_index = -1
    for turn in range(self._brainstorming_times_per_prompt):
      if turn == 0: prompt1, prompt2 = prompt, prompt
      tuple1 = asyncio.create_task(self._chat_llm(0, prompt1, history_dialogs[0]))
      tuple2 = asyncio.create_task(self._chat_llm(1, prompt2, history_dialogs[1]))

      tuple1, tuple2 = await asyncio.gather(tuple1, tuple2)
      resp1, hist1 = tuple1
      resp2, hist2 = tuple2
      history_dialogs = [hist1, hist2]
      responses.append(resp1)
      responses.append(resp2)
      # send to eval procs
      for _ in range(2):
          target_evaluate_idx = evaluated_index+1
          a_or_b = 'a' if target_evaluate_idx%2==0 else 'b'
          code_name = f"r{self.prompt_count}_{a_or_b}_{target_evaluate_idx//2}"
          # evaluator.analyse(responses[target_evaluate_idx], island_id, version_generated, generator_id, code_name)
          self._q_generator2eval.put((responses[target_evaluate_idx], island_id, version_generated, generator_id, code_name))
          evaluated_index = target_evaluate_idx
      self._log_brainstorm((prompt1, prompt2), (resp1, resp2), self.prompt_count)
      if not self._active:    # exit when ctrl+c
          break
      prompt1, prompt2 = self.fit_brainstorming_prompt(resp2), self.fit_brainstorming_prompt(resp1)
    self.prompt_count += 1
    return responses, evaluated_index

  def _log_brainstorm(self, prompts: Tuple[str], responses: Tuple[str], index: int):
    """   """
    if self.log_path is not None:
      with open(self.log_path / f"prompt_{index}_A.log", "a") as fa:
        with open(self.log_path / f"prompt_{index}_B.log", "a") as fb:
          fp_list = [fa, fb]
          for i, p in enumerate(prompts):
            f = fp_list[i]
            f.write(p)
            f.write("\n=======-+++-=======\n")
      with open(self.log_path / f"response_{index}_A.log", "a") as fa:
        with open(self.log_path / f"response_{index}_B.log", "a") as fb:
          fp_list = [fa, fb]
          for i, resp in enumerate(responses):
            f = fp_list[i]
            f.write(str(resp))
            f.write("\n=======-+++-=======\n")


class SingleLLM(BrainstormLLMs):
    
  def __init__(self, samples_per_prompt: int, 
               brainstorming_times_per_prompt: int, 
               model_items, 
               generator_id,
               generator_alive,
               q_generator2eval,
               log_path=None) -> None:
    self._generator_id=generator_id
    self._samples_per_prompt = samples_per_prompt
    self._brainstorming_times_per_prompt = brainstorming_times_per_prompt
    assert len(model_items) == 1, ("Only need 1 LLMs in ablation")
    # model_items: [{"api_key": XXX, "base_url": "yyy"}, ...]
    self.prompt_count = 0
    self.log_path = log_path
    if not self.log_path.exists():
      self.log_path.mkdir(parents=True)
    self.clients = [AsyncOpenAI(api_key=item["api_key"], base_url=item["base_url"]) for item in model_items]
    self.model_names = [item["model_name"] for item in model_items]
    # generation config
    self.generation_params = {}
    self.generation_params['top_p'] = 0.95
    self.generation_params['temperature'] = 1.02
    self.generation_params['presence_penalty'] = 0.88
    self.generation_params['max_tokens'] = 2048
    self.system_prompt = {"role": "system", "content": "You are a heuristic algorithm designer."}
    # mp
    self._q_generator2eval = q_generator2eval
    self._active = generator_alive #mp.Value('b', True)

  def fit_brainstorming_prompt(self, response_from_another):
    return f"You are free to design a new version of the heuristic algorithm to solve the problem."

  async def _generate_code(self, prompt: str, evaluator, island_id, version_generated, generator_id, evaluated_index=-1) -> str:
    """Returns a predicted continuation of `prompt`."""
    responses = []
    history_dialogs = [[{"role": "system", "content": "You are a heuristic algorithm designer. It is OK to import numpy as np and use other libraries at the beginning, but please put all the auxiliary python code you write *inside the solve function*, as they will be *only* used by the solve function and only the code in the solve function will be checked and used for evaluation."},] for _ in range(len(self.model_names))]
    evaluated_index = -1
    for turn in range(self._brainstorming_times_per_prompt):
      if turn == 0: prompt1 = prompt
      tuple1 = await asyncio.create_task(self._chat_llm(0, prompt1, history_dialogs[0]))
      # tuple2 = asyncio.create_task(self._chat_llm(1, prompt2, history_dialogs[1]))
      # tuple1 = await asyncio.gather(tuple1)
      resp1, hist1 = tuple1
      # resp2, hist2 = tuple2
      history_dialogs = [hist1]
      responses.append(resp1)
      # responses.append(resp2)
      # send to eval procs
      for _ in range(1):
          target_evaluate_idx = evaluated_index+1
          # a_or_b = 'a' if target_evaluate_idx%2==0 else 'b'
          code_name = f"r{self.prompt_count}_{target_evaluate_idx}"
          # evaluator.analyse(responses[target_evaluate_idx], island_id, version_generated, generator_id, code_name)
          self._q_generator2eval.put((responses[target_evaluate_idx], island_id, version_generated, generator_id, code_name))
          evaluated_index = target_evaluate_idx
      self._log_brainstorm((prompt1, ), (resp1, ), self.prompt_count)
      if not self._active:    # exit when ctrl+c
          break
      # prompt1, prompt2 = self.fit_brainstorming_prompt(resp2), self.fit_brainstorming_prompt(resp1)
      prompt1 = self.fit_brainstorming_prompt(resp1)
    self.prompt_count += 1
    return responses, evaluated_index

  def _log_brainstorm(self, prompts: Tuple[str], responses: Tuple[str], index: int):
    if self.log_path is not None:
      with open(self.log_path / f"prompt_{index}_A.log", "a") as fa:
        # with open(self.log_path / f"prompt_{index}_B.log", "a") as fb:
        fp_list = [fa, ]
        for i, p in enumerate(prompts):
          f = fp_list[i]
          f.write(p)
          f.write("\n=======-+++-=======\n")
      with open(self.log_path / f"response_{index}_A.log", "a") as fa:
        # with open(self.log_path / f"response_{index}_B.log", "a") as fb:
        fp_list = [fa, ]
        for i, resp in enumerate(responses):
          f = fp_list[i]
          f.write(str(resp))
          f.write("\n=======-+++-=======\n")

class Generator(mp.Process):
  """Node that generates program continuations and sends them for analysis."""

  def __init__(
      self, # database: programs_database.ProgramsDatabase,
      # evaluator: evaluator.Evaluator,    # single eval, # evaluators: Sequence[evaluator.Evaluator],
      q_generator2eval,
      # llm
      model_items,
      samples_per_prompt,
      brainstorming_times_per_prompt, 
      # Queue in DB and Pipe here
      recv_prompt_pipe: Connection,
      log_path,
      generator_id,
  ) -> None:
    super().__init__()
    # self._database = database
    self._recv_prompt_pipe = recv_prompt_pipe
    self._evaluator = evaluator
    self._id = generator_id
    self._alive = mp.Value('b', True)
    
    LLM_CLASS = BrainstormLLMs if len(model_items)==2 else SingleLLM
    self._llm = LLM_CLASS(samples_per_prompt=samples_per_prompt, 
                               brainstorming_times_per_prompt=brainstorming_times_per_prompt, 
                               model_items=model_items, 
                               generator_id=self._id,
                               generator_alive = self._alive,
                               q_generator2eval=q_generator2eval,
                               log_path= log_path / f"G{self._id}")
    # mp
    self._exit_flag = mp.Event()  # 用于通知进程退出的事件

  def generate(self):
    """Continuously gets prompts, generates codes, sends them for analysis."""
    # prompt = self._database.get_prompt()
    self._recv_prompt_pipe.send("p")
    prompt = self._recv_prompt_pipe.recv()

    codes = []
    evaluated_index = -1
    for _ in range(self._llm._samples_per_prompt):
      # resps, eva_idx = loop.run_until_complete(self._llm._generate_code(prompt.code,
      #     self._evaluator, prompt.island_id, prompt.version_generated, self._id, evaluated_index
      # ))
      resps, eva_idx = asyncio.run(self._llm._generate_code(prompt.code,
          self._evaluator, prompt.island_id, prompt.version_generated, self._id, evaluated_index
      ))
      codes = codes + resps
      evaluated_index = eva_idx
      if not self._alive.value: break    # exit when ctrl+c
    # Close the loop when done
    # loop.close()
    
  def stop(self):
    self._exit_flag.set()
    self._alive.value = False
    
  def run(self):
    while not self._exit_flag.is_set():
      self.generate()
      if self._exit_flag.is_set():
        print(f"Generator exits.")
        break