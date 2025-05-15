"""This file will be used as an executable script by the ContainerSandbox and ExternalProcessSandbox.
"""
import logging
import pickle
import sys


def main(code_file: str, input_file: str, output_file: str):
  """The method takes executable function as a cloudpickle file, then executes it with input data,
  and writes the output data to another file."""
  logging.debug(f"Running main(): {code_file}, {input_file}, {output_file}")
  with open(code_file, "rb") as f:
    func = pickle.load(f)

    with open(input_file, "rb") as input_f:
      input_data = pickle.load(input_f)

      ret = func(input_data)
      # print(func, input_data, ret)
      with open(output_file, "wb") as of:
        logging.debug(f"Writing output to {output_file}")
        pickle.dump(ret, of)

def _main(code_file: str, input_file: str):
  """The method takes executable function as a cloudpickle file, then executes it with input data,
  and writes the output data to another file."""
  logging.debug(f"Running main(): {code_file}, {input_file}")
  with open(code_file, "rb") as f:
    func = pickle.load(f)

    with open(input_file, "rb") as input_f:
      input_data = pickle.load(input_f)

      ret = func(input_data)
      print(func, input_data, ret)



if __name__ == '__main__':
  if len(sys.argv) != 4:
    sys.exit(-1)
  main(sys.argv[1], sys.argv[2], sys.argv[3])
  # _main(sys.argv[1], sys.argv[2])
