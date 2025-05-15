def evolve(func):
  """@blade.evolve decorator is used in the problem specification to detect the method that
  should be evolved using LLM.
  """
  return func


def run(func):
  """@blade.run decorator is used in the problem specification to detect the method that
  should be used to verify and grade generated code.
  """
  return func
