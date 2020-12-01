import networkx as nx
import numpy as np
import pandas as pd
import random


def generate_network(n_attacks=2, n_subsystems=3, n_workstations=6):
  """
  Make an example network with the number of components.

  Args:
    n_attacks (int): number of "attack" random variables
    n_subsystems (int): number of "subsystem" random variables, assuming a single subsystem layer
    n_workstations (int): number of "workstation" random variables

  Returns:
    BayesNet: a bayes net with the specified random variables.

  """
  # i am also building a graph that could be used to represent the bayes net at the same time
  G=nx.DiGraph()
  graph_width = max(n_attacks, n_subsystems, n_workstations)


  # handle inadequate inputs
  if n_attacks == 0:
    raise ValueError("Need at least one attack!")
  if n_workstations == 0:
    raise ValueError("Need at least one workstation!")

  # generate attack nodes
  attack_nodes = []
  for i in range(n_attacks):
    G.add_node('a'+str(i),
               pos = (graph_width * (i+0.5)/n_attacks, 4),
               color = 'r')
    attack_nodes.append( Node('a'+str(i),
                              generate_random_cpt( [] )
                              ) )
  # generate subsystem nodes
  subsystem_nodes = []
  for i in range(n_subsystems):
    # a random sample of 1 through n attacks
    G.add_node('s'+str(i),
               pos = (graph_width * (i+0.5)/n_subsystems, 2),
               color = 'b')
    sample_attacks = random.sample(range(0, n_attacks), random.randint(1, n_attacks))
    for j in sample_attacks:
        G.add_edge('a' + str(j), 's' + str(i))
    subsystem_nodes.append(Node('s'+str(i),
                                generate_random_cpt(['a' + str(j) for j in sample_attacks ]),
                                parents = [attack_nodes[j] for j in sample_attacks]
                                ) )

  # generate workstation nodes
  workstation_nodes = []
  for i in range(n_workstations):
    G.add_node('w'+str(i),
               pos = (graph_width * (i+0.5)/n_workstations, 0),
               color = '')

    if n_subsystems != 0:
      sample_attacks = random.sample(range(0, n_attacks), random.randint(0, n_attacks//2)) # sample some attacks
      sample_subsystems = random.sample(range(0, n_subsystems), random.randint(1, n_subsystems)) # sample at least one subsystem
      for j in sample_attacks:
        G.add_edge('a' + str(j), 'w' + str(i))
      for j in sample_subsystems:
        G.add_edge('s' + str(j), 'w' + str(i))
      subsystem_nodes.append(Node('w'+str(i),
                                  generate_random_cpt(['a' + str(j) for j in sample_attacks ] + ['s' + str(j) for j in sample_subsystems ]),
                                  [attack_nodes[j] for j in sample_attacks] + [subsystem_nodes[j] for j in sample_subsystems]
                                  ) )
    else:
      # there are no subsystems, sample just attacks
      sample_attacks = random.sample(range(0, n_attacks), random.randint(1, n_attacks))
      for j in sample_attacks:
        G.add_edge('a' + str(j), 'w' + str(i))
      workstation_nodes.append(Node('w'+str(i),
                                generate_random_cpt(['a' + str(j) for j in sample_attacks ]),
                                [attack_nodes[j] for j in sample_attacks]
                                ) )

  dictionary_of_nodes = dict()
  all_nodes = attack_nodes + subsystem_nodes + workstation_nodes
  for n in all_nodes:
    dictionary_of_nodes[n.name] = n
  return BayesNet(dictionary_of_nodes, G)



def generate_random_cpt(parents=[]):
  """
  Make a random cpt from from a given list of parents, assuming that the value of nodes are either true or false.

  Args:
    parents (list of str): list of parents; can be empty

  Returns:
    pandas dataframe: a cpt in a dataframe representation;

  """
  df = pd.DataFrame(columns = parents + ['prob'])
  ar = [False] * len(parents)
  # we need to create 2^parents rows - the cpt table.
  # these few lines generate all combinations of false / true statements
  for i in range(2**len(parents)):
    df.loc[i] = ar + [random.random()]
    j = 0
    while j < len(parents) and True:
      if ar[j] == False:
        ar[j] = True
        ar[0:j] = [False] * (j)
        break
      else:
        j+=1
  return df


