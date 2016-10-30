import math
from collections import defaultdict

class Rule:

    def __init__(self, prob, lhs, rhs):
        self.prob   = prob
        self.weight = - math.log(prob) / math.log(2.0)
        self.lhs    = lhs
        self.rhs    = rhs


class Grammar:
    '''
    for the following grammar file:
        1 S NP VP
        1 VP signed
        2 VP find it
        1 NP I
    the parsed grammar will be:
    rules = {'S': [['NP', 'VP']],
            'VP': [['signed'], ['find', 'it']],
            'NP': [['I']]}
    prob_of_rules = {'S' : [1],
                     'VP': [1, 2],
                     'NP': [1]}
    '''
    def __init__(self, grammar_file):

        # First, construct a rule dictionary, mapping from a root symbol to a
        # list of its children symbols.
        self.rules = defaultdict(list) 

        # Next, populate the rule dictionary using a grammar file.
        self.__read_file(grammar_file)


    def __read_file(self, filename):
        '''
        helper function for parsing grammar file
        '''
        with open(filename, 'r') as f:            # Iterate through all of the
            for line in f.readlines():            # lines, and populate the
                                                  # rule dictionary of our
                tokens   = line.split()           # grammar.
                prob     = float(tokens[0])
                lhs, rhs = tokens[1], tokens[2:]

                self.rules[lhs].append(Rule(prob, lhs, rhs))
