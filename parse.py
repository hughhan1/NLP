#!/usr/bin/python

import math
import sys
from collections import defaultdict


ROOT = "ROOT"


class Rule:

    def __init__(self, prob, lhs, rhs):
        self.prob = prob
        self.weight = - math.log(prob) / math.log(2.0)
        self.lhs = lhs
        self.rhs = rhs


    def __repr__(self):
        return ("prob: {0}, weight: {1}, lhs: {2}, rhs: {3}"
                .format(self.prob, self.weight, self.lhs, self.rhs))


class RulePointer:

    def __init__(self, lhs, grammar_idx, col, dot_idx):
        self.lhs     = lhs
        self.grammar_idx = grammar_idx
        self.col     = col
        self.dot_idx = dot_idx


    def __eq__(self, other):
        return self.lhs         == other.lhs         and \
               self.grammar_idx == other.grammar_idx and \
               self.col         == other.col         and \
               self.dot_idx     == other.dot_idx


    def __hash__(self):
        return hash(self.lhs) ^ hash(self.grammar_idx) ^ \
               hash(self.col) ^ hash(self.dot_idx)


    def __repr__(self):
        return ("lhs: {0}, grammar_idx: {1}, col: {2}, dot_idx: {3} "
                .format(self.lhs, self.grammar_idx, self.col, self.dot_idx))


class Parser:
    def __init__(self, grammar_file):

        # First, construct a rule dictionary, mapping from a root symbol to a
        # list of its children symbols.
        self.grammar = defaultdict(list)

        # Next, populate the rule dictionary using a grammar file.
        self.__read_file(grammar_file)

        self.curr_rule_ptrs = set()


    def __read_file(self, filename):
        '''
        helper function for parsing grammar file
        '''
        with open(filename, 'r') as f:             # Here, we iterate through
            for line in f.readlines():             # all of the lines, and
                                                   # populate the rule
                tokens = line.split()              # dictionary of our grammar.
                prob = float(tokens[0])
                lhs, rhs = tokens[1], tokens[2:]

                if lhs not in self.grammar:
                    self.grammar[lhs] = []

                self.grammar[lhs].append(Rule(prob, lhs, rhs))


    def __build_rule_ptrs(self, symbol, col):
        possible_rules = self.grammar[symbol]
        rule_ptrs = []
        for i, rule in enumerate(possible_rules):
            rule_ptr = RulePointer(symbol, i, col, 0)
            if not rule_ptr in self.curr_rule_ptrs:
                rule_ptrs.append(rule_ptr)
                self.curr_rule_ptrs.add(rule_ptr)
        return rule_ptrs


    def get_rule(self, rule_pointer):
        return self.grammar[rule_pointer.lhs][rule_pointer.grammar_idx]


    def parse(self, sentence):

        words = sentence.split()

        # Initialize an empty table
        self.table = [[] for _ in range(len(words) + 1)]

        # First we need to append our root rule
        rule_ptrs = self.__build_rule_ptrs(ROOT, 0)
        self.table[0].extend(rule_ptrs)

        curr_col = 0
        while curr_col < len(self.table):

            # iterate through all tuples in our current column, and add all of
            # the existing rules to our set of current rules
            for r_ptr in self.table[curr_col]:
                self.curr_rule_ptrs.add(r_ptr)

            curr_row = 0
            while curr_row < len(self.table[curr_col]):

                rule_ptr = self.table[curr_col][curr_row]  # Here, we get the
                dot_idx = rule_ptr.dot_idx                 # current cell of
                rule = self.get_rule(rule_ptr)             # our table.

                if dot_idx >= len(rule.rhs):

                    # If the index of our dot (the current position in the 
                    # rule) is greater than or equal to the length of the
                    # right-hand side of the rule, then that means we've 
                    # reached the end of the rule.
                    #
                    # Thus, we need to look back to the associated column, take
                    # all relevant entries, shift over the dot_idx by 1 to the
                    # right, and copy them into the current column.
                    #
                    # The relevant entries are the entries that contain the
                    # current entry's left hand side as the dot_idx-th symbol
                    # in the associated entry's right hand side.

                    back_col = rule_ptr.col        # First, retrieve the actual
                    column = self.table[back_col]  # associated column (list).

                    for r_ptr in column:           # Now iterate through all
                                                   # entries in the column.
                        d = r_ptr.dot_idx
                        r = self.get_rule(r_ptr)

                        if d < len(r.rhs) and r.rhs[d] == rule.lhs:
                            updated_rule_ptr = RulePointer(r_ptr.lhs, r_ptr.grammar_idx, r_ptr.col, d+1)
                            if updated_rule_ptr not in self.curr_rule_ptrs:
                                self.table[curr_col].append(updated_rule_ptr)
                                self.curr_rule_ptrs.add(updated_rule_ptr)

                else:

                    # Otherwise, we can check if the current symbol is a
                    # terminal or non-terminal, and then go from there.

                    symbol = rule.rhs[dot_idx]
                    if symbol not in self.grammar:     

                        # TERMINAL SYMBOL
                        # If our symbol is not a key in our grammar, then it 
                        # must be a terminal. If it is a terminal, then we try 
                        # to match the current symbol with the corresponding 
                        # symbol in our sentence.

                        if curr_col >= len(words):
                            # TODO: ask what to do in this case
                            pass
                        elif words[curr_col] == symbol:
                            next_col = curr_col + 1
                            updated_rule_ptr = RulePointer(rule_ptr.lhs, rule_ptr.grammar_idx, rule_ptr.col, dot_idx + 1)
                            self.table[next_col].append(updated_rule_ptr)

                    else:                               
                        # NON-TERMINAL SYMBOL
                        # If our symbol is a key in our grammar, then it must
                        # be a nonterminal. We unravel the symbol, and then 
                        # append its children to our current column.

                        rule_ptrs = self.__build_rule_ptrs(symbol, curr_col)
                        self.table[curr_col].extend(rule_ptrs)

                print "(%d, %d)" % (curr_col, curr_row)
                
                curr_row += 1

            curr_col += 1
                                         # After finishing our current column 
            self.curr_rule_ptrs = set()  # and advancing to the next, we need 
                                         # to clear our list of rule pointers 
                                         # that were already used.
def main():

    parser = Parser("papa.gr")

    parser.parse("Papa ate the caviar with the spoon")

main()

