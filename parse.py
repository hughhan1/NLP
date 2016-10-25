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
        return "prob: {0}, weight: {1}, lhs: {2}, rhs: {3}".format(self.prob, self.weight, self.lhs, self.rhs)


    def __str__(self):
        return "\{ prob: {0}, weight: {1}, lhs: {2}, rhs: {3} \}".format(self.prob, self.weight, self.lhs, self.rhs)


class RulePointer:
    def __init__(self, lhs, idx, col):
        self.lhs = lhs
        self.idx = idx
        self.col = col

    def __repr__(self):
        return "lhs: {0}, idx: {1}, col: {2}".format(self.lhs, self.idx, self.col)


class Parser:
    def __init__(self, grammar_file):

        # First, construct a rule dictionary, mapping from a root symbol to a
        # list of its children symbols.
        self.grammar = defaultdict(list)

        # Next, populate the rule dictionary using a grammar file.
        self.__read_file(grammar_file)

        self.curr_rules = set()


    def __read_file(self, filename):
        '''
        helper function for parsing grammar file
        '''
        with open(filename, 'r') as f:  # Iterate through all of the
            for line in f.readlines():  # lines, and populate the
                # rule dictionary of our
                tokens = line.split()  # grammar.
                prob = float(tokens[0])
                lhs, rhs = tokens[1], tokens[2:]

                if lhs not in self.grammar:
                    self.grammar[lhs] = []

                self.grammar[lhs].append(Rule(prob, lhs, rhs))


    def __build_tuples(self, symbol, col):
        possible_rules = self.grammar[symbol]
        tuples = []
        for i, rule in enumerate(possible_rules):
            if not rule in self.curr_rules:
                rule_ptr = RulePointer(symbol, i, col)
                tuples.append((0, rule_ptr))
                #self.curr_rules.add(rule_ptr)
                self.curr_rules.add(rule)
        return tuples


    def get_rule(self, rule_pointer):
        return self.grammar[rule_pointer.lhs][rule_pointer.idx]


    def parse(self, sentence):

        words = sentence.split()

        # Initialize an empty table
        self.table = [[] for _ in range(len(words) + 1)]

        # First we need to append our root rule
        tuples = self.__build_tuples(ROOT, 0)
        self.table[0].extend(tuples)

        curr_col = 0
        while curr_col < len(self.table):

            curr_row = 0
            while curr_row < len(self.table[curr_col]):

                dot_idx, rule_pointer = self.table[curr_col][curr_row]
                rule = self.get_rule(rule_pointer)
                if dot_idx >= len(rule.rhs):

                    # Use the rule_pointer.col attribute to go back to the correct
                    # column. Then iterate through that column, and copy over all
                    # relevant rule_pointers (+1 to the dot_idx) to this column.
                    #
                    # The relevant rule_pointers are going to be the ones that have
                    # rule

                    back_col = rule_pointer.col
                    column = self.table[back_col]
                    for d, r_ptr in column:
                        # d is the dot index for each element
                        # r is the rule pointer
                        r = self.get_rule(r_ptr)
                        if d < len(r.rhs) and r.rhs[d] == rule.lhs:
                            updated_tuple = (d + 1, r_ptr)
                            self.table[curr_col].append(updated_tuple)

                else:
                    # If our symbol is not a key in our grammar, then it must
                    # be a terminal. If it is a terminal, then we try to match
                    # the current symbol with the corresponding symbol in our
                    # sentence.
                    #
                    # If our symbol is in a key in our grammar, then it must
                    # be a nonterminal. In this case, we continue to unravel
                    # the symbol, and then append its children to our current
                    # column.

                    symbol = rule.rhs[dot_idx]
                    if not symbol in self.grammar:
                    	if curr_col >= len(words):
                    		pass
                    		# TODO: ask what to do in this case
                        elif words[curr_col] == symbol:
                            next_col = curr_col + 1
                            updated_tuple = (dot_idx + 1, rule_pointer)
                            self.table[next_col].append(updated_tuple)
                    else:
                        tuples = self.__build_tuples(symbol, curr_col)
                        self.table[curr_col].extend(tuples)

                print "(col: %d, row: %d)" % (curr_col, curr_row)
                curr_row += 1

            self.curr_rules = set()
            curr_col += 1

        print self.table


def main():

    parser = Parser("papa.gr")

    parser.parse("Papa ate the caviar")

main()

