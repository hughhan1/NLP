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


class TerminalEntry:

    def __init__(self, lhs, weight=0, curr_prob=1):
        self.lhs = lhs
        self.rhs = []
        self.weight = weight
        self.curr_prob = curr_prob
        self.back_ptrs = []


class TableEntry:

    def __init__(self, lhs, grammar_idx, col, dot_idx, weight=None, curr_prob=None):
        self.lhs         = lhs
        self.grammar_idx = grammar_idx
        self.col         = col
        self.dot_idx     = dot_idx
        self.back_ptrs   = []
        self.weight      = weight
        self.curr_prob   = curr_prob


    def __eq__(self, other):
        return isinstance(other, TableEntry)         and \
               self.lhs         == other.lhs         and \
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

        self.existing_entries = set()
        self.existing_lhs     = set()


    def __read_file(self, filename):
        '''
        helper function for parsing grammar file
        '''

        counter = 0
        with open(filename, 'r') as f:             # Here, we iterate through
            for line in f.readlines():             # all of the lines, and
                                                   # populate the rule
                tokens = line.split()              # dictionary of our grammar.
                prob = float(tokens[0])
                lhs, rhs = tokens[1], tokens[2:]

                if lhs not in self.grammar:
                    self.grammar[lhs] = []

                self.grammar[lhs].append(Rule(prob, lhs, rhs))


    def __build_entries(self, symbol, col):

        entries = []

        if symbol not in self.existing_lhs:
            self.existing_lhs.add(symbol)
            possible_rules = self.grammar[symbol]
            for i, rule in enumerate(possible_rules):
                entry = TableEntry(symbol, i, col, 0, rule.weight, rule.prob)
                if entry not in self.existing_entries:
                    entries.append(entry)
                    self.existing_entries.add(entry)
        
        return entries


    def print_table(self):
        '''
        Prints each column of the table in the following format

        COL   LHS       RHS      DOT-INDEX
         3     N  -> ['caviar']     (1)
         2     NP -> ['Det', 'N']   (2)
         1     VP -> ['V', 'NP']    (2)
         0     S  -> ['NP', 'VP']   (2)
         4     P  -> ['with']       (0)
        '''
        col_idx = 0

        for col in self.table:
            print "COLUMN %d" % (col_idx)
            for entry in col:
                if entry == None:
                    print "  [ -- None -- ]"
                else:
                    rule = self.get_rule(entry)
                    print ("{0} {1}\t-> {2}\t({3})\t({4})"
                           .format(entry.col, rule.lhs, rule.rhs, entry.dot_idx, entry.weight))


            print ""
            col_idx += 1


    def get_rule(self, entry):
        if entry == None:
            return None
        return self.grammar[entry.lhs][entry.grammar_idx]


    def parse(self, sentence_file):

        with open(sentence_file, 'r') as f:
            for line in f.readlines():
                self.parse_sentence(line)


    def parse_sentence(self, sentence):

        words = sentence.split()

        # Initialize an empty table
        self.table = [[] for _ in range(len(words) + 1)]

        # First we need to append our root rule
        entries = self.__build_entries(ROOT, 0)
        self.table[0].extend(entries)
        curr_col = 0
        while curr_col < len(self.table):

            # iterate through all entries in our current column, and add all of
            # the existing rules to our set of current rules
            for entry in self.table[curr_col]:
                self.existing_entries.add(entry)

            curr_row = 0
            while curr_row < len(self.table[curr_col]):

                entry = self.table[curr_col][curr_row]  # Here, we get the

                if entry == None:
                    curr_row += 1
                    continue

                dot_idx = entry.dot_idx                 # current cell of
                rule = self.get_rule(entry)             # our table.

                if dot_idx >= len(rule.rhs):

                    # ATTACH
                    #
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

                    back_col = entry.col           # First, retrieve the actual
                    column = self.table[back_col]  # associated column (list).

                    for e in column:               # Now iterate through all
                                                   # entries in the column.
                        if e == None:
                            continue

                        d = e.dot_idx
                        r = self.get_rule(e)

                        if d < len(r.rhs) and r.rhs[d] == rule.lhs:

                            # Obtain the total weight of the previous entry
                            # (which is the entry being attached) and the
                            # weight of the current rule.

                            prob   = e.curr_prob * entry.curr_prob
                            weight = e.weight + entry.weight

                            updated_entry = TableEntry(e.lhs, e.grammar_idx, e.col, d+1, weight, prob)
                            updated_entry.back_ptrs.extend(e.back_ptrs)
                            updated_entry.back_ptrs.append(entry)

                            if updated_entry not in self.existing_entries:
                                self.table[curr_col].append(updated_entry)
                                self.existing_entries.add(updated_entry)
                            else:
                                for existing_entry in self.existing_entries:
                                    if existing_entry == updated_entry and \
                                       updated_entry.weight <= existing_entry.weight:
                                        self.existing_entries.remove(existing_entry)
                                        self.existing_entries.add(updated_entry)

                                        temp_idx = self.table[curr_col].index(existing_entry)

                                        self.table[curr_col][temp_idx] = None
                                        self.table[curr_col].append(updated_entry)
                                        break
                                    # if existing_entry == updated_entry and \
                                    #    updated_entry.curr_prob > existing_entry.curr_prob:
                                    #     self.existing_entries.remove(existing_entry)
                                    #     self.existing_entries.add(updated_entry)

                                    #     temp_idx = self.table[curr_col].index(existing_entry)

                                    #     self.table[curr_col][temp_idx] = None
                                    #     self.table[curr_col].append(updated_entry)
                                    #     break

                    last_symbol = rule.rhs[len(rule.rhs) - 1]
                    if entry.back_ptrs and last_symbol not in self.grammar:
                        t = TerminalEntry(rule.rhs[len(rule.rhs) - 1])
                        entry.back_ptrs.append(t)

                else:

                    # Otherwise, we can check if the current symbol is a
                    # terminal or non-terminal, and then go from there.

                    symbol = rule.rhs[dot_idx]
                    if symbol not in self.grammar:     

                        # SCAN
                        #
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
                            updated_entry = TableEntry(entry.lhs, entry.grammar_idx, entry.col, dot_idx + 1, entry.weight, entry.curr_prob)
                            updated_entry.back_ptrs.extend(entry.back_ptrs)

                            if dot_idx + 1 < len(rule.rhs):
                                # If this happens, it must be that we scanned some terminal
                                # symbol, but we haven't advanced the dot index to the end
                                # of the rhs yet.
                                #
                                # Now, we need to make a temporary TableEntry, and point a
                                # backpointer to that, which will represent the terminal
                                # symbol.

                                t = TerminalEntry(rule.rhs[dot_idx])
                                updated_entry.back_ptrs.append(t)

                            self.table[next_col].append(updated_entry)

                    else:  

                        # PREDICT
                        #                             
                        # NON-TERMINAL SYMBOL
                        # If our symbol is a key in our grammar, then it must
                        # be a nonterminal. We unravel the symbol, and then 
                        # append its children to our current column.

                        entries = self.__build_entries(symbol, curr_col)
                        self.table[curr_col].extend(entries)
                
                curr_row += 1

            curr_col += 1
                                           # After finishing our current
            self.existing_entries = set()  # column and advancing to the next,
            self.existing_lhs     = set()  # we need to clear our set of
                                           # entries that were already used.

        # self.print_table()                 # print the finished table

        temp = TableEntry(ROOT, 0, 0, 1)

        res = None
        for tab_entry in self.table[curr_col - 1]:
            if tab_entry == temp:
                res = tab_entry
        if res is None:
            print "NONE"
        else:
            self.print_parse(res)
            print res.weight


    def print_parse(self, entry):
        '''
        (ROOT (rhs[0] (rhs'[0])
                      (rhs'[1]))
              (rhs[1] (rhs''[0]))
        )
        '''

        if isinstance(entry, TerminalEntry):
            sys.stdout.write(" {0}".format(entry.lhs))
        else:
            rule = self.get_rule(entry)
            sys.stdout.write("({0} ".format(rule.lhs))

            if not entry.back_ptrs:
                sys.stdout.write(" {0}".format(rule.rhs[0]))

            for e in entry.back_ptrs:
                self.print_parse(e)

            sys.stdout.write(")")


def main():

    if len(sys.argv) != 3:
        return

    grammar_file  = sys.argv[1]
    sentence_file = sys.argv[2]

    parser = Parser(grammar_file)
    parser.parse(sentence_file)
    # parser.parse_sentence("{ 3 }")


if __name__ == "__main__":
    main()
