#!/usr/bin/python

import math
import random
import sys
from collections import defaultdict


ROOT = "ROOT"


class Rule:

    def __init__(self, prob, lhs, rhs):
        self.prob = prob
        self.weight = - math.log(prob) / math.log(2.0)
        self.lhs = lhs
        self.rhs = rhs

    def __eq__(self, other):
        return isinstance(other, Rule) and \
               self.prob         == other.prob and \
               self.weight == other.weight and \
               self.lhs         == other.lhs and \
               self.rhs     == other.rhs

    def __repr__(self):
        return ("prob: {0}, weight: {1}, lhs: {2}, rhs: {3}"
                .format(self.prob, self.weight, self.lhs, self.rhs))


class TerminalEntry:

    def __init__(self, lhs, weight=0):
        self.lhs = lhs
        self.rhs = []
        self.weight = weight
        self.back_ptrs = []


class TableEntry:

    def __init__(self, lhs, grammar_idx, col, dot_idx, weight=None):
        self.lhs         = lhs
        self.grammar_idx = grammar_idx
        self.col         = col
        self.dot_idx     = dot_idx
        self.back_ptrs   = []
        self.weight      = weight


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
        self.prefix_table = {}
        self.left_parent_table = {}
        self.left_ancestor_pair_table = {}
        # Next, populate the rule dictionary using a grammar file.
        self.__read_file(grammar_file)

        self.existing_entries = {}
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

                if self.is_terminal(lhs):
                    self.grammar[lhs] = []

                if (lhs, rhs[0]) not in self.prefix_table:
                    self.prefix_table[(lhs, rhs[0])] = []
                # self.prefix_table[(lhs, rhs[0])].append(Rule(prob, lhs, rhs))


                if rhs[0] not in self.left_parent_table:
                    self.left_parent_table[rhs[0]] = []


                if len(self.prefix_table[(lhs, rhs[0])]) == 0:
                    self.left_parent_table[rhs[0]].append(lhs)
                    self.prefix_table[(lhs, rhs[0])].append(Rule(prob, lhs, rhs))

                self.grammar[lhs].append(Rule(prob, lhs, rhs))


    def fill_left_ancestor_pair_table(self, token):
        self.left_ancestor_pair_table = self.dfs(token, self.left_ancestor_pair_table)

    def dfs(self, y, sj):

        if y in self.left_parent_table:
            for x in self.left_parent_table[y]:
                if x not in sj:
                    sj[x] = []
                    sj[x].append(y)
                    self.dfs(x, sj)
                else:
                    sj[x].append(y)
            return sj
        else:
            return sj

        '''
        for x in self.left_parent_table[y]:
            if x not in sj:
                sj[x] = []
                sj[x].append(y)
                self.dfs(x, sj)
            else:
                sj[x].append(y)
        return sj
        '''
    def get_grammar_index(self, symbol, rule):
        possible_rules = self.grammar[symbol]

        for i, g_rule in enumerate(possible_rules):
            if rule == g_rule:
                return i


    def __opredict(self, symbol, col):
        entries = []
        if symbol not in self.existing_lhs and symbol in self.left_ancestor_pair_table:
            self.existing_lhs.add(symbol)

            for b in self.left_ancestor_pair_table[symbol]:
                rules = self.prefix_table[(symbol, b)]
                for rule in rules:
                    entry = TableEntry(symbol, self.get_grammar_index(symbol, rule), col, 0, rule.weight)
                    if entry not in self.existing_entries:
                        entries.append(entry)
                        self.existing_entries[entry] = entry.weight

        return entries

    def __predict(self, symbol, col, sentence_token=None):

        entries = []
        desired_lhs = ""
        if symbol not in self.existing_lhs:
            self.existing_lhs.add(symbol)
            possible_rules = self.grammar[symbol]
            for i, rule in enumerate(possible_rules):

                # If the current rule contains a terminal, but that terminal
                # doesn't match what our next sentence word is, then we don't
                # add it to our table.
                s = rule.rhs[0]
                if self.is_terminal(s) and s != sentence_token:
                    continue
                elif self.is_terminal(s) and s == sentence_token:
                    desired_lhs = rule.lhs

                if not rule.rhs[0] == desired_lhs:
                    entry = TableEntry(symbol, i, col, 0, rule.weight)
                    if entry not in self.existing_entries:
                        entries.append(entry)
                        self.existing_entries[entry] = entry.weight
        
        return entries


    def is_terminal(self, symbol):
        return symbol not in self.grammar


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
                           .format(entry.col, rule.lhs, rule.rhs,
                                   entry.dot_idx, entry.weight))
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
        entries = self.__predict(ROOT, 0)
        # entries = self.__opredict(ROOT, 0)
        self.table[0].extend(entries)
        curr_col = 0
        while curr_col < len(self.table):

            # iterate through all entries in our current column, and add all of
            # the existing rules to our set of current rules
            for entry in self.table[curr_col]:
                self.existing_entries[entry] = entry.weight

            if not curr_col == len(words):
                self.fill_left_ancestor_pair_table(words[curr_col])

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

                        # We can look for the correct rule to attach by
                        # comparing the symbol after the dot index.
                        if d < len(r.rhs) and r.rhs[d] == rule.lhs:

                            # Obtain the total weight of the previous entry
                            # (which is the entry being attached) and the
                            # weight of the current rule.

                            weight = e.weight + entry.weight

                            # Next we need to create an updated entry using the
                            # following steps:
                            #
                            #   (1) Create a copy the previous entry, and shift
                            #       its dot index to the right by 1,
                            #   (2) Copy the previous entry's back pointers
                            #   (3) Keep a back pointer the entry representing
                            #       the rule that is a child of this one.

                            new_entry = TableEntry(e.lhs, e.grammar_idx, 
                                                   e.col, d + 1, weight)
                            new_entry.back_ptrs.extend(e.back_ptrs)
                            new_entry.back_ptrs.append(entry)

                            if new_entry not in self.existing_entries:

                                # If the new entry doesn't exist in the current
                                # column, we can simply add it to our table.

                                # Note that the new_entry is technically
                                # equivalent to the old one, using our
                                # overloaded __eq__ function.

                                self.table[curr_col].append(new_entry)
                                self.existing_entries[new_entry] = new_entry.weight

                            else:

                                # If the entry already has been added, we need
                                # to keep the entry with the lower weight,
                                # which denotes the higher probability.

                                existing_weight = self.existing_entries[new_entry]

                                if (
                                    (new_entry.weight < existing_weight) or
                                    (new_entry.weight == existing_weight and 
                                        random.randint(0, 1) == 1
                                    )
                                ):
                                    # If the new entry is more probable, delete
                                    # the old one and add the new one.
                                    #
                                    # If we have a tie in likelihood, we don't 
                                    # want to choose deterministically. So we 
                                    # flip a coin, and pick at random.

                                    self.existing_entries[new_entry] = new_entry.weight

                                    temp_idx = self.table[curr_col].index(new_entry)

                                    self.table[curr_col][temp_idx] = None
                                    self.table[curr_col].append(new_entry)

                    # Finally, we need to check if the last symbol of our entry
                    # is a terminal symbol. If it is, then we can keep a
                    # pointer to a terminal symbol to be printed.
                    #
                    # But we need to be careful. We only do this when the entry
                    # is not a leaf. Otherwise, it will get printed, and we
                    # don't want to double-print.

                    last_symbol = rule.rhs[len(rule.rhs) - 1]

                    if entry.back_ptrs and self.is_terminal(last_symbol):
                        t = TerminalEntry(rule.rhs[len(rule.rhs) - 1])
                        entry.back_ptrs.append(t)

                else:

                    symbol = rule.rhs[dot_idx]      # Get the symbol, and check
                    if self.is_terminal(symbol):    # if it is a terminal

                        # SCAN:
                        #   If our symbol is a terminal, then we try to match
                        #   current symbol with the corresponding symbol in our
                        #   sentence.

                        if curr_col >= len(words):
                            # TODO: ask what to do in this case
                            pass
                        elif words[curr_col] == symbol:
                            next_col = curr_col + 1
                            new_entry = TableEntry(entry.lhs, entry.grammar_idx, entry.col, dot_idx + 1, entry.weight)
                            new_entry.back_ptrs.extend(entry.back_ptrs)

                            if dot_idx + 1 < len(rule.rhs):
                                # If this happens, it must be that we scanned some terminal
                                # symbol, but we haven't advanced the dot index to the end
                                # of the rhs yet.
                                #
                                # Now, we need to make a temporary TableEntry, and point a
                                # backpointer to that, which will represent the terminal
                                # symbol.

                                t = TerminalEntry(rule.rhs[dot_idx])
                                new_entry.back_ptrs.append(t)

                            self.table[next_col].append(new_entry)

                    else:  

                        # PREDICT:
                        #   If our symbol is a key in our grammar, then it must
                        #   be a nonterminal. We unravel the symbol, and then 
                        #   append its children to our current column.

                        if curr_col != len(words):
                            # entries = self.__predict(symbol, curr_col, words[curr_col])
                            entries = self.__opredict(symbol, curr_col)
                        else:
                            # entries = self.__predict(symbol, curr_col, None)
                            entries = self.__opredict(symbol, curr_col)
                        self.table[curr_col].extend(entries)
                
                curr_row += 1

            curr_col += 1
                                           # After finishing our current
            self.existing_entries = {}     # column and advancing to the next,
            self.existing_lhs     = set()  # we need to clear our set of
            self.left_ancestor_pair_table = {}                         # entries that were already used.

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
    '''
    if len(sys.argv) != 3:
        return

    grammar_file  = sys.argv[1]
    sentence_file = sys.argv[2]
    '''
    parser = Parser("papa.gr")
    # parser = Parser(grammar_file)
    parser.parse_sentence("Papa ate the caviar with the spoon")
    # parser.parse(sentence_file)

    # parser = Parser("papa.gr")
    # parser.parse_sentence("Papa ate the caviar with the spoon")


if __name__ == "__main__":
    main()
