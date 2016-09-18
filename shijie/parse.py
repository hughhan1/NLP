from collections import defaultdict
import os
import sys
import random
import argparse

def weighted_sample(choices, weights):
    '''
    sample from choices propotion to weights
    '''
    assert len(choices) == len(weights), 'length of choices and weights should be equal'
    sum_of_weights = sum(weights)
    r = random.uniform(0, sum_of_weights)
    upto = 0
    for c, w in zip(choices, weights):
        if upto + w >= r:
            return c
        upto += w
    return choices[-1]

class Grammar:
    def __init__(self, grammar_file):
        # key is LHS, value (list) is a list of RHSs, each RHS is a list
        self.rules = defaultdict(list)
        # key is LHS, value (list) is a list of float
        self.prob_of_rules = defaultdict(list)
        self.read_file(grammar_file) # parse grammar file

    def read_file(self, filename):
        '''
        helper function for parsing grammar file
        '''
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line[:-1] # cleanup '\n'
                line = line.split('#', 1)[0] # cleanup comment
                tokens = line.split()
                if tokens == []: # skip empty line
                    continue
                prob, lhs, rhs = tokens[0], tokens[1], tokens[2:]
                self.rules[lhs].append(rhs)
                self.prob_of_rules[lhs].append(float(prob))

    def generate(self, rule_process_callback, posprocess_callback):
        '''
        helper function for generating sentence or tree
        '''
        stack = []
        sentence = []
        stack.append('ROOT')
        while len(stack) > 0:
            lhs = stack.pop()
            if lhs not in self.rules: # handle terminal symbols
                sentence.append(lhs)
                continue
            rhs = self.rules[lhs] # list of possible rhs(list)
            # get list of probability of corresponding rhs
            prob_of_rhs = self.prob_of_rules[lhs]
            # sample one RHS from all RHSs based on probability
            chosen_rule = weighted_sample(rhs, prob_of_rhs)
            processed_rule = rule_process_callback(chosen_rule, lhs)
            reversed_rule = processed_rule[::-1]
            stack.extend(reversed_rule)

        return posprocess_callback(' '.join(sentence))

    def gen_sentence(self):
        rule_process_callback = lambda rule, lhs: rule
        posprocess_callback = lambda sentence: sentence
        return self.generate(rule_process_callback, posprocess_callback)

    def structure_posprocess(self, sentence):
        # clean up '#' introduced by internal symbol and unnecessary space
        return sentence.replace(' #', '').replace('# ', '')

    def gen_sentence_in_tree(self):
        def rule_process_callback(rule, lhs):
            # contain '#' as a special internal symbol
            return ['(# ' + lhs] + rule + ['#)']
        return self.generate(rule_process_callback, self.structure_posprocess)

    def gen_sentence_in_bracket(self):
        def rule_process_callback(rule, lhs):
            # contain '#' as a special internal symbol
            if lhs == 'S':
                return ['{#'] + rule + ['#}']
            elif lhs == 'NP':
                return ['[#'] + rule + ['#]']
            else:
                return rule
        return self.generate(rule_process_callback, self.structure_posprocess)

def main():
    parser = argparse.ArgumentParser(description='Generate some sentences.')
    parser.add_argument('filename', help='the path and filename of the grammar file')
    parser.add_argument('num_of_sentences', help='the number of generated sentence',
                        default=1, type=int, nargs='?')
    print_option = parser.add_mutually_exclusive_group()
    print_option.add_argument('-t', help='print tree instead', action='store_true')
    print_option.add_argument('-b', help='print structure instead', action='store_true')
    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        print 'grammar file doesn\'t exist'
        return

    grammar = Grammar(args.filename)
    for _ in xrange(args.num_of_sentences):
        if args.t: # print sentence in tree structure
            print grammar.gen_sentence_in_tree()
        elif args.b: # print sentence in bracket 
            print grammar.gen_sentence_in_bracket()
        else:
            print grammar.gen_sentence()

if __name__ == '__main__':
    main()
