from collections import defaultdict
import os
import sys
import random
import argparse

def weighted_sample(choices, weights):
    sum_of_weights = sum(weights)
    r = random.uniform(0, sum_of_weights)
    upto = 0
    for c, w in zip(choices, weights):
        if upto + w >= r:
            return c
        upto += w
    return choices[-1]

def read_file(filename):
    '''
    Return Value:
    rules - defaultdict(list), key is LHS, value (list) is a list of RHSs, each
            RHS is a list
    prob_of_rules - defaultdict(list), key is LHS, value (list) is a list of float
    '''
    rules = defaultdict(list)
    prob_of_rules = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line[:-1] # cleanup '\n'
            line = line.split('#', 1)[0] # cleanup comment
            line = line.strip() # cleanup left/right space
            if line == '': # skip empty line
                continue
            tokens = line.split()
            prob, lhs, rhs = tokens[0], tokens[1], tokens[2:]
            rules[lhs].append(rhs)
            prob_of_rules[lhs].append(float(prob))
    return rules, prob_of_rules

def generate_sentence(rules, prob_of_rules, print_tree=False, print_structure=False):
    '''
    Input Value:
    rules - must be the return of read_file
    '''
    assert not(print_tree and print_structure), 'only print in 1 format'
    stack = []
    sentence = []
    stack.append('ROOT')
    while len(stack) > 0:
        lhs = stack.pop()
        if lhs not in rules: # handle terminal symbols
            sentence.append(lhs)
            continue
        rhs = rules[lhs]
        prob_of_rhs = prob_of_rules[lhs]
        # sample one RHS from all RHSs based on probability
        chosen_rule = weighted_sample(rhs, prob_of_rhs)
        reversed_rule = chosen_rule[::-1]

        if print_tree:
            # contain '#' as a special internal symbol
            reversed_rule = ['#)'] + reversed_rule + ['(# ' + lhs]
        if print_structure:
            # same as above
            if lhs == 'S':
                reversed_rule = ['#}'] + reversed_rule + ['{#']
            if lhs == 'NP':
                reversed_rule = ['#]'] + reversed_rule + ['[#']
        stack.extend(reversed_rule)

    # sentence post process
    if print_tree or print_structure:
        # clean up '#' introduced by internal symbol and unnecessary space
        return ' '.join(sentence).replace(' #', '').replace('# ', '')
    return ' '.join(sentence)

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

    rules, prob_of_rules = read_file(args.filename)
    for _ in xrange(args.num_of_sentences):
        print generate_sentence(rules, prob_of_rules, args.t, args.b)

main()
