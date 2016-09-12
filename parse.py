from collections import defaultdict
import os
import sys
import random

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
            prob, lhs, rhs = line.split('\t', 2)
            rules[lhs].append(rhs.split(' '))
            prob_of_rules[lhs].append(float(prob))
    return rules, prob_of_rules

def generate_sentence(rules, prob_of_rules):
    '''
    Input Value:
    rules - must be the return of read_file
    '''
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
        stack.extend(reversed_rule)
    return ' '.join(sentence)

def main():
    if not os.path.isfile(sys.argv[1]):
        print 'grammar file doesn\'t exist'
        return
    filename = sys.argv[1]
    
    num_of_sentences = 1
    try:
        num_of_sentences = int(sys.argv[2])
    except:
        pass

    rules, prob_of_rules = read_file(filename)
    for _ in xrange(num_of_sentences):
        print generate_sentence(rules, prob_of_rules)

main()
