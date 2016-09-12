from collections import defaultdict
import os
import sys
import random

def read_file(filename):
    '''
    Return Value:
    rules - defaultdict(list), key is LHS, value(list) is a list of RHSs, each
            RHS is a list
    '''
    rules = defaultdict(list) # dist -> list -> list
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line[:-1] # cleanup '\n'
            line = line.split('#', 1)[0] # cleanup comment
            line = line.strip() # cleanup left/right space
            if line == '': # skip empty line
                continue
            _, lhs, rhs = line.split('\t', 2)
            rules[lhs].append(rhs.split(' '))
    return rules

def generate_sentence(rules):
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
        chosen_rule = random.choice(rhs) # sample one RHS from all RHSs
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

    rules = read_file(filename)
    for _ in xrange(num_of_sentences):
        print generate_sentence(rules)

main()
