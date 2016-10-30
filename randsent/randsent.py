#!/usr/bin/python

from collections import defaultdict
import os
import sys
import random
import argparse

def myPrettyPrint(inString):
    '''
    take a string that generated from SubSentence.getTree(), such as:
    (ROOT (S (NP (Det every) (Noun president)) (VP (V_intran jumped))) .)
    and return a pretty string like
    (ROOT (S (NP (Det every) 
                 (Noun president)) 
             (VP (V_intran jumped)))
          .)
    
    does not support comment (#).
    implies that the input is right about left and right parenthesis.
    '''
    def hasLeftPar(word):
        '''
        is this word starts with '('?
        a helper function of myPrettyPrint()
        '''
        return word.startswith('(')
    def nOfRightPar(word):
        '''
        how many '('s is in the end of this word?
        a helper function of myPrettyPrint()
        '''
        count = 0
        length = len(word)
        for i in range(length):
            if word[length - i - 1] == ')': count += 1
            else                          : break
        return count
        
    indentStack = [0]                   # a stack that keeps the record of how much indentation is needed now
                                        # initialized to 0 because the first line do not need indentation
    inWords = inString.split()          # split the input string into words
    resultList = []                     # string builder
    for word in inWords:                # iterate over words
        resultList.append(word+' ')     # add to the result this word and a whitespace
        for i in range(nOfRightPar(word)):  
                                        # the number of right parenthesis 
            indentStack.pop()           # for each right parenthesis, pop the stack once
                                        # popping the stack means that back to the last indentation level
        if hasLeftPar(word):            # having left parenthesis means new indentation is needed
            indentStack.append(indentStack[-1] + len(word) + 1)
                                        # push the new indentation to the stack
        else:                           # not having left parenthesis means that a new line is needed
            resultList.append('\n' + ' '*indentStack[-1])
                                        # add a new line
    return ''.join(resultList)          # return the result  

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
    prob_of_rules = {'S': [1],
                     'VP': [1, 2],
                     'NP': [1]}
    '''
    def __init__(self, grammar_file):
        # key is LHS, value (list) is a list of RHSs, each RHS is a list
        self.rules = defaultdict(list)
        # key is LHS, value (list) is a list of float
        self.prob_of_rules = defaultdict(list)
        self.__read_file(grammar_file) # parse grammar file

    def __read_file(self, filename):
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
        rule_process_callback - input: rule, lhs
                                output: processed_rule
                                for example, for chosen rule "S -> NP VP",
                                rule is ['NP', 'VP'], lhs is 'S'
        posprocess_callback - input: concated sentence
                              output: processed sentence
        '''
        stack = []
        sentence = []
        # Depth-first iteration
        stack.append('ROOT')
        while len(stack) > 0:
            lhs = stack.pop()
            if lhs not in self.rules: # handle terminal symbols
                sentence.append(lhs)
                continue
            rhs = self.rules[lhs] # list of possible RHSs(list)
            # get list of probability of corresponding RHSs
            prob_of_rhs = self.prob_of_rules[lhs]
            # sample one RHS from all RHSs based on probability
            chosen_rule = weighted_sample(rhs, prob_of_rhs)
            # process the rule to add special token for tree and backet output
            processed_rule = rule_process_callback(chosen_rule, lhs)
            # travel the descendants from left to right, so added it in reversed oreder
            reversed_rule = processed_rule[::-1]
            stack.extend(reversed_rule)

        # process the sentence before return, to clean up the special token
        # introduced by preserving tree structure.
        return posprocess_callback(' '.join(sentence))

    def gen_sentence(self):
        rule_process_callback = lambda rule, lhs: rule # just return the rule
        posprocess_callback = lambda sentence: sentence # just return the sentence
        return self.generate(rule_process_callback, posprocess_callback)

    def __structure_posprocess(self, sentence):
        # clean up '#' introduced by internal symbol and unnecessary space
        return sentence.replace(' #', '').replace('# ', '')

    def gen_sentence_in_tree(self):
        def rule_process_callback(rule, lhs):
            # contain '#' as a special internal symbol
            return ['(# ' + lhs] + rule + ['#)']
        return self.generate(rule_process_callback, self.__structure_posprocess)

    def gen_sentence_in_bracket(self):
        def rule_process_callback(rule, lhs):
            # contain '#' as a special internal symbol
            if lhs == 'S':
                return ['{#'] + rule + ['#}']
            elif lhs == 'NP':
                return ['[#'] + rule + ['#]']
            else:
                return rule
        return self.generate(rule_process_callback, self.__structure_posprocess)

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
            print myPrettyPrint(grammar.gen_sentence_in_tree())
        elif args.b: # print sentence in bracket 
            print grammar.gen_sentence_in_bracket()
        else:
            print grammar.gen_sentence()

if __name__ == '__main__':
    main()
