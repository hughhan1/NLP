import random
import sys
import argparse

class Symbol:  
    '''
    a symbol
    '''
    def __init__(self, name, probability = None, descendants = None):
        self.rules = list()          # a list of all the rules that turn this symbol into other symbol(s)
        self.nOfRules = 0            # number of the rules above
        self.totalProbability = 0    # the sum of all the probabilities of its rules
        self.name = name             # the name of this symbol. eg. ROOT, Noun, dog
        if probability != None:
            self.addRule(probability, descendants)

    def __str__(self):
        '''
        return a string listing all the rules of this symbol
        '''
        resultList = []
        lastProb = 0
        for rule in self.rules:
            ruleConstructor = []
            ruleConstructor.append(str(rule[0] - lastProb))
            lastProb = rule[0]
            ruleConstructor.append(self.name)
            for des in rule[1]:
                ruleConstructor.append(des)
            ruleConstructor.append('\n') 
            resultList.append('\t'.join(ruleConstructor))
        return ''.join(resultList)
            
        # return self.isTerminal
    def addRule(self, probability, descendants):
        '''
        add a new rule whose ancestor is this symbol
        '''
        self.totalProbability += probability
        self.rules.append([self.totalProbability, descendants]) # the format of the rule:
                                                                # [accumulated probability, list of descendants]
        self.nOfRules += 1
    def expand(self):
        '''
        expand this symbol into some symbols according to the rule
        '''
        rdm = random.random() * self.totalProbability
        for rule in self.rules:
            if rdm < rule[0]:
                return rule[1]
        else:
            raise StandardError('code went wrong and expand failed!')
        

class Grammar:
    '''
    a set of grammar
    '''
    def __init__(self, input):
        '''
        takes a input stream which lists all the rules of this grammar
        '''
        self.root = 'ROOT'                  # the root symbol
        self.symbols = {'ROOT' : self.root} # dictionary of all the symbols in this grammar
        
        for line in input.readlines():
            rule = self.__parseLine(line)   # format of the rule:
                                            # {'name':name, 'probability':probability, 'descendants':descendants}
                                            # if this line does not contain a rule, rule=None
            if rule == None:
                continue
            try:
                self.symbols[rule['name']].addRule(rule['probability'], rule['descendants'])
            except:
                self.symbols[rule['name']] = Symbol(rule['name'], rule['probability'], rule['descendants'])
                
    def __parseLine(self, line):
        '''
        return {'name':name, 'probability':probability, 'descendants':descendants}
        return None, if this line does not contain a rule
        raise ValueError, if this line is not understandable
        skip '#' and all its followers in a line
        '''
        isGrammar = False   # a flag denotes whether this line is a grammar of an empty line
        words = line.split()
        for i, word in enumerate(words):
            if word[0] =='#':
                break
            if i == 0:
                try:
                    probability = float(word)
                    isGrammar = True
                except:
                    raise ValueError('\'' + line + '\' is not a valid grammar line')
            elif i == 1:
                name = word
                descendants = []
            else:
                descendants.append(word)
        if not isGrammar:
            return None
        try:
            # print 'processed: ', line
            return {'name':name, 'probability':probability, 'descendants':descendants}
        except:
            raise ValueError('\'' + line + '\' is not a valid grammar line')
        
    def getRoot(self):
        '''
        return the root string of this grammar
        (always 'ROOT')
        '''
        return self.root
    
    def __getitem__(self, symbol):
        '''
        return the Symbol instance of the symbol string
        raise ValueError if this symbol is not in this grammar
        '''
        try:
            return self.symbols[symbol]
        except:
            raise ValueError('no nonterminal symbol \'' + symbol + '\' is in this grammar')
    
    def contains(self, symbol):
        '''
        does this grammar contain a rule that can expand this symbol?
        '''
        return symbol in self.symbols
       
    def __str__(self):
        '''
        return a string listing all the rules in this grammar
        '''
        resultList = []
        for symbol in self.symbols.itervalues():
            resultList.append(str(symbol))
        return ''.join(resultList)


class SubSentence:
    '''
    part of a sentence.
    take any symbol as its root, and expand it until there is no nonterminals in the subsentence
    '''
    def __init__(self, grammar, root):
        self.root = root                # the string of the root symbol
        self.descendants = None         # a list of all the descendent sub-sentences
        self.grammar = grammar
        self.init()
        
    def init(self):
        '''
        a helper function to initialize the object
        '''
        if not self.grammar.contains(self.root):
            self.descendants = None
            return
        self.descendants = [SubSentence(self.grammar, x) for x in self.grammar[self.root].expand()]

    def getSentence(self):
        '''
        return a string of the sub-sentence
        '''
        if self.descendants == None:
            return self.root
        resultList = [x.getSentence() for x in self.descendants]
        return ' '.join(resultList)
        
    def getTree(self):
        '''
        return a one-line string of the tree of the sub-sentence, such as:
        (ROOT (S (NP (Det every) (Noun president)) (VP (V_intran jumped))) .)
        (for problem 4, -t)
        '''
        if self.descendants == None:
            return self.root
        resultList = [self.root]
        for x in self.descendants:
            resultList.append(x.getTree())
        return '(' + ' '.join(resultList) + ')'
        
    def getSentence_b(self):
        '''
        return a string of the bracketed sentence, such as:
        {[the floor] kissed [the delicious chief of staff]} .
        where S constituents are surrounded with curly braces 
        and NP constituents are surrounded with square brackets.
        (for problem 4, -b)
        '''
        if self.descendants == None:
            return self.root
        resultList = [x.getSentence_b() for x in self.descendants]
        if self.root == 'S':
            resultList[0] = '{' + resultList[0]
            resultList[-1] = resultList[-1] + '}'
        if self.root == 'NP':
            resultList[0] = '[' + resultList[0]
            resultList[-1] = resultList[-1] + ']'
        return ' '.join(resultList)
        
class Sentence(SubSentence):
    def __init__(self, grammar):
        self.grammar = grammar          # the grammar of this sentence
        self.descendants = None         # a list of all the descendent sub-sentences
        self.root = grammar.getRoot()   # initialize the sentence with the root of the grammar
        self.init()                     # use its superclass's init()   

    def getTree(self):
        '''
        return a pretty-printed string of the tree of the sub-sentence 
        (for problem 4, -t)
        ''' 
        result = SubSentence.getTree(self)
        return myPrettyPrint(result)

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
    
    
        
def main(input = None):
    g = Grammar(input)
    for i in range(10):
        s = Sentence(g)
        print i+1, ':\t', s.getSentence(), '\n'
        # print i+1, ':\t', s.getSentence_b(), '\n'
        # print s.getTree(), '\n'
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate some sentences.')
    parser.add_argument('grammar_name', help='the path and filename of the grammar file')
    parser.add_argument('num_of_sentences', help='the number of generated sentence',
                        default=1, type=int, nargs='?')
    parser.add_argument('-t', help='print tree instead', action='store_true')
    parser.add_argument('-b', help='print brackets instead', action='store_true')
    args = parser.parse_args()
    
    try:
        input = file(args.grammar_name,'r')
    except IOError:
        sys.stderr.write('ERROR: Cannot read inputfile %s.\n' % sys.argv[1])
        sys.exit(1)
    # main(input)
    
    g = Grammar(input)
    for i in range(args.num_of_sentences):
        s = Sentence(g)
        if args.t:
            print s.getTree()
        elif args.b:
            print s.getSentence_b()
        else:
            print s.getSentence()
