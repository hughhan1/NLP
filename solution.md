1. Solution:
see `no1-1.txt`

2. Solution:
    1. NP, as a nonterminal symbol, has two rules to rewrite it ("NP -> Det
       Noun" and "NP -> NP PP"). So there is 50% of chance the program will
       choose "NP -> NP PP" when constructing a sentence. This rule create a
       self loop linking NP back to NP. In addition, PP has only one rule ("PP
       -> Prep NP") to rewrite it, which also links back to NP. As a result,
       there is 50% of chance the program will rewrite NP as NP Prep NP,
       leading to two more NP. The expected number of NP in the rewrite rule
       of NP is 1. So it is very likely that this grammar will create many
       very long sentence.

    2. Even though it is possible that a Noun could have many Adj given "Noun
       -> Adj Noun", it is less likely for the program to choose this rule
       because there are 5 rules to rewrite a Noun with a terminal symbol. In
       other word, the probability of choosing "Noun -> Adj Noun" is only 1/6.
       The probability of having n Adj before a Noun is (1/6)^n. So it is rare
       to have multiple Adj.

    3. 
