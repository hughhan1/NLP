# Random Sentence Generator

This subdirectory contains an implementation of a random sentence generator using context-free-grammars (CFGs).

### What is it?

`randsent.py` is a random sentence generator that takes in a CFG and produces sentences using the specified rules. Each file
included in this repository with a `.gr` suffix is a grammar file.

A grammar file is just a fancy name for a `.txt` file that contains text in a special format, used to represent a CFG. An
extremely simple `.gr` file is pasted below, with some basic documentation.

```
1  ROOT  S .          # Each sentence begins with a ROOT

1  S     NP   VP      # These lines contain the rules that describe non-terminal
1  VP    Verb NP      # symbols. 
1  NP    Det  Noun    #
1  NP    NP   PP      # A non-terminal symbol could be a noun-phrase (NP),
1  PP    Prep NP      # verb-phrase (VP), prepositional-phrase (PP), or even
1  Noun  Adj  Noun    # just a noun. These are different from specific words.

1  Verb  ate          # The following rules speicify how non-terminals should
1  Verb  wanted       # be mapped to terminals. That is, a verb symbol could
                      # turn into the words "ate" or "wanted". 
1  Det   the          #
1  Det	 a            # A determiner could turn into the words "the" or "a"
                     
1  Noun  president    # Now note that each rule also contains a number in the
1  Noun  sandwich     # leftmost column. That number represents a weight, or
                      # representation of probability. The higher the number,
1  Adj   fine         # the more likely a specific rule is bound to happen.
                      # The implementation is a bit more complicated that that,
1  Prep  with         # but that's the general idea.
1  Prep  on
1  Prep  in           # For more information, read the documentation in any
                      # of the .gr files.
```

### How does it work?

The algorithm itself is actually quite simple. Given a text file, `randsent.py` reads in each line, making sure to ignore
whitespace and anything that is specified as a comment. (In our `.gr` file format, anything after a `#` symbol represents a
comment.)

After the grammar is read in, it is stored as two dictionaries of lists (`defaultdict(list)`). Each dictionary maps from the
parent symbol of a particular rule to a list of:
  1. all the possible children the parent symbol could turn into,
  2. the probabilities corresponding to each child symbol.
  
After generating these data structures, we can begin building random sentences. Although we are traversing through a
probabilistic tree structure, we do so iteratively to avoid running out of memory on the stack. English sentences shouldn't
be obtrusively long (or nonsensical, for that matter), but our simpler grammars aren't quite smart, and sometimes produce
quite repetitive sentences that could cause our program to run out of memory, if generation were done recursively.

### Sample Grammars

|    Complexity     |  Grammar File  |                                  Description                                 |
|-------------------|----------------|------------------------------------------------------------------------------|
| Few Rules         | `uniform.gr`   |  Equal probability for each rule.                                            |
| Few Rules         | `simple.gr`    |  More realistic probability for each rule.                                   |
| Basic Rules       | `basic.gr`     |  Basic English. Transitive sentences, simple questions, clauses.             |
| Basic Rules       | `trump.gr`     |  Donald Trump speak! Basic English with a twist.                             |
| More Rules        | `phenomena.gr` |  More complex. Auxillary verbs, intransitive verbs, present tense.           |
| Even *More* Rules | `extra.gr`     |  One step further. Subjects & direct objects, infinitives, "for-to" phrases. |

Note that the more complex a grammar is, the better the sentences will be. (That is, the English grammar itself is quite
complex.) 

### Usage

Simply clone this repository, and run the following command.
```
$ python randsent.py <filename> <num_sentences>
```
-  `<filename` indicates a CFG text file, which will indicate the rules of the sentences to be generated.
-  `<num_sentences>` indicates the number of random sentences to be generated.

### Collaborators

- *Hugh Han (Johns Hopkins University)*
- *Xiaochen Li (Johns Hopkins University)*
- *Shijie Wu (Johns Hopkins University)*
