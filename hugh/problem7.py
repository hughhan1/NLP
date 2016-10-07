#!/usr/bin/python

# Sample program for hw-lm
# CS465 at Johns Hopkins University.

# Converted to python by Eric Perlman <eric@cs.jhu.edu>

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

import math
import sys

import Probs

# Computes the log probability of the sequence of tokens in file,
# according to a trigram model.  The training source is specified by
# the currently open corpus, and the smoothing method used by
# prob() is specified by the global variable "smoother". 

def main():
  course_dir = '/usr/local/data/cs465/'
  argv = sys.argv[1:]

  if len(argv) < 2:
    print """
Prints the log-probability of each file under a smoothed n-gram model.

Usage:   %s smoother lexicon trainpath files...
Example: %s add0.01 %shw-lm/lexicons/words-10.txt switchboard-small %shw-lm/speech/sample*

Possible values for smoother: uniform, add1, backoff_add1, backoff_wb, loglinear1
  (the \"1\" in add1/backoff_add1 can be replaced with any real lambda >= 0
   the \"1\" in loglinear1 can be replaced with any C >= 0 )
lexicon is the location of the word vector file, which is only used in the loglinear model
trainpath is the location of the training corpus
  (the search path for this includes "%s")
""" % (sys.argv[0], sys.argv[0], course_dir, course_dir, Probs.DEFAULT_TRAINING_DIR)
    sys.exit(1)

  smoother = argv.pop(0)
  lexicon = argv.pop(0)
  corpus_1 = argv.pop(0)
  corpus_2 = argv.pop(0)

  if not argv:
    print "warning: no input files specified"

  
  lm = Probs.LanguageModel()
  lm.set_vocab_size(corpus_1, corpus_2)
  lm.set_smoother(smoother)
  lm.read_vectors(lexicon)
  lm.train(corpus_1)

  # Calculate trigram model P(w|w is gen) using gen corpus for each test files
  gen = []
  for testfile in argv:
    p_gen = []
    x, y = Probs.BOS, Probs.BOS
    corpus = lm.open_corpus(testfile)
    for line in corpus:
      for z in line.split():
        prob = lm.prob(x, y, z)
        p_gen.append(prob)
        x = y
        y = z
    p_gen.append(lm.prob(x, y, Probs.EOS))
    corpus.close()
  	
    gen.append(p_gen)

  # Calculate trigram model P(w|w is spam) using spam corpus for each test files
  lm.train(corpus_2)
  spam = []
  cnt_gen = 0
  cnt_spam = 0
  for testfile in argv:
    p_spam = []
    x, y = Probs.BOS, Probs.BOS
    corpus = lm.open_corpus(testfile)
    for line in corpus:
      for z in line.split():
        prob = lm.prob(x, y, z)
        p_spam.append(prob)
        x = y
        y = z
    p_spam.append(lm.prob(x, y, Probs.EOS))
    corpus.close()
    
    spam.append(p_spam)

  # For each test files using 1.0 / (1.0 + 2.0 * (P(w|w is gen) / P(w| w is spam)))
  for i in xrange(0, len(argv)):
    p_wg_ws = 2.0
    for j in xrange(0, len(gen[i])):
      p_wg_ws *= gen[i][j] / spam[i][j]

    p_sw = 1.0 / (1.0 + p_wg_ws)

    # Determine whether this test file is spam depending on P(w is spam|w)    
    if p_sw < 0.9:
      print 'gen\t{0}'.format(argv[i])
      cnt_gen += 1
    else:
      print 'spam\t{0}'.format(argv[i])
      cnt_spam += 1
  
  print '{0} looked more like gen ({1})'.format(cnt_gen, float(cnt_gen) / (cnt_gen + cnt_spam))
  print '{0} looked more like spam ({1})'.format(cnt_spam, float(cnt_spam) / (cnt_gen + cnt_spam))

if __name__ ==  "__main__":
  main()
