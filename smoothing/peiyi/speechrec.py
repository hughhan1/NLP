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
  train_file = argv.pop(0)

  if not argv:
    print "warning: no input files specified"

  lm = Probs.LanguageModel()
  lm.set_smoother(smoother)
  lm.read_vectors(lexicon)
  lm.train(train_file)

  overall_error_rate = 0.0
  total_words = 0
  for testfile in argv:
    f = open(testfile)
    line = f.readline()
    tokens_num_u = int(line.split()[0])

    candidates = []

    # Read data from test files according to their format
    line = f.readline()
    while line:
      items = line.split()
      line = f.readline()

      error_rate = float(items[0])
      log_p_uw = float(items[1])
      words = int(items[2])

      w = []
      for i in xrange(3, words + 5):
        w.append(items[i])

      w = w[1:-1]

      # calculate the log probability of a sentence using trigram model
      log_p_z = 0.0
      x, y = Probs.BOS, Probs.BOS
      for z in w:
        log_p_z += math.log(lm.prob(x, y, z))

        x = y
        y = z

      log_p_z += math.log(lm.prob(x, y, Probs.EOS))

      # For bigram model:
      #y = Probs.BOS
      #for z in w:
      #  log_p_z += math.log(lm.prob_bigram(y, z))
      #  y = z
      #log_p_z += math.log(lm.prob_bigram(y, Probs.EOS))

      # For unigram model:
      #for z in w:
      #  log_p_z += math.log(lm.prob_unigram(z))

      candidates.append((error_rate, words, log_p_uw + log_p_z / math.log(2)))

    # Pick the candidate with highest probability
    candidate = max(candidates, key = lambda item : item[2])
    overall_error_rate += candidate[0] * candidate[1]
    total_words += candidate[1]
    print '{0}\t{1}'.format(candidate[0], testfile)

  print '{0:0.03f}\t{1}'.format(overall_error_rate / total_words, "OVERALL")

if __name__ ==  "__main__":
  main()
