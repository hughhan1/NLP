#!/usr/bin/python

# textcat.py
# Hugh Han
# CS465 at Johns Hopkins University.

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

        Usage:   %s smoother lexicon trainpath1 trainpath2 files...
        Example: %s add0.01 %shw-lm/lexicons/words-10.txt gen spam %shw-lm/speech/sample*

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
    train_file1 = argv.pop(0)
    train_file2 = argv.pop(0)

    if not argv:
        print "warning: no input files specified"

    lm = Probs.LanguageModel()                   # Create a new instance of a language
    lm.set_smoother(smoother)                    # model and set its attributes.
    lm.read_vectors(lexicon)
                                                 # Here, we set the vocabulary size to be
    lm.set_vocab_size(train_file1, train_file2)  # the union of both the first training
                                                 # corpus and the second training corpus.
  
    # We use natural log for our internal computations and that's
    # the kind of log-probability that fileLogProb returns.  
    # But we'd like to print a value in bits: so we convert
    # log base e to log base 2 at print time, by dividing by log(2).

    file1_logprob = dict()    # map from each testfile to log-probability using file1
    file2_logprob = dict()    # map from each testfile to log-probability using file2

    lm.train(train_file1)     # first, train using train_file1; then calculate the
    for testfile in argv:     # log-probabilities for each test file
        file1_logprob[testfile] = lm.filelogprob(testfile) / math.log(2)

    lm.train(train_file2)     # next, train using train_file2; then calculate the
    for testfile in argv:     # log-probabilities for each test file
        file2_logprob[testfile] = lm.filelogprob(testfile) / math.log(2)

    file1_count = 0                                             # Here, we count the number
    file2_count = 0                                             # of test files more similar
    for testfile in argv:                                       # to file1 and number of
        if file1_logprob[testfile] > file2_logprob[testfile]:   # test files more similar to
            print "%s\t%s" % (train_file1, testfile)            # file2. We print out which
            file1_count += 1                                    # test file is more similar
        else:                                                   # to which training file,
            print "%s\t%s" % (train_file2, testfile)            # for each test file.
            file2_count += 1

    file1_percent = 100 * float(file1_count)/float(file1_count + file2_count)
    file2_percent = 100 * float(file2_count)/float(file1_count + file2_count)

    print "%d looked more like %s (%.2f%%)" % (file1_count, train_file1, file1_percent)
    print "%d looked more like %s (%.2f%%)" % (file2_count, train_file2, file2_percent)


if __name__ ==  "__main__":
    main()
