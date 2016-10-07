# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)


import math
import random
import re
import sys
import numpy

# TODO for TA: Currently, we use the same token for BOS and EOS as we only have
# one sentence boundary symbol in the word embedding file.  Maybe we should
# change them to two separate symbols?
BOS = 'EOS'   # special word type for context at Beginning Of Sequence
EOS = 'EOS'   # special word type for observed token at End Of Sequence
OOV = 'OOV'    # special word type for all Out-Of-Vocabulary words
OOL = 'OOL'    # special word type for all Out-Of-Lexicon words
DEFAULT_TRAINING_DIR = "/usr/local/data/cs465/hw-lm/All_Training/"
OOV_THRESHOLD = 3  # minimum number of occurrence for a word to be considered in-vocabulary


# TODO for TA: Maybe we should use inheritance instead of condition on the
# smoother (similar to the Java code).
class LanguageModel:
    def __init__(self):
        # The variables below all correspond to quantities discussed in the assignment.
        # For log-linear or Witten-Bell smoothing, you will need to define some 
        # additional global variables.
        self.smoother = None       # type of smoother we're using
        self.lambdap = None        # lambda or C parameter used by some smoothers

        # The word vector for w can be found at self.vectors[w].
        # You can check if a word is contained in the lexicon using
        #    if w in self.vectors:
        self.vectors = None     # loaded using read_vectors()

        self.vocab = None       # set of words included in the vocabulary
        self.vocab_size = None  # V: the total vocab size including OOV.

        self.tokens = None      # the c(...) function
        self.types_after = None # the T(...) function

        self.progress = 0        # for the progress bar

        self.bigrams = None
        self.trigrams = None
        
        # the two weight matrices U and V used in log linear model
        # They are initialized in train() function and represented as two
        # dimensional lists.
        self.U, self.V = None, None  

        self.Z_dict = None # maps from (x,y)   pairs   to their Z(x,y)   values
        self.u_dict = None # maps from (x,y,z) triples to their u(xyz)   values
        self.p_dict = None # maps from (z,x,y) triples to their p(z|x,y) values

        # self.tokens[(x, y, z)] = # of times that xyz was observed during training.
        # self.tokens[(y, z)]    = # of times that yz was observed during training.
        # self.tokens[z]         = # of times that z was observed during training.
        # self.tokens[""]        = # of tokens observed during training.
        #
        # self.types_after[(x, y)]  = # of distinct word types that were
        #                             observed to follow xy during training.
        # self.types_after[y]       = # of distinct word types that were
        #                             observed to follow y during training.
        # self.types_after[""]      = # of distinct word types observed during training.


    def __get_theta_and_f(self, x, y, z, theta=None, f=None):

        if theta is None:                                 # If we are not passed a theta or f
            theta = numpy.zeros(2 * self.dim * self.dim)  # vector, we can just initialize them
        if f is None:                                     # to vectors of size 2 * d^2.
            f = numpy.zeros(2 * self.dim * self.dim)

        if x not in self.vectors:
            x = OOL
        if y not in self.vectors:
            y = OOL
        if z not in self.vectors:
            z = OOL

        x_vec = self.vectors[x]
        y_vec = self.vectors[y]
        z_vec = self.vectors[z]

        idx = 0                                      # Here, we iterate through the
        for i in xrange(0, self.dim):                # elements of the matrix U and
            for j in xrange(0, self.dim):            # add its values to their
                theta[idx] = self.U[i][j]            # corresponding indices in theta.
                f[idx]     = x_vec[i] * z_vec[j]     # We then calculate theta's
                idx += 1                             # corresponding elements for f.

        for i in xrange(0, self.dim):                # Next, we iterate through the
            for j in xrange(0, self.dim):            # elements of the matrix V, and
                theta[idx] = self.V[i][j]            # perform the same calculations,
                f[idx]     = y_vec[i] * z_vec[j]     # also to be added to their
                idx += 1                             # corresponding indices in theta.

        return theta, f


    def prob(self, x, y, z):
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """

        if self.smoother == "UNIFORM":         # With uniform smoothing, we can simply
            return float(1) / self.vocab_size  # assign 1/V probability to everything.

        elif self.smoother == "ADDL":
            if x not in self.vocab:
                x = OOV
            if y not in self.vocab:
                y = OOV
            if z not in self.vocab:
                z = OOV
            return ((self.tokens.get((x, y, z), 0) + self.lambdap) /
                (self.tokens.get((x, y), 0) + self.lambdap * self.vocab_size))

            # Notice that summing the numerator over all values of typeZ
            # will give the denominator.  Therefore, summing up the quotient
            # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
            # as is required for any probability function.

        elif self.smoother == "BACKOFF_ADDL":
            if x not in self.vocab:
                x = OOV
            if y not in self.vocab:
                y = OOV
            if z not in self.vocab:
                z = OOV

            # To calculate the probability of z given x, y, we need to backoff and calculate the
            # probability of z given y. But to calculate that, we again need to backoof and
            # calculate the probability of z.

            # In the following code segment, we first calculate the probability of z, and use it
            # to build into the probability of z given y, which is finally used to build into
            # the probability of z given x, y.

            prob_z = ((self.tokens.get((z), 0) + self.lambdap) / 
                      (self.tokens.get((''), 0) + self.lambdap * self.vocab_size))

            prob_zy = ((self.tokens.get((y, z), 0) + self.lambdap * self.vocab_size * prob_z) /
                       (self.tokens.get((y), 0) + self.lambdap * self.vocab_size))

            return ((self.tokens.get((x, y, z), 0) + self.lambdap * self.vocab_size * prob_zy) /
                    (self.tokens.get((x, y), 0) + self.lambdap * self.vocab_size))

        elif self.smoother == "BACKOFF_WB":
            sys.exit("BACKOFF_WB is not implemented yet (that's your job!)")
        elif self.smoother == "LOGLINEAR":

            if self.u_dict is None:
                self.u_dict = dict()
            if self.Z_dict is None:
                self.Z_dict = dict()
            if self.p_dict is None:
                self.p_dict = dict()

            if x not in self.vocab:
                x = OOV
            if y not in self.vocab:
                y = OOV
            if z not in self.vocab:
                z = OOV

            theta, f = self.__get_theta_and_f(x, y, z)

            # Now, we can simply calculate u(xyz) by taking e to the power of the product
            # of theta and f. We will store this value in a dictionary, so that we can use
            # dynamic programming.

            self.u_dict[(x, y, z)] = numpy.exp(numpy.dot(theta, f))

            if (x, y) not in self.Z_dict:

                summation = 0
                for zp in self.vocab:
                    if zp not in self.vectors:
                        zp = OOL
                    zp_vec = self.vectors[zp]

                    if (x, y, zp) not in self.u_dict:

                        # If the tuple (x, y, z') hasn't been seen, then we need to
                        # calculate u(xyz'). We use the same procedure as above to do so,
                        # and like above, will then store it into the dictionary.

                        theta_temp, f_temp = self.__get_theta_and_f(x, y, zp)
                        self.u_dict[(x, y, zp)] = numpy.exp(numpy.dot(theta_temp, f_temp)) 

                    up = self.u_dict[(x, y, zp)]  # Fetch our value of u' from our u dict,
                    summation += up               # and add it to the value of Z

                self.Z_dict[(x, y)] = summation

            u = self.u_dict[(x, y, z)]    # Now let's get the value of u from the u dict
            Z = self.Z_dict[(x, y)]       # Similary, get the value of Z from the Z dict
            p = u / Z

            self.p_dict[(z,x,y)] = p      # Store the calculated probability in our
                                          # probability dictionary
            return p

        else:
            sys.exit("%s has some weird value" % self.smoother)


    def filelogprob(self, filename):
        """Compute the log probability of the sequence of tokens in file.
        NOTE: we use natural log for our internal computation.  You will want to
        divide this number by log(2) when reporting log probabilities.
        """
        logprob = 0.0
        x, y = BOS, BOS
        corpus = self.open_corpus(filename)
        for line in corpus:
            for z in line.split():
                prob = self.prob(x, y, z)
                logprob += math.log(prob)
                x = y
                y = z
        logprob += math.log(self.prob(x, y, EOS))
        corpus.close()
        return logprob


    def read_vectors(self, filename):
        """Read word vectors from an external file.  The vectors are saved as
        arrays in a dictionary self.vectors.
        """
        with open(filename) as infile:
            header = infile.readline()
            self.dim = int(header.split()[-1])
            self.vectors = {}
            for line in infile:
                arr = line.split()
                word = arr.pop(0)
                self.vectors[word] = [float(x) for x in arr]


    def train(self, filename):
        """Read the training corpus and collect any information that will be needed
        by the prob function later on.  Tokens are whitespace-delimited.

        Note: In a real system, you wouldn't do this work every time you ran the
        testing program. You'd do it only once and save the trained model to disk
        in some format.
        """
        sys.stderr.write("Training from corpus %s\n" % filename)

        # Clear out any previous training
        self.tokens = { }
        self.types_after = { }
        self.bigrams = []
        self.trigrams = [];

        # While training, we'll keep track of all the trigram and bigram types
        # we observe.  You'll need these lists only for Witten-Bell backoff.
        # The real work:
        # accumulate the type and token counts into the global hash tables.

        # If vocab size has not been set, build the vocabulary from training corpus
        if self.vocab_size is None:
            self.set_vocab_size(filename)

        # We save the corpus in memory to a list tokens_list.  Notice that we
        # appended two BOS at the front of the list and a EOS at the end.  You
        # will need to add more BOS tokens if you want to use a longer context than
        # trigram.
        x, y = BOS, BOS  # Previous two words.  Initialized as "beginning of sequence"
        # count the BOS context
        self.tokens[(x, y)] = 1
        self.tokens[y] = 1

        tokens_list = [x, y]  # the corpus saved as a list
        corpus = self.open_corpus(filename)
        for line in corpus:
            for z in line.split():
                # substitute out-of-vocabulary words with OOV symbol
                if z not in self.vocab:
                    z = OOV
                # substitute out-of-lexicon words with OOL symbol (only for log-linear models)
                if self.smoother == 'LOGLINEAR' and z not in self.vectors:
                    z = OOL
                self.count(x, y, z)
                self.show_progress()
                x=y; y=z
                tokens_list.append(z)
        tokens_list.append(EOS)   # append a end-of-sequence symbol 
        sys.stderr.write('\n')    # done printing progress dots "...."
        self.count(x, y, EOS)     # count EOS "end of sequence" token after the final context
        corpus.close()

        if self.smoother == 'LOGLINEAR': 
            # Train the log-linear model using SGD.

            # Initialize parameters
            self.U = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]
            self.V = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]

            # Optimization parameters
            gamma0 = 0.1                                # initial learning rate, used to 
                                                        #     compute actual learning rate
            theta0 = numpy.zeros(2*self.dim*self.dim)   # set original theta to the 0 vector
            f0     = numpy.zeros(2*self.dim*self.dim)   # set original f to the 0 vector
            epochs = 10                                 # number of passes

            self.N = len(tokens_list) - 2  # number of training instances

            sys.stderr.write("Start optimizing.\n")

            ######################### BEGIN Stochastic Gradient Ascent #########################

            # theta = theta0
            # f     = f0
            # t     = 0
            # for e in range(epochs):
            #     for i in range(self.N):

            #         gamma = gamma0 / (1 + gamma0 * (self.lambdap/self.N) * t)

            #         summation = 0
            #         for (z, x, y) in self.p_dict:
            #             probability = self.p_dict[(z, x, y)]
            #             _, fp = self.__get_theta_and_f(x, y, z, theta0, f0)  # fp denotes f(xyz')
            #             summation += probability + fp

            #         F_gradient = f - summation - (2*self.lambdap / self.N) * theta
            #         theta = theta + gamma * F_gradient
            #         t += 1

            ########################## END Stochastic Gradient Ascent ##########################

        sys.stderr.write("Finished training on %d tokens\n" % self.tokens[""])


    def count(self, x, y, z):
        """Count the n-grams.  In the perl version, this was an inner function.
        For now, I am just using a class variable to store the found tri-
        and bi- grams.
        """
        self.tokens[(x, y, z)] = self.tokens.get((x, y, z), 0) + 1
        if self.tokens[(x, y, z)] == 1:       # first time we've seen trigram xyz
            self.trigrams.append((x, y, z))
            self.types_after[(x, y)] = self.types_after.get((x, y), 0) + 1

        self.tokens[(y, z)] = self.tokens.get((y, z), 0) + 1
        if self.tokens[(y, z)] == 1:        # first time we've seen bigram yz
            self.bigrams.append((y, z))
            self.types_after[y] = self.types_after.get(y, 0) + 1

        self.tokens[z] = self.tokens.get(z, 0) + 1
        if self.tokens[z] == 1:         # first time we've seen unigram z
            self.types_after[''] = self.types_after.get('', 0) + 1 

        self.tokens[''] = self.tokens.get('', 0) + 1  # the zero-gram


    def set_vocab_size(self, *files):
        """When you do text categorization, call this function on the two
        corpora in order to set the global vocab_size to the size
        of the single common vocabulary.

        NOTE: This function is not useful for the loglinear model, since we have
        a given lexicon.
         """
        count = {} # count of each word

        for filename in files:
            corpus = self.open_corpus(filename)
            for line in corpus:
                for z in line.split():
                    count[z] = count.get(z, 0) + 1
                    self.show_progress();
            corpus.close()

        self.vocab = set(w for w in count if count[w] >= OOV_THRESHOLD)

        self.vocab.add(OOV)  # add OOV to vocab
        self.vocab.add(EOS)  # add EOS to vocab (but not BOS, which is never a possible outcome but only a context)

        sys.stderr.write('\n')    # done printing progress dots "...."

        if self.vocab_size is not None:
            sys.stderr.write("Warning: vocab_size already set; set_vocab_size changing it\n")
        self.vocab_size = len(self.vocab)
        sys.stderr.write("Vocabulary size is %d types including OOV and EOS\n"
                                            % self.vocab_size)

    def set_smoother(self, arg):
        """Sets smoother type and lambda from a string passed in by the user on the
        command line.
        """
        r = re.compile('^(.*?)-?([0-9.]*)$')
        m = r.match(arg)
        
        if not m.lastindex:
            sys.exit("Smoother regular expression failed for %s" % arg)
        else:
            smoother_name = m.group(1)
            if m.lastindex >= 2 and len(m.group(2)):
                lambda_arg = m.group(2)
                self.lambdap = float(lambda_arg)
            else:
                self.lambdap = None

        if smoother_name.lower() == 'uniform':
            self.smoother = "UNIFORM"
        elif smoother_name.lower() == 'add':
            self.smoother = "ADDL"
        elif smoother_name.lower() == 'backoff_add':
            self.smoother = "BACKOFF_ADDL"
        elif smoother_name.lower() == 'backoff_wb':
            self.smoother = "BACKOFF_WB"
        elif smoother_name.lower() == 'loglinear':
            self.smoother = "LOGLINEAR"
        else:
            sys.exit("Don't recognize smoother name '%s'" % smoother_name)
        
        if self.lambdap is None and self.smoother.find('ADDL') != -1:
            sys.exit('You must include a non-negative lambda value in smoother name "%s"' % arg)


    def open_corpus(self, filename):
        """Associates handle CORPUS with the training corpus named by filename."""
        try:
            corpus = file(filename, "r")
        except IOError, err:
            try:
                corpus = file(DEFAULT_TRAINING_DIR + filename, "r")
            except IOError, err:
                sys.exit("Couldn't open corpus at %s or %s" % (filename,
                                 DEFAULT_TRAINING_DIR + filename))
        return corpus


    def show_progress(self, freq=5000):
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write('.')
