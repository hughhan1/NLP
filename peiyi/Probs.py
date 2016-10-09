# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Modified by Hugh Han <hughhan@jhu.edu> and Peiyi Zheng <pzheng4@jhu.edu>
# for the CS465 course at Johns Hopkins University.  (10/8/2016)


import math
import random
import re
import sys
import numpy as np

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
        self.beta = None

        self.Z_dict = None # maps from (x,y)   pairs   to their Z(x, y)   values
        self.u_dict = None # maps from (x,y,z) triples to their u(xyz)    values
        self.p_dict = None # maps from (x,y,z) triples to their p(z| x,y) values
        self.count_z = None

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
        """
        TODO: Add documentation
        """
        if theta is None:                              # If we are not passed
            theta = np.zeros(2 * self.dim * self.dim + 1)  # a theta or f vector,
        if f is None:                                  # we can initialize
            f = np.zeros(2 * self.dim * self.dim + 1)      # them to vectors of
        
        beta_z = z                                               # size 2 * d^2.
        if beta_z not in self.vocab:
            beta_z = OOV
        if x not in self.vectors:
            x = OOL
        if y not in self.vectors:
            y = OOL
        if z not in self.vectors:
            z = OOL

        vec_x = self.vectors[x]
        vec_y = self.vectors[y]
        vec_z = self.vectors[z]

        idx = 0                                      # Here, we iterate through the
        for i in xrange(0, self.dim):                # elements of the matrix U and
            for j in xrange(0, self.dim):            # add its values to their
                theta[idx] = self.U[i][j]            # corresponding indices in theta.
                f[idx]     = vec_x[i] * vec_z[j]     # We then calculate theta's
                idx += 1                             # corresponding elements for f.

        for i in xrange(0, self.dim):                # Next, we iterate through the
            for j in xrange(0, self.dim):            # elements of the matrix V, and
                theta[idx] = self.V[i][j]            # perform the same calculations,
                f[idx]     = vec_y[i] * vec_z[j]     # also to be added to their
                idx += 1                             # corresponding indices in theta.

        theta[idx] = self.beta
        f[idx] = math.log(self.tokens.get(beta_z, 0) + 1.0)
        return theta, f


    def __get_XZ_and_YZ(self, x, y, z, XZ=None, YZ=None):
        """
        TODO: Add documentation
        """
        if XZ is None:
            XZ = np.zeros((self.dim, self.dim))
        if YZ is None: 
            YZ = np.zeros((self.dim, self.dim))

        if x not in self.vectors:
            x = OOL
        if y not in self.vectors:
            y = OOL
        if z not in self.vectors:
            z = OOL

        vec_x = self.vectors[x]
        vec_y = self.vectors[y]
        vec_z = self.vectors[z]

        XY = np.outer(vec_x, vec_z)
        YZ = np.outer(vec_y, vec_z)

        return XZ, YZ


    def __Z(self, x, y):
        """
        TODO: Add documentation
        """
        if x not in self.vocab:
            x = OOV
        if y not in self.vocab:
            y = OOV

        Z_xy = 0                                         # iterate and calculate the summation of
        for z_ in self.vocab:                            # u(xyz') for all z' in the vocabulary

            if z_ not in self.vectors:                   # if z' is not in the
                z_ = OOL                                 # lexicon, set it to OOL

            theta, f = self.__get_theta_and_f(x, y, z_)  # get theta and feature f
            u_xyz_ = np.exp(np.dot(theta, f))            # calculate u(xyz')
            Z_xy += u_xyz_                               # add u(xyz') to Z(xy)

        return Z_xy


    def prob_bigram(self, y, z):
        """
        Computes a smoothed estimate of the unigram probability p(z | y) according to the language
        model.
        """
        if self.smoother == "UNIFORM":
            return float(1) / self.vocab_size

        elif self.smoother == "ADDL":
            if y not in self.vocab:
                y = OOV
            if z not in self.vocab:
                z = OOV
            return ((self.tokens.get((y, z), 0) + self.lambdap) /
                    (self.tokens.get((y), 0) + self.lambdap * self.vocab_size))

        elif self.smoother == "BACKOFF_ADDL":
            if y not in self.vocab:
                y = OOV
            if z not in self.vocab:
                z = OOV

            p_z = ((self.tokens.get(z, 0) + self.lambdap) / 
                   (self.tokens.get('', 0) + self.lambdap * self.vocab_size))

            return ((self.tokens.get((y, z), 0) + self.lambdap * self.vocab_size * p_z) / 
                    (self.tokens.get(y, 0) + self.lambdap * self.vocab_size))

        elif self.smoother == "BACKOFF_WB":
            sys.exit("BACKOFF_WB is not implemented yet (that's your job!)")
        else:
            sys.exit("%s has some weird value" % self.smoother)


    def prob_unigram(self, z):
        """
        Computes a smoothed estimate of the unigram probability p(z) according to the language
        model.
        """
        if self.smoother == "UNIFORM":
            return float(1) / self.vocab_size

        elif self.smoother == "ADDL" or self.smoother == "BACKOFF_ADDL":
            if z not in self.vocab:
                z = OOV
            return ((self.tokens.get(z, 0) + self.lambdap) /
                    (self.tokens.get('', 0) + self.lambdap * self.vocab_size))

        elif self.smoother == "BACKOFF_WB":
            sys.exit("BACKOFF_WB is not implemented yet (that's your job!)")
        else:
            sys.exit("%s has some weird value" % self.smoother)


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

            if (x, y, z) in self.p_dict:        # If p(z | xy) had aleady been calculated, just
              return self.p_dict[(x, y, z)]     # quickly look it up in the cache.

            theta, f = self.__get_theta_and_f(x, y, z)

            if (x, y, z) in self.u_dict:            # After getting the values of theta and f, we
                u_xyz = self.u_dict[(x, y, z)]      # can fetch the values of u(xyz) and Z(xy)
            else:                                   # from a cache if they have already been
                u_xyz = np.exp(np.dot(theta, f))    # calculated.
                self.u_dict[(x, y, z)] = u_xyz      # 
                                                    # If they have not been calculated, calculate
            if (x, y) in self.Z_dict:               # them now, and store them in a cache for 
                Z_xy = self.Z_dict[(x, y)]          # faster future retrieval.
            else:                                   #
                Z_xy = self.__Z(x, y)               # Finally, after getting u(xyz) and Z(xy), we
                self.Z_dict[(x, y)] = Z_xy          # can easily compute the value of p(z | xy).

            p_xyz = u_xyz / Z_xy 

            self.p_dict[(x,y,z)] = p_xyz            # Store the calculated probability in our
                                                    # probability dictionary
            return p_xyz

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

            self.u_dict = None    # Because we are training our probability model using a new
            self.Z_dict = None    # file, we need to reset all of our cached values of u(xyz),
            self.p_dict = None    # Z(xy), and p(z | xy). (These values used in self.prob()).

            for i in range(2, len(tokens_list)):
                x, y, z = tokens_list[i - 2], tokens_list[i - 1], tokens_list[i]
                self.trigrams.append((x, y, z))

            # Initialize parameters
            self.U = np.zeros((self.dim, self.dim))
            self.V = np.zeros((self.dim, self.dim))
            self.beta = 1.0

            # Optimization parameters
            gamma0 = 0.01                                   # initial learning rate, used to 
                                                            #     compute actual learning rate
            epochs = 10                                     # number of passes

            self.N = len(tokens_list) - 2                   # number of training instances

            sys.stderr.write("Start optimizing.\n")

            ####################################################################################
            ####################################################################################
            ########################  BEGIN Stochastic Gradient Ascent  ########################
            ####################################################################################

            t = 0
            for e in range(epochs):

                F_theta = 0

                for i in range(self.N):

                    gamma = gamma0 / (1 + gamma0 * (self.lambdap/self.N) * t)  # update gamma

                    x, y, z = self.trigrams[i]  # Fetch the i-th trigram (x, y, z) of our training

                    beta_z = z
                    if beta_z not in self.vocab:
                        beta_z = OOV
                                                # corpus.
                    if x not in self.vectors:
                        x = OOL                 # If any piece of the trigram is not in the
                    if y not in self.vectors:   # lexicon, we set it to the value OOL, meaning
                        y = OOL                 # 'out of lexixon'.
                    if z not in self.vectors:   #
                        z = OOL                 # Note that OOL != OOV.

                    theta, f = self.__get_theta_and_f(x, y, z)

                    u_xyz = np.exp(np.dot(theta, f))  # Calculate the value of u(xyz) and Z(xy)
                    Z_xy  = self.__Z(x, y)            # using the current theta value.

                    p_xyz = u_xyz / Z_xy              # Use u(xyz) and Z(xy) to calculate p(z|xy)

                    # Below, we compute one component of F(theta). That is, F_i(theta). To do so,
                    # we add log(p(z | xy)) and subtract (C/N) * (magnitude ot theta). We do this
                    # on every iteration to generate a summation from i=1 to i=N.

                    F_theta += math.log(p_xyz) - (self.lambdap/self.N) * np.sum(np.square(theta))

                    # Below, we calculate the matrices XZ and YZ. XZ and YZ are matrices
                    # representing the element-by-element product of X * Z and Y * Z. Note that F
                    # is exactly equivalent to the enumeration of XZ concatenated with YZ.

                    XZ, YZ = self.__get_XZ_and_YZ(x, y, z)
                                                                              # Now we calculate
                    gradient_U = XZ - (2.0 * self.lambdap / self.N) * self.U  # part of the
                    gradient_V = YZ - (2.0 * self.lambdap / self.N) * self.V  # gradient using the
                                                                              # 1st and 3rd terms.

                    gradient_Beta = math.log(self.tokens.get(beta_z, 0) + 1.0) - (2.0 * self.lambdap / self.N) * self.beta
                    #print gradient_Beta
                    for z_ in self.vocab:           # Now we iterate through all possible values
                                                    # of z' and finish calculating the gradient
                        beta_z = z_
                        if beta_z not in self.vocab:
                            beta_z = OOV
                        if z_ not in self.vectors:  # using the 2nd term.    
                            z_ = OOL

                        XZ_, YZ_ = self.__get_XZ_and_YZ(x, y, z_)

                        # Below, we calculate the probability p(z'| xy) by first calculating the
                        # value of u(xyz') and dividing by the value of Z(xy). Note that Z(xy)
                        # remains constaint for this entire inner loop.

                        theta_, f_ = self.__get_theta_and_f(x, y, z_)
                        u_xyz_ = np.exp(np.dot(theta_, f_)) 
                        p_xyz_ = u_xyz_ / Z_xy

                        gradient_U -= p_xyz_ * XZ_  # We use p(z'| xy) to calculate the partial
                        gradient_V -= p_xyz_ * YZ_  # derivatives of F with respect to U and V.
                        gradient_Beta -= p_xyz_ * math.log(self.tokens.get(beta_z, 0) + 1.0)
                        #print p_xyz_ * math.log(self.tokens.get(beta_z, 0) + 1.0)

                        # Note: in the stochastic gradient ascent algorithm, the summation of the
                        #       products of p(z'| xy) and yz' are calculated, then subtracted from
                        #       the gradient. But in our implementation, however, we iterate 
                        #       through the elements z' and instead of summing up all of their
                        #       products, we subtract a single piece of the summation on every
                        #       iteration.

                    self.U += gamma * gradient_U   # Finally, we need to update our U and V using
                    self.V += gamma * gradient_V   # the partial derivatives of F with respect to
                                                   # matrices U and V
                    #print gradient_Beta
                    #print '------- {0} ----------'.format(i)

                    self.beta += gamma * gradient_Beta
                    # Note that when U and V are updated, theta and f are also implicitly updated.

                    t += 1
                                    # After N iterations, we have a sum of all F_i(theta), for all
                F_theta /= self.N   # i from 1 to N. To calculate F(theta), we take the average,
                                    # which is calculated simply dividing by N.

                print self.beta

                print "epoch %d: F=%f" % (e+1, F_theta)

            ####################################################################################
            ########################   END Stochastic Gradient Ascent   ########################
            ####################################################################################
            ####################################################################################

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
