import numpy as np
import string
import sys

initial = {}        # First word per line
                    # P(w(t)) - goal

second_word = {}    # Take the previous state into account
                    # P(w(t)|w(t-1)) - goal

transitions = {}    # Take the two previous states into account
                    # P(w(t)|w(t-1), w(t-2)) - goal


def remove_punctuation(s):
    return s.translate(str.maketrans('','', string.punctuation))

def add2dict(d, k, v):
    if k not in d:
        d[k] = []
    d[k].append(v)

# Two roads diverged in a yellow wood,
# And sorry I could not travel both
# And be one traveler, long I stood
# And looked down one as far as I could
# To where it bent in the undergrowth;
#
# Then took the other, as just as fair,
# And having perhaps the better claim
# ...


with open('walt_whitman/walt_whitman.txt', encoding='utf-8') as f:
    for line in f:
        tokens = remove_punctuation(line.rstrip().lower()).split()      # "The woods are" -> ['the', 'woods', 'are']
        T = len(tokens)     # tokens = ['the', 'woods', 'are', 'lovely'] -> T=4
        for i in range(T):
            t = tokens[i]
            if i == 0:
                # measure the distribution of the first word
                initial[t] = initial.get(t, 0.) + 1     # initial = {'the': 10, 'two': 5, 'whose': 2, 'i': 3}
            else:
                t_1 = tokens[i - 1]
                if i == T - 1:
                    # measure probability of ending the line
                    add2dict(transitions, (t_1, t), 'END')  # P(END|w(t), w(t-1)) -> # transitions = {('i', 'like'): ['END']}
                if i == 1:
                    # measure distribution of second word
                    add2dict(second_word, t_1, t)   # P(w(t)|w(t-1)) -> second_word = {'the': ['wood']}
                else:
                    t_2 = tokens[i - 2]
                    add2dict(transitions, (t_2, t_1), t)    # P(w(t)|w(t-1), w(t-2)) -> # transitions = {('i', 'like'): ['bread']}

# normalize the distributions
initial_total = sum(initial.values())   # initial = {'the': 10, 'two': 5, 'whose': 2, 'i': 3}
for t, c in initial.items():
    initial[t] = c / initial_total      # initial_total = {'the': 10/20, 'two': 5/20, 'whose': 2, 'i': 3/20}

# turn each list of possibilities into a dictionary of probabilities
def list2pdict(ts):
    d = {}
    n = len(ts)
    for t in ts:
        d[t] = d.get(t, 0.) + 1
    for t, c in d.items():
        d[t] = c / n
    return d

for t_1, ts in second_word.items():     # P(w(t)|w(t-1)) -> second_word = {'the': ['woods', 'road', 'woods'], ...}
    second_word[t_1] = list2pdict(ts)   # second_word = {'the': {'woods': 2/3, 'road': 1/3}, ...}

for k, ts in transitions.items():       # P(w(t) | w(t-2), w(t-1)) -> transitions = {('the', 'woods'): ['are', 'are', 'whisper'], ...}
    transitions[k] = list2pdict(ts)     # transitions = {('the', 'woods'): {'are': 2/3, 'whisper': 1/3}, ...}

# generate 4 lines
def sample_word(d):
    p0 = np.random.random()
    cumulative = 0
    for t, p in d.items():
        cumulative += p
        if p0 < cumulative:
            return t
    raise RuntimeError("Sampling failed — check if probabilities sum to 1.")

def generate():
    for i in range(4):    # i=0
        sentence = []

        # initial word
        w0 = sample_word(initial)   # initial_total = {'the': 10/20, 'two': 5/20, ...}
                                    # Random p -> 'two'
        sentence.append(w0)

        # sample second word
        w1 = sample_word(second_word[w0])   # second_word = {'two': {'roads': 1.0}, ...}
                                            # Random p -> 'two' -> 'roads'
        sentence.append(w1)

        # second-order transitions until END
        while True:
            w2 = sample_word(transitions[(w0, w1)])     # transitions = {('two', 'roads'): {'diverged': 1.0, 'END': 0.0}, ...}
                                                        # Random p -> 'two' -> 'roads' -> 'diverged'
            if w2 == 'END':
                break
            sentence.append(w2)
            w0 = w1
            w1 = w2

        print(' '.join(sentence))

generate()
