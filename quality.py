from __future__ import print_function

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import copy
import json
import math
import numpy as np
import os
import pickle
import pprint
import random
import sys
import time
import gc

from collections import defaultdict

verbosity = 1

metadata_jitter = np.array([
    # file properties
    0.1, # width
    0.1, # file size
    0.1, # color count
    0.1, # quality
    0  , # is PNG?
    0  , # is JPEG?
    0  , # is GIF?
    # sample properties
    0  , # color count
    0  , # entropy
])
num_metadata = len(metadata_jitter) # Number of metadata variables per sample

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()

class Data:
    pass

def load_data(fraction=1):
    log('Loading data ...')

    basefn = "data-8x8-size_0-samples_512-order_entropy-filter_false-tests_10000"

    d = Data()
    d.pixels   = np.memmap(basefn + "-pixels-none.u8", dtype=(np.uint8  , (64,)), mode='r')
    d.dct      = np.memmap(basefn + "-pixels-dct.f32", dtype=(np.float32, (64,)), mode='r')
    d.labels   = np.memmap(basefn + "-labels.u8"     , dtype='uint8'            , mode='r')
    d.metadata = np.memmap(basefn + "-metadata.f32"  , dtype=(np.float32, (num_metadata,)) , mode='r')
    d.indices  = np.memmap(basefn + "-indices.u32"   , dtype=np.uint32          , mode='r')

    # Use a smaller dataset during hypersearch
    if fraction != 1:
        h_num_tests = d.indices[-1] + 1
        h_test_cutoff = int(h_num_tests * fraction)
        h_cutoff = np.where(d.indices == h_test_cutoff)[0][0]

        d.pixels   = d.pixels  [:h_cutoff]
        d.dct      = d.dct     [:h_cutoff]
        d.labels   = d.labels  [:h_cutoff]
        d.metadata = d.metadata[:h_cutoff]
        d.indices  = d.indices [:h_cutoff]

    d.pixels = d.pixels.astype('float32')
    d.pixels = d.pixels.reshape((len(d.pixels), 8, 8, 1))
    d.pixels /= 255

    d.dct = np.copy(d.dct)
    d.dct += 2048
    d.dct /= 4096

    d.num_samples = len(d.pixels)
    assert len(d.pixels  ) == d.num_samples
    assert len(d.dct     ) == d.num_samples
    assert len(d.labels  ) == d.num_samples
    assert len(d.metadata) == d.num_samples
    assert len(d.indices ) == d.num_samples

    d.num_tests = d.indices[-1] + 1 # indices[x] < num_tests for any x
    d.test_cutoff = d.num_tests * 3 // 4
    d.cutoff = np.where(d.indices == d.test_cutoff)[0][0]

    log('Train/total: tests:%d/%d samples:%d/%d' %
          (d.test_cutoff, d.num_tests, d.cutoff, d.num_samples))

    if keras.backend.floatx() != 'float32':
        d.pixels   = d.pixels  .astype(keras.backend.floatx())
        d.dct      = d.dct     .astype(keras.backend.floatx())
        d.metadata = d.metadata.astype(keras.backend.floatx())

    d.noisy_metadata = np.random.rand(d.num_samples, num_metadata)
    d.noisy_metadata -= 0.5
    d.noisy_metadata *= metadata_jitter
    d.noisy_metadata *= d.metadata
    d.noisy_metadata += d.metadata

    d.inputs = [
        d.pixels,
        d.dct,
        d.noisy_metadata,
    ]

    d.x_train = [arr[:d.cutoff] for arr in d.inputs]
    d.x_test  = [arr[d.cutoff:] for arr in d.inputs]
    d.y_train = d.labels[:d.cutoff]
    d.y_test  = d.labels[d.cutoff:]

    return d

class Problem:
    def __init__(self):
        self.model_id = None

def calc_real_accuracy(problem):
    p = problem
    d = p.data

    y = p.model.predict(d.inputs, batch_size=4096, verbose=verbosity)
    assert len(y) == d.num_samples

    sums = np.zeros((d.num_tests,2)) # per-image sample score sum
    counts = np.zeros((d.num_tests,2)) # per-image sample count
    for i in range(d.num_samples):
        sums[d.indices[i], d.labels[i]] += y[i][0]
        counts[d.indices[i], d.labels[i]] += 1
    totals = np.divide(sums, counts) # per-image average sample score
    return calc_real_accuracy_from_data(totals, d.num_tests, d.test_cutoff)

def calc_real_accuracy_from_data(totals, num_tests, test_cutoff):
    for which in range(2):
        t_start = 0 if which == 0 else test_cutoff
        t_end = test_cutoff if which == 0 else num_tests

        hits = 0
        for t in range(t_start, t_end):
            #assert counts[t][0] > 0 and counts[t][1] > 0
            if totals[t][0] < totals[t][1]:
                hits += 1
            elif totals[t][0] > totals[t][1]:
                pass
            else:
                hits += 0.5
        tests = t_end - t_start
        accuracy = float(hits) / tests
        log('Real', 'train' if which == 0 else 'test ', 'accuracy:', accuracy)
        if which == 1:
            real_accuracy = accuracy

    if os.path.exists("tests/%04d-info.json" % (0)):
        log("Best/worst transforms:")
        seen = defaultdict(int)
        good = defaultdict(int)
        for t in range(num_tests):
            hit = totals[t][0] < totals[t][1]
            info = json.load(open("tests/%04d-info.json" % (t)))
            for word in info['imCmdLine']:
                seen[word] += 1
                if hit:
                    good[word] += 1

        word_score = lambda word : good.get(word, 0) / float(seen[word])
        words = seen.keys()
        words = sorted(words, key=word_score)
        for word in words:
            if seen[word] > num_tests / 10:
                log(" %1.3f\t%4d/%5d\t%s" % (word_score(word), good[word], seen[word], word))

    tests = [i for i in range(num_tests)]
    test_score = lambda t : totals[t][1] - totals[t][0]
    tests = sorted(tests, key=test_score)
    for name, testseg in [('Worst', tests[:10]), ('Best ', tests[-10:])]:
        log("%s tests:" % (name))
        for t in testseg:
            infofn = "tests/%04d-info.json" % (t)
            info = ' '.join(json.load(open(infofn))['imCmdLine'][2:-1]) if os.path.exists(infofn) else ''
            log(" %04d %1.3f %s" % (t, test_score(t), info))

    gc.collect()
    keras.backend.clear_session()
    gc.collect()

    return real_accuracy    

# Constants

optimizers_list = [
    'SGD'      ,
    'NSGD'     ,
    'RMSprop'  ,
    'Adagrad'  ,
    'Adadelta' ,
    'Adam'     ,
    'Adamax'   ,
    'Nadam'    ,
]
optimizers = {
    'SGD'      : keras.optimizers.SGD(),
    'NSGD'     : keras.optimizers.SGD(nesterov=True),
    'RMSprop'  : keras.optimizers.RMSprop(),
    'Adagrad'  : keras.optimizers.Adagrad(),
    'Adadelta' : keras.optimizers.Adadelta(),
    'Adam'     : keras.optimizers.Adam(),
    'Adamax'   : keras.optimizers.Adamax(),
    'Nadam'    : keras.optimizers.Nadam(),
}
assert sorted(optimizers_list) == sorted(optimizers.keys())
    
def build_model(problem):
    p = problem

    p.model = keras.models.Model(
        inputs=[
            p.l_pi,
            p.l_di,
            p.l_mi,
        ],
        outputs=p.l_out,
    )


def compile_model(problem):
    p = problem

    build_model(p)
    p.model.summary(print_fn=log)

    p.metaparams={
        'batch_size' : p.batch_size,
        'floatx' : keras.backend.floatx(),
        'loss' : p.loss,
        'optimizer' : p.optimizer,
    }
    for name in p.metaparams:
        log('%-15s : %s' % (name, p.metaparams[name]))

    p.model.compile(loss=p.loss,
                    optimizer=optimizers[p.optimizer],
                    metrics=['binary_accuracy'])

    p.model_id = '%016x' % hash(p.model.to_json() + json.dumps(p.metaparams))
    log('%-15s : %s' % ('Model ID', p.model_id))


def save_model_structure(problem, require_unique=False):
    p = problem

    if require_unique and os.path.exists('quality-%s.yaml' % (p.model_id)):
        raise Exception('Model '+p.model_id+' already examined')
    open('quality-%s.yaml' % (p.model_id), 'w').write(p.model.to_yaml())
    json.dump(p.metaparams, open('quality-%s.json' % (p.model_id), 'w'))


def cmd_diff_model(model_desc_base_fn):
    p = Problem()

    make_layers(p)
    build_model(p)

    if hasattr(p, 'metaparams'):
        json.dump(p.metaparams, open('diff_model.json', 'w'))
        os.system('diff --color -u diff_model.json %s.json' % (model_id, model_desc_base_fn))

    open('diff_model.yaml', 'w').write(p.model.to_yaml())
    for b in ('diff_model', model_desc_base_fn):
        import yaml
        y = yaml.load(open('%s.yaml' % b))
        y['config']['layers'] = sorted(y['config']['layers'], key=lambda l : l['name'])
        yaml.dump(y, open('%s.s.yaml' % b, 'w'))
    import subprocess
    subprocess.Popen(['/bin/bash', '-c',
                      'diff --color -u -U 10 ' +
                      ' '.join([
                          "<(< %s sed -e \"s#!!python/unicode '\\([^']*\\)'#\\1#g\" -e 's#&id... !!python/tuple#!!python/tuple#g')" % (fn)
                          for fn in ['diff_model.s.yaml', '%s.s.yaml' % (model_desc_base_fn)]])])


def fit(problem, controller):
    p = problem
    c = controller
    d = p.data

    class SearchState:
        pass
    s = SearchState()

    s.search_start = s.now = time.time()
    s.real_accuracy_history = []
    s.epoch = 0

    while True:
        if c.should_calc_accuracy(s):
            acc = calc_real_accuracy(p)
            if acc == 0.5:
                log('Real accuracy == 0.5 (vanishing gradient?)')
                break
            if len(s.real_accuracy_history) == 0 or acc > max(s.real_accuracy_history):
                log('> New best!')
                p.model.save('quality-%s-%s.h5' % (p.model_id, acc))
            s.real_accuracy_history.append(acc)

        if c.should_stop(s):
            break

        next_epoch = c.get_next_epoch(s)
        history = p.model.fit(
            d.x_train, d.y_train,
            batch_size=p.batch_size,
            initial_epoch=s.epoch,
            epochs=next_epoch,
            verbose=verbosity,
            validation_data=(d.x_test, d.y_test))
        if history.history['binary_accuracy'][-1] == 0.5:
            log('binary_accuracy == 0.5 (vanishing gradient?)')
            break
        # if history.history['val_binary_accuracy'][-1] < 0.5001:
        #     log('val_binary_accuracy < 0.5001 (crap result)')
        #     break
        if math.isnan(history.history['loss'][-1]):
            log('Loss is NaN')
            break
        s.epoch = next_epoch
        s.now = time.time()

    log('=======================================================================================================================================================')
    if len(s.real_accuracy_history) > 0:
        log('Best real accuracy seen:', max(s.real_accuracy_history))

    return s


def fit_model(problem):
    p = problem

    p.data = load_data()

    class Controller:
        plateau_epochs = 50 # stop if real accuracy not improving after this many epochs
        calc_real_accuracy_batch_interval = 5

        def should_calc_accuracy(self, s):
            return True

        def should_stop(self, s):
            tail_len = self.plateau_epochs // self.calc_real_accuracy_batch_interval
            if len(s.real_accuracy_history) > tail_len:
                head = s.real_accuracy_history[:-tail_len]
                tail = s.real_accuracy_history[-tail_len:]
                if max(head) >= max(tail):
                    log('Real accuracy plateaued, stopping.')
                    return True
            return False

        def get_next_epoch(self, s):
            return s.epoch + self.calc_real_accuracy_batch_interval

    c = Controller()
    fit(p, c)


def make_layers(problem):
    p = problem

    # Pixel data (convolution)
    l = p.l_pi = Input(shape=(8,8,1))
    l = Conv2D(64, (3, 3), activation='relu')(l)
    l = Conv2D(64, (2, 4), activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 1))(l)
    l = Dropout(0.25)(l)
    l = Flatten()(l)
    l_p = l

    # DCT-preprocessed pixels
    l = p.l_di = Input(shape=(64,))
    l = Dense(4, activation='linear')(l)
    l = Dropout(0.25)(l)
    l = Dense(16, activation='relu')(l)
    l = Dropout(0.25)(l)
    l = Dense(4, activation='linear')(l)
    l = Dropout(0.25)(l)
    l = Dense(4, activation='tanh')(l)
    l = Dropout(0.75)(l)
    l_d = l

    # Metadata
    l = p.l_mi = Input(shape=(num_metadata,))
    l = Dropout(0.75)(l)
    l = Dense(16, activation='selu')(l)
    l = Dropout(0.75)(l)
    l_m = l

    # Merge
    l_c = Concatenate()([
        l_p,
        l_d,
        l_m,
    ])

    # Hidden
    l = l_c
    l_h = l

    # Output
    p.l_out = Dense(1, activation='sigmoid')(l_h)


def cmd_fit():
    p = Problem()

    p.batch_size = 512
    keras.backend.set_floatx('float32')
    p.loss = 'mean_squared_error'
    p.optimizer = 'Adam'

    # Set up layers
    make_layers(p)

    compile_model(p)
    save_model_structure(p)

    fit_model(p)

def cmd_fit_model(model_fn):
    p = Problem()

    p.batch_size = 2048
    keras.backend.set_floatx('float32')
    p.loss = 'mean_squared_error'
    p.optimizer = 'Adam'

    p.model = keras.models.load_model(model_fn)

    fit_model(p)

class HyperSearch:
    floatx_list = [
        # 'float16', # seems to fail every time
        'float32',
        # 'float64', # not in ROCm yet
    ]
    losses_list = [
        'mean_squared_error', 'mean_absolute_error',
        'mean_squared_logarithmic_error', 'hinge',
        'logcosh', 'poisson',
    ]
    activations_list = [
        'softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu',
        'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear'
    ]

    structure = (
        # Metaparameters
        (
            7, # batch size (1<<7 .. 1<<13)
            len(floatx_list),
            len(losses_list),
            len(optimizers_list),
            len(activations_list),
        ),

        # Generic fragments
        (
            [(
                8, # layer outputs (1<<0 .. 1<<7)
                len(activations_list),
                4, # dropout
            )],
        ) * 4,

        # Convolution layers
        [(
            8, # output filters
            5, 5, # convolution size (+1)
            len(activations_list),
            25, # pooling (w*5+h)
            4, # dropout
        )],
    )

    initial_tree = [
        [
            # batch size
            2, # 1 << (7 + 2) == 512
            # floatx
            0, # float32 - only choice
            # loss
            0, # mean_squared_error
            # optimizer
            5, # Adam
            # final layer activation function
            7, # sigmoid
        ],

        # Generic fragments
        [
            # Pixel data (post-convolution)
            [],
            # DCT
            [],
            # Metadata
            [],
            # Hidden (post-concatenation)
            [],
        ],

        # Convolution fragment
        [],
    ]

    max_siblings = 4

    def __init__(self):
        if os.path.exists('history.p'):
            self.model_history = pickle.load(open('history.p', 'rb'))
        else:
            self.model_history = []

        for h in self.model_history:
            self.fix_tree(h['tree'])
        self.model_history = list(filter(lambda h : self.is_tree_valid(h['tree']), self.model_history))

        self.data = None

    def traverse_tree(self, node, struc=structure, addr=[]):
        # Tree is made of ints, tuples, and lists
        # Can modify int by changing it to a random value
        # Can modify list by deleting/inserting children
        # Can't modify tuple
        # Can descend into tuple and list
        addresses = []
        if isinstance(struc, int):
            addresses += [addr]
        elif isinstance(struc, list):
            addresses += [addr]
            for i in range(len(node)):
                addresses += self.traverse_tree(node[i], struc[0], addr + [i])
        elif isinstance(struc, tuple):
            for i in range(len(node)):
                addresses += self.traverse_tree(node[i], struc[i], addr + [i])
        else:
            assert False
        return addresses

    def new_tree(self, struc=structure):
        if isinstance(struc, int):
            return random.randrange(struc)
        if isinstance(struc, tuple):
            return [self.new_tree(x) for x in struc]
        if isinstance(struc, list):
            return [self.new_tree(struc[0]) for i in range(random.randint(0, self.max_siblings))]
        assert False

    def mutate_node(self, node, struc):
        if isinstance(struc, int):
            return random.randrange(struc)
        # list
        if random.randint(0, 1) == 0: # insert
            if len(node) < self.max_siblings:
                if len(node) == 0 or random.randint(0, 1) == 0: # random
                    v = self.new_tree(struc[0])
                else: # clone
                    v = copy.deepcopy(random.choice(node))
                node.insert(random.randint(0, len(node)), v)
            return node
        else: # delete
            if len(node) > 0:
                del node[random.randint(0, len(node)-1)]
            return node

    def flatten_tree(self, node, struc=structure):
        if isinstance(node, int):
            return [node == i for i in range(struc)]
        result = []
        if isinstance(struc, tuple):
            for i in range(len(node)):
                result += self.flatten_tree(node[i], struc[i])
        if isinstance(struc, list):
            assert len(node) <= self.max_siblings
            for i in range(len(node)):
                result += self.flatten_tree(node[i], struc[0])
            pad = [False] * len(self.flatten_tree(self.new_tree(struc[0]), struc[0]))
            for i in range(self.max_siblings - len(node)):
                result += pad
        return result

    def fix_tree(self, node, struc=structure):
        fixed = False
        for i in range(len(node)):
            struc_i = struc[0] if isinstance(struc, list) else struc[i]
            if isinstance(node[i], int) and node[i] >= struc_i:
                node[i] %= struc_i
                fixed = True
            if isinstance(node[i], list):
                fixed |= self.fix_tree(node[i], struc_i)
        return fixed

    def is_tree_valid(self, tree):
        for dim in range(2):
            x = 8
            for ld in tree[2]: # convolution fragment
                w = 1 + ld[1 + dim]
                x = x + 1 - w
                if x <= 0:
                    return False
                v = ld[4]
                w = (1 + v % 5, 1 + v // 5)[dim]
                x = x // w
                if x == 0:
                    return False
        return True

    def mutate_tree(self, orig_tree):
        while True:
            model_tree = copy.deepcopy(orig_tree)
            for mi in range(random.randint(1, 3)):
                addresses = self.traverse_tree(model_tree)
                target_addr = random.choice(addresses)
                target_struc = self.structure
                for a in target_addr:
                    if isinstance(target_struc, list):
                        target_struc = target_struc[0]
                    else: # tuple
                        target_struc = target_struc[a]
                target_node = model_tree
                for a in target_addr[:-1]:
                    target_node = target_node[a]
                a = target_addr[-1]
                target_node[a] = self.mutate_node(target_node[a], target_struc)
            if self.is_tree_valid(model_tree):
                return model_tree

    def gen_tree(self):
        if len(self.model_history) > 0:
            # shuffle for validation split
            model_history = self.model_history[:]
            random.shuffle(model_history)

            x = np.asarray([self.flatten_tree(h['tree']) for h in model_history], dtype=np.uint8)
            y = np.asarray([h['score'] for h in model_history], dtype=np.float32)
            x_seen = np.add.reduce(x)
            log('Seen inputs: %d/%d' % (np.count_nonzero(x_seen), len(x[0])))

            model = Sequential()
            initializer = keras.initializers.RandomUniform(minval=-1, maxval=1)
            model.add(Dense(16,
                            activation='sigmoid',
                            kernel_initializer=initializer,
                            bias_initializer=initializer))
            model.add(Dropout(0.5))
            model.add(Dense(1,
                            activation='sigmoid',
                            kernel_initializer=initializer,
                            bias_initializer=initializer))
            model.compile(loss='mean_squared_error', optimizer=Adam())
            split = False # len(model_history) > 50
            history = model.fit(x, y,
                                batch_size=16,
                                epochs=1000,
                                callbacks=[
                                    keras.callbacks.EarlyStopping(
                                        monitor='loss',
                                        patience=50,
                                        restore_best_weights=True,
                                    )
                                ],
                                verbose=2 if verbosity>0 else 0,
                                validation_split=0.5 if split else 0.0,
            )

            model_history_sorted = sorted(model_history, key=lambda p : p['score'])
            population = [p['tree'] for p in model_history_sorted]
            num_trees = 10000

            best_score = 0
            while True:
                log('Generating and evaluating %d random trees...' % num_trees)
                population = [
                    self.mutate_tree(
                        population[
                            int((1 - random.expovariate(15)) * len(population))
                        ]
                    )
                    for i in range(num_trees)
                ] + [population[-1]] # include previous best

                px = np.asarray([self.flatten_tree(t) for t in population], dtype=np.uint8)
                py = model.predict(px, verbose=verbosity)
                porder = py.flatten().argsort()
                log('Best score: %s' % py[porder[-1]])
                if py[porder[-1]] == best_score:
                    best_tree = population[porder[-1]]
                    break
                else:
                    assert best_score < py[porder[-1]]
                    best_score = py[porder[-1]]
                    population = [population[i] for i in porder]

            best_tree_flattened = self.flatten_tree(best_tree)
            new_inputs = sum([1 if best_tree_flattened[i] > 0 and x_seen[i] == 0 else 0 for i in range(len(best_tree_flattened))])
            log('Best tree (predicted score = %s, %d untried inputs):' % (best_score, new_inputs))
            pprint.pprint(best_tree)

            x_diffs = np.add.reduce(np.equal(x, best_tree_flattened), axis=1)
            closest_index = np.argmax(x_diffs)
            log('Closest known tree (score = %s, hits = %d/%d, distance = %d/%d/%d):' % (
                model_history[closest_index]['score'],
                x_diffs[closest_index], len(best_tree_flattened),
                len(best_tree_flattened) - x_diffs[closest_index],
                np.count_nonzero(best_tree_flattened),
                np.count_nonzero(x[closest_index]),
            ))
            pprint.pprint(model_history[closest_index]['tree'])

            return best_tree
        else:
            return self.initial_tree

    def gen_model(self, problem, model_tree):
        p = problem

        p.batch_size = 1 << (7 + model_tree[0][0])
        keras.backend.set_floatx(self.floatx_list[model_tree[0][1]])
        p.loss = self.losses_list[model_tree[0][2]]
        p.optimizer = optimizers_list[model_tree[0][3]]

        # Pixel data (convolution)
        l = p.l_pi = Input(shape=(8,8,1))
        for ld in model_tree[2]:
            l = Conv2D(1 << ld[0], (1 + ld[1], 1 + ld[2]), activation=self.activations_list[ld[3]])(l)
            v = ld[4]
            if v != 0:
                l = MaxPooling2D(pool_size=(1 + v % 5, 1 + v // 5))(l)
            v = [0, 0.25, 0.5, 0.75][ld[5]]
            if v != 0:
                l = Dropout(v)(l)
        l = Flatten()(l)
        for ld in model_tree[1][0]:
            l = Dense(1 << ld[0], activation=self.activations_list[ld[1]])(l)
            v = [0, 0.25, 0.5, 0.75][ld[2]]
            if v != 0:
                l = Dropout(v)(l)
        l_p = l

        # DCT-preprocessed pixels
        l = p.l_di = Input(shape=(64,))
        for ld in model_tree[1][1]:
            l = Dense(1 << ld[0], activation=self.activations_list[ld[1]])(l)
            v = [0, 0.25, 0.5, 0.75][ld[2]]
            if v != 0:
                l = Dropout(v)(l)
        l_d = l

        # Metadata
        l = p.l_mi = Input(shape=(num_metadata,))
        for ld in model_tree[1][2]:
            l = Dense(1 << ld[0], activation=self.activations_list[ld[1]])(l)
            v = [0, 0.25, 0.5, 0.75][ld[2]]
            if v != 0:
                l = Dropout(v)(l)
        l_m = l

        # Merge
        l_c = Concatenate()([
            l_p,
            l_d,
            l_m,
        ])

        # Hidden
        l = l_c
        for ld in model_tree[1][3]:
            l = Dense(1 << ld[0], activation=self.activations_list[ld[1]])(l)
            v = [0, 0.25, 0.5, 0.75][ld[2]]
            if v != 0:
                l = Dropout(v)(l)
        l_h = l

        # Output
        p.l_out = Dense(1, activation=self.activations_list[model_tree[0][4]])(l_h)

    def get_data(self):
        if self.data is None:
            self.data = load_data(fraction = 0.25)
        return self.data

    def eval_tree(self, problem, model_tree):
        p = problem
        self.gen_model(p, model_tree)
        compile_model(p)
        save_model_structure(p, require_unique = True)

        p.data = self.get_data()

        class Controller:
            search_duration = 5 * 60

            def should_calc_accuracy(self, s):
                return s.now - s.search_start > self.search_duration

            def should_stop(self, s):
                if s.now - s.search_start > self.search_duration:
                    log('Search duration exceeded, stopping.')
                    return True
                return False

            def get_next_epoch(self, s):
                return s.epoch + 1

        c = Controller()
        s = fit(p, c)

        if len(s.real_accuracy_history) > 0:
            score = max(s.real_accuracy_history)
        else:
            score = 0
        return score

    def search(self, model_tree=None):
        if model_tree is None:
            model_tree = self.gen_tree()

        p = Problem()
        score = self.eval_tree(p, model_tree)

        score_pos = sum(1 for h in self.model_history if score > h['score'])
        log('Score %s beat %d/%d past results.' % (
            score, score_pos, len(self.model_history),
        ))
        if score_pos == len(self.model_history):
            log('> New record!')
        self.model_history += [{
            'tree' : model_tree,
            'model_id' : p.model_id,
            'score' : score,
        }]
        pickle.dump(self.model_history, open('history.p', 'wb'))

def cmd_hypersearch():
    h = HyperSearch()
    h.search()

def cmd_hypersearch_loop():
    h = HyperSearch()
    start = time.time()
    while time.time() - start < (5 * 60 + 30) * 60:
        try:
            h.search()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(e)
        time.sleep(1)
    log('Reaching Kaggle timeout, stopping.')

def cmd_eval(model_fn):
    p = Problem()

    p.model = keras.models.load_model(model_fn)
    p.data = load_data()

    calc_real_accuracy(p)


def cmd_fit_summarizer():
    evaluator_model_fn = 'quality.h5'

    p = Problem()

    p.model = keras.models.load_model(evaluator_model_fn)
    p.data = load_data()
    d = p.data

    # Three dimensions:
    # - X samples. One sample is one sequence. Each sample consists of:
    #   - Y timesteps. One timestep represents is a point in time and includes the features in it. Each timestep consists of:
    #     - Z features. Each feature is a datum observation.

    # Note on confusing terminology - in the local variables in this function,
    # "samples" refers to RNN sequences, but in "d", it refers to the 8x8 blocks.

    num_features  = num_metadata + 3 # metadata + presence + sample score (prediction from sample evaluator model) + dumb average
    num_timesteps = 0 # sequence/sample length - will fill out below
    num_samples   = 0 # number of sequences - will fill out below

    log('Segmenting data...')
    cur_timestep = 0
    is_edge = (d.indices[1:] != d.indices[:-1]) | (d.labels[1:] != d.labels[:-1])
    for i in range(d.num_samples):
        if i == 0 or is_edge[i-1]:
            num_samples += 1
            cur_timestep = 0
        cur_timestep += 1
        if num_timesteps < cur_timestep:
            num_timesteps = cur_timestep

    log('Input has %d samples (series), with at most %d timesteps, and %d features.' %
        (num_samples, num_timesteps, num_features))
    assert num_samples == d.num_tests * 2 # image with no samples?

    log('Running sample evaluator prediction...')
    d = p.data
    y = p.model.predict(d.inputs, batch_size=4096, verbose=verbosity)
    assert len(y) == d.num_samples

    log('Calculating dumb averages...')
    sums = np.zeros((d.num_tests,2)) # per-image sample score sum
    counts = np.zeros((d.num_tests,2)) # per-image sample count
    for i in range(d.num_samples):
        sums[d.indices[i], d.labels[i]] += y[i][0]
        counts[d.indices[i], d.labels[i]] += 1
    totals = np.divide(sums, counts) # per-image average sample score
    # log('REAL ACCURACY (INPUT) vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
    # calc_real_accuracy_from_data(totals, d.num_tests, d.test_cutoff)
    # log('REAL ACCURACY (INPUT) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    # The summarizer output should be at least as good as the dumb average.
    print('Minimal expected results: ', np.sum(totals[:, 0] < totals[:, 1]), '/', d.num_tests)

    inputs = np.zeros(shape=(num_samples, num_timesteps, num_features), dtype=np.float32)

    log('Populating data...')

    num_samples = 0
    for i in range(d.num_samples):
        if i == 0 or is_edge[i-1]:
            # labels[num_samples] = d.labels[i]
            # sample_index[d.indices[i], d.labels[i]] = i
            num_samples += 1
            cur_timestep = num_timesteps
        cur_timestep -= 1
        inputs[num_samples - 1, cur_timestep, 0] = totals[d.indices[i], d.labels[i]]
        inputs[num_samples - 1, cur_timestep, 1] = 1.0
        inputs[num_samples - 1, cur_timestep, 2] = y[i]
        inputs[num_samples - 1, cur_timestep, 3:] = d.metadata[i]

    # We want the output to indicate which of the two input images is better.
    # 0 means the left image is better, and 1 means the right image is better.
    # As such, if we were to just feed the NN our image pairs as is,
    # the expected output would be always 0.  Of course, we don't want to train
    # the NN to always output 0, so we repeat each test (image pair) again
    # with the images flipped, and expect a result of 1 for the flipped pair.

    # Group inputs in pairs, one pair per test, two images' inputs per pair (orig and edit)
    inputs = inputs.reshape((d.num_tests, 2, num_timesteps, num_features))
    # Add an additional dimension (2 rows), and move the old data to the first row
    inputs2 = np.empty((d.num_tests, 2, 2, num_timesteps, num_features))
    inputs2[:, 0] = inputs[:]
    inputs = inputs2
    # Create a swapped copy of each test (so that the NN doesn't just
    # learn to pick the left-hand one all the time)
    inputs[:, 1, 0] = inputs[:, 0, 1]
    inputs[:, 1, 1] = inputs[:, 0, 0]
    inputs = inputs.reshape((d.num_tests * 2, 2, num_timesteps, num_features))

    # Labels are pairs of tests, of which in the first test, the first image is better,
    # and in the second test, the second image is better.
    labels = np.array([0, 1], dtype=np.float32)
    labels = np.tile(labels, d.num_tests)
    labels = labels.reshape((d.num_tests * 2, 1))

    # log('Inputs:'); print(inputs[:16]); print('...'); print(inputs[-16:])
    # log('Labels:'); print(labels[:16]); print('...'); print(labels[-16:])

    log('Building model...')

    image_layers = []

    image_layers.append( LSTM(2) )
    # image_layers.append( Dense(16) )
    # image_layers.append( Dense(4, activation='selu') )

    image_layers.append( Dense(1, activation='sigmoid') )

    image_inputs = []
    image_outputs = []
    for which in range(2):
        l = Input(shape=(num_timesteps, num_features))
        image_inputs.append(l)
        for il in image_layers:
            l = il(l)
        image_outputs.append(l)

    l = Concatenate()(image_outputs)
    # l = Dense(4, activation='selu')(l)
    L_cmp = Dense(1, activation='sigmoid')
    l = L_cmp(l)

    model = keras.models.Model(
        inputs=image_inputs,
        outputs=l
    )

    r = 1e3
    L_cmp.set_weights([np.array([[r], [-r]]), np.array([0])])
    L_cmp.trainable = False

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
    model.summary()

    # print('Layers:', model.layers)

    model_id = '%016x' % hash(model.to_json())

    num_inputs = d.num_tests * 2
    cutoff = num_inputs * 3 // 4
    n_batch = 64
    best_real_accuracy = 0
    epoch = 0
    min_weight = 0
    base_wrong_weight = 0.1

    # weights = np.empty((d.num_tests, 2))

    while True:
        # print('Weights    : ', l.get_weights())
        if epoch % 5 == 0:
            predictions = model.predict(
                [inputs[:, which] for which in range(2)],
                batch_size=4096, verbose=verbosity)

            # for i in range(16):
            #     print(inputs[i], labels[i], predictions[i])
            # print('-------------------')

            predictions = predictions.reshape((d.num_tests, 2))
            totals = np.empty((d.num_tests, 2))
            totals[:, 0] = (predictions[:, 0] + (1 - predictions[:, 1])) / 2
            totals[:, 1] = 1 - totals[:, 0]

            # for i in range(16):
            #     print(i, inputs[i * 2], labels[i * 2], predictions[i], totals[i])
            # print('...')
            # for i in range(d.num_tests - 16, d.num_tests):
            #     print(i, inputs[i * 2], labels[i * 2], predictions[i], totals[i])
            # print('-------------------')

            # print(inputs.shape, labels.shape, predictions.shape, totals.shape)

            print('Real accuracy (left ): ', sum(np.round(1 - predictions[:, 0])), '/', d.num_tests)
            print('Real accuracy (right): ', sum(np.round(    predictions[:, 1])), '/', d.num_tests)

            real_accuracy = calc_real_accuracy_from_data(totals, d.num_tests, d.test_cutoff)

            if best_real_accuracy < real_accuracy:
                log('New best!')
                best_real_accuracy = real_accuracy
                model.save('summarizer-%s-%s.h5' % (model_id, real_accuracy))

        model.reset_states()
        model.fit([inputs[:cutoff, which] for which in range(2)],
                  labels[:cutoff],
                  epochs=1, batch_size=n_batch,
                  verbose=verbosity,
                  shuffle=True)

        epoch += 1

# extracts one branch of the training summarizer model, excluding the
# comparator, and returning a model which computes one file's score
def extract_summarizer(source_model):
    for i, layer in enumerate(source_model.layers):
        if isinstance(layer, Concatenate):
            last_layer = source_model.layers[i - 1]
            # Should be a single-output dense layer
            assert isinstance(last_layer, Dense)
            assert last_layer.output_shape == (None, 1)

            input = l = source_model.inputs[0]
            # assert isinstance(input, InputLayer)

            for j in range(1, i):
                L = source_model.layers[j]
                if not isinstance(L, InputLayer):
                    # log('Connecting', l, ' to ', L)
                    l = L(l)
                    # log(' -> ', l)

            model = keras.models.Model(
                inputs=[input],
                outputs=[l])
            model.compile()
            return model
    assert False

# replacement for np.fromfile which works with stdin to work around stupid bug in numpy
def np_fromfile(f, shape, dtype):
    arr = np.empty(shape=shape, dtype=dtype)
    f.readinto(memoryview(arr))
    return arr

def np_tofile(f, arr):
    f.write(memoryview(arr))

def cmd_server():
    model_fn = 'quality.h5'
    summarizer_model_fn = 'summarizer.h5'
    import struct

    model = keras.models.load_model(model_fn)

    summarizer_model = keras.models.load_model(summarizer_model_fn)
    summarizer_model = extract_summarizer(summarizer_model)

    num_timesteps = summarizer_model.inputs[0].shape[1]

    if hasattr(sys.stdin, 'buffer'):
        in_buf = sys.stdin.buffer
    else:
        in_buf = sys.stdin

    if hasattr(sys.stdout, 'buffer'):
        out_buf = sys.stdout.buffer
    else:
        out_buf = sys.stdout

    while True:
        log('Waiting for input.')

        num_samples_bytes = in_buf.read(4)
        if len(num_samples_bytes) == 0:
            break # EOF
        num_samples = struct.unpack('i', num_samples_bytes)[0]
        log('Reading %d samples.' % num_samples)

        pixels   = np_fromfile(in_buf, shape=(num_samples, 8, 8, 1     ,), dtype=np.uint8  )
        dct      = np_fromfile(in_buf, shape=(num_samples, 64          ,), dtype=np.float32)
        metadata = np_fromfile(in_buf, shape=(num_samples, num_metadata,), dtype=np.float32)

        log('Processing input.')

        pixels = pixels.astype('float32')
        pixels /= 255

        dct += 2048
        dct /= 4096

        inputs = [
            pixels,
            dct,
            metadata,
        ]

        log('Running prediction.')

        y = model.predict(inputs, batch_size=4096, verbose=0)
        assert len(y) == num_samples

        log('Calculating dumb average.')

        dumb_average = np.mean(y)

        log('Preparing summarizer data.')

        summarizer_input = np.zeros((1, num_timesteps, 3 + num_metadata))
        cur_timestep = num_timesteps
        for i in range(num_samples):
            cur_timestep -= 1
            summarizer_input[0, cur_timestep, 0] = dumb_average
            summarizer_input[0, cur_timestep, 1] = 1.0
            summarizer_input[0, cur_timestep, 2] = y[i]
            summarizer_input[0, cur_timestep, 3:] = metadata[i]

        log('Running summarizer.')

        s_y = summarizer_model.predict(summarizer_input, batch_size=256, verbose=0)
        assert len(s_y) == 1
        result = s_y

        log('Writing result.')

        np_tofile(out_buf, result)
        sys.stdout.flush()

def cmd_hypersearch_print_best():
    h = HyperSearch()
    pprint.pprint(sorted(h.model_history, key=lambda p : p['score'])[-1])

def cmd_hypersearch_try_tree():
    tree =  [[4, 0, 0, 5, 7],
             [[],
              [[2, 10, 1], [4, 5, 1], [2, 10, 1], [2, 6, 3]],
              [[4, 2, 1]],
              []],
             [
                 [6, 1, 1, 5, 0, 0],
                 [6, 1, 1, 5, 0, 0],
                 [6, 1, 1, 5, 1, 1],
             ]]

    h = HyperSearch()
    h.search(tree)

def cmd_extract_desc(model_fn):
    model = keras.models.load_model(model_fn)
    open('%s.yaml' % (model_fn), 'w').write(model.to_yaml())

def cmd_check():
    log('Python is OK')

def main():
    import argparse
    import inspect

    parser = argparse.ArgumentParser()

    fun = globals()['cmd_' + sys.argv[1]]

    if hasattr(inspect, 'getfullargspec'):
        spec = inspect.getfullargspec(fun)
    else:
        spec = inspect.getargspec(fun)

    num_defaults = len(spec.defaults) if spec.defaults is not None else 0
    for i in range(len(spec.args)):
        if i < len(spec.args) - num_defaults:
            parser.add_argument(spec.args[i])
        elif spec.defaults[i - len(spec.args)] is False:
            parser.add_argument('--' + spec.args[i],
                                default=False, action='store_true')
        else:
            default = spec.defaults[i - len(spec.args)]
            parser.add_argument('--' + spec.args[i],
                                default=default,
                                type=type(default))
    if spec.varargs is not None:
        parser.add_argument(spec.varargs,
                            nargs='*')

    kwargs = vars(parser.parse_args(sys.argv[2:]))
    args = []
    for arg in spec.args:
        args += [kwargs[arg]]
    if spec.varargs is not None:
        args += kwargs[spec.varargs]

    fun(*args)

if __name__ == "__main__":
    main()
