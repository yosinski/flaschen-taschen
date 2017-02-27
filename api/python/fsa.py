#! /usr/bin/env python

import numpy as np
import flaschen

def fsa_line(in_line, one_patterns, pad_with = False):
    '''Operates on bool array'''
    #one_patterns_head = [op[0:2] for op in one_patterns]
    #one_patterns_tail = [op[1:3] for op in one_patterns]
    _in_line_padded = np.pad(in_line, 1, 'constant', constant_values=pad_with)
    ret = np.zeros(in_line.shape[0], dtype='bool')    # All false to start
    if not isinstance(in_line, np.ndarray):
        in_line = np.array(in_line)
    if not isinstance(one_patterns, np.ndarray):
        one_patterns = np.array(one_patterns)
    if len(one_patterns.shape) == 1:
        one_patterns = one_patterns[np.newaxis]
    for pp in xrange(one_patterns.shape[0]):
        conv = np.correlate(_in_line_padded.astype('int')*2-1,
                            one_patterns[pp].astype('int')*2-1,
                            'valid')
        ret |= (conv >= 3)

    return ret

class FlaschenFSA(object):
    def __init__(self, ff, line0, one_patterns):
        self.ff = ff     # flaschen
        self.line = line0.copy()
        self.one_patterns = one_patterns
        self.store_line()
        
    def store_line(self):
        self.ff.data[0] = map(lambda xx:[0, 255, 0] if xx else [255, 0, 0], self.line)
        
    def step(self):
        self.line = fsa_line(self.line, self.one_patterns)
        self.ff.data[1:,:,:] = self.ff.data[:-1,:,:]
        self.store_line()
        
    def send(self):
        self.ff.send()

def main():
    ff = flaschen.Flaschen()

if __name__ == '__main__':
    main()
