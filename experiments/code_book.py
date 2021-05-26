import math
import numpy as np
import pandas as pd


VEC_LEN = 4
L1_NORM = 8


get_bin = lambda x, n: format(x, 'b').zfill(n)


class Row(object):
    def __init__(self, index, point, normalized):
        self.index = index
        self.point = point
        self.normalized = normalized


class CodeBook(object):
    def __init__(self, vec_length, l1_norm_val):
        assert l1_norm_val > 0
        assert vec_length > 0

        self.book = []
        self.vec_length = vec_length
        self.l1_norm_val = l1_norm_val
        self._create_codebook()

    def get_range(self):
        return len(self.book)

    def _create_codebook(self):
        p = [-self.l1_norm_val] + [0] * (self.vec_length - 1)
        while True:
            if sum(abs(x) for x in p) == self.l1_norm_val:
                l2_norm = math.sqrt(sum(x ** 2 for x in p))
                normalized = tuple(x / l2_norm for x in p)
                #print(len(self.book), p, normalized)
                cb_instance = Row(len(self.book), tuple(p), normalized)
                self.book.append(cb_instance)

            index = np.nonzero(p)[-1][-1]
            if p[index] > 0:
                left_index = index - 1
                if p[left_index] == 0:
                    p[index] = -(p[index])
                    p[index] += 1
                    p[left_index] += 1
                else:
                    p[index] = -(p[index])
                    p[index] += -1 if p[left_index] < 0 else 1
                    p[left_index] += 1
            else:
                if index >= self.vec_length - 1:
                    p[index] = -(p[index])
                else:
                    p[index] += 1
                    p[index + 1] -= 1

            if p[0] == self.l1_norm_val:
                break

    def rescale_book(self, scale):
        for cb in self.book:
            cb.normalized = tuple(scale * x for x in cb.normalized)

    def find_nearest_pvq_code(self, value):
        assert len(value) == self.vec_length, f"{len(value)}, expected {self.vec_length}"
        ret = None
        min_dist = None
        for i in range(len(self.book)):
            q = self.book[i].normalized
            dist = math.sqrt(sum(abs(q[j] - value[j]) ** 2 for j in range(len(value))))
            if min_dist is None or dist < min_dist:
                ret = self.book[i]
                min_dist = dist

        return ret, min_dist


class Encoder(object):
    def __init__(self):
        self.cb = CodeBook(VEC_LEN, L1_NORM)
        self.range_of_number = int(math.log2(self.cb.get_range())) + 1

    def to_int_pow_rep(self, std):
        integer = None
        power = 1
        c_std = std
        for i in range(0, 10):
            if c_std * (10 ** i) >= 1.0:
                integer = c_std * 10 ** i
                power = i
                break
        return int(integer), int(power)

    def __call__(self, latent):
        """
        Transform array of float to bit stream:
        [010100101 | 010101010| 010101010101001001...10010100101]
        int(std)     pow(std)   array
        :param latent: list of floats
        :return: string
        """
        assert len(latent) >= self.cb.vec_length
        ret = ""

        ret += get_bin(len(latent), 16)
        scale = np.max([np.abs(np.min(latent)), np.abs(np.max(latent))]) #  np.std(latent) 
        scale /= 2
        value, power = self.to_int_pow_rep(scale)

        ret += get_bin(value, self.range_of_number)
        ret += get_bin((power), self.range_of_number)
        
        scale = value * 10 ** (-power)
        step = self.cb.vec_length
        self.cb.rescale_book(scale)
        print(self.cb.get_range(),value, power, scale)
        for i in range(0, len(latent), step):
            value = latent[i: (i + step)]
            if i + step >= len(latent):
                zeros_len = (i + step) - len(latent)
                value = latent[i: len(latent)] + [0] * zeros_len
            node, min_dist = self.cb.find_nearest_pvq_code(value)
            ret += get_bin(node.index, self.range_of_number)

        self.cb.rescale_book(1 / scale)
        return ret


class Decoder(object):
    def __init__(self):
        self.cb = CodeBook(VEC_LEN, L1_NORM)
        self.range_of_number = int(math.log2(self.cb.get_range())) + 1

    def __call__(self, stream):
        """

        :param stream:
        :return:
        """
        stream_len = int(stream[0:16], 2)
        value = int(stream[16: 16 + self.range_of_number], 2)
        power = - int(stream[16 + self.range_of_number:16 + (2*self.range_of_number)], 2)
        std = value * 10 ** power

        self.cb.rescale_book(std)
        ret = []
        for step in range(16 + 2 * self.range_of_number,
                          len(stream) ,
                          self.range_of_number):
            idx = int(stream[step:(step + self.range_of_number)], 2)

            x = self.cb.book[idx]
            ret.extend(list(x.normalized))
            #print(idx, x.normalized)
            #break

        self.cb.rescale_book(1 / std)
        return ret[:stream_len]

