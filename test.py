import unittest
import numpy as np
import os
from generate import daubechies

class TestDaubechies(unittest.TestCase):
    def test_all(self):
        for N in range(2,100): 
            scaling = np.loadtxt(os.path.join('coefficients', f'db{N:02d}_coefficients.txt'))

            # test summation
            x = np.abs(scaling.sum() - np.sqrt(2.0)) 
            self.assertLess(x, 1.0e-10, f'test summation N={N}')

            # test orthogonality
            for M in range(2, 2*N, 2):
                x = np.abs(np.sum([s1*s2 for s1, s2 in zip(scaling, scaling[M:])]))
                self.assertLess(x, 1.0e-10, f'test orthogonality N={N}, M={M}')

            # test vanishing moments(max:4)
            for L in range(0, min(4, N)):
                x = np.abs(np.sum([(1 if k%2==0 else -1)*s*(k**L) for k, s in enumerate(scaling)]))
                self.assertLess(x, 1.0e-10, f'test vanishing moments(max:4) N={N}, L={L}')

if __name__ == '__main__':
    unittest.main()