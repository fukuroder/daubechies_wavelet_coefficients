import unittest
import mpmath
from generate import daubechies

class TestDaubechies(unittest.TestCase):
    def test_all(self):
        for N in range(2,100): 
            scaling = daubechies(N)

            # test summation
            x = mpmath.fabs(mpmath.fsum(scaling) - mpmath.sqrt('2')) 
            self.assertLess(x, 1.0e-40, f'test summation N={N}')

            # test orthogonality
            for M in range(2, 2*N, 2):
                x = mpmath.fabs(mpmath.fsum([s1*s2 for s1, s2 in zip(scaling, scaling[M:])]))
                self.assertLess(x, 1.0e-40, f'test orthogonality N={N}, M={M}')

            # test vanishing moments
            for L in range(1, N):
                x = mpmath.fabs(mpmath.fsum([(1 if k%2==0 else -1)*s*(k**L) for k, s in enumerate(scaling)]))
                self.assertLess(x, 1.0e-40, f'test vanishing moments N={N}, L={L}')

if __name__ == '__main__':
    unittest.main()