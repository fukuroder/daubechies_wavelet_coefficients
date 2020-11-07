import sympy
import mpmath
import scipy.signal
import matplotlib.pyplot as plt
import os
import tqdm

# precition
mpmath.mp.prec = 1024

def daubechies(N):
    # make polynomial
    q_y = [mpmath.binomial(N-1+k, k) for k in reversed(range(N))]

    # get polynomial roots y[k]
    y = mpmath.mp.polyroots(q_y, maxsteps=200, extraprec=64)

    z = []
    for yk in y:
        # subustitute y = -1/4z + 1/2 - 1/4/z to factor f(y) = y - y[k]
        f = [mpmath.mpf('-1/4'), mpmath.mpf('1/2') - yk, mpmath.mpf('-1/4')]

        # get polynomial roots z[k] within unit circle
        z += [ zk for zk in mpmath.mp.polyroots(f) if mpmath.fabs(zk) < 1 ]

    # make polynomial using the roots
    h0z = mpmath.sqrt('2')
    for zk in z:
        h0z *= sympy.sympify('(z-zk)/(1-zk)').subs('zk', zk)

    # adapt vanishing moments
    hz = (sympy.sympify('(1+z)/2')**N*h0z).expand()

    # get scaling coefficients
    return [sympy.re(hz.coeff('z', k)) for k in reversed(range(N*2))]

def main():
    coefficients_dir = 'coefficients'
    scaling_png_dir = 'scaling_png'
    wavelet_png_dir = 'wavelet_png'
    os.makedirs(coefficients_dir, exist_ok=True)
    os.makedirs(scaling_png_dir, exist_ok=True)
    os.makedirs(wavelet_png_dir, exist_ok=True)

    for N in tqdm.tqdm(range(2, 100)):
        # get dbN coeffients
        dbN = daubechies(N)

        # write coeffients
        lines = []
        lines.append(f'# db{N} scaling coefficients\n')
        for i, h in enumerate(dbN):
            lines.append(f'{mpmath.nstr(h, 40, min_fixed=0)}\n')
        with open(os.path.join(coefficients_dir, f'db{N:02d}_coefficients.txt'), 'w', newline='\n') as f:
            f.writelines(lines)

        # get an approximation of scaling function
        x, phi, psi = scipy.signal.cascade(dbN)

        # plot scaling function
        plt.plot(x, phi)
        plt.grid()
        plt.title(f'db{N} scaling function')
        plt.savefig(os.path.join(scaling_png_dir, f'db{N:02d}_scaling.png'))
        plt.clf()

        # plot wavelet
        plt.plot(x, psi)
        plt.grid()
        plt.title(f'db{N} wavelet')
        plt.savefig(os.path.join(wavelet_png_dir, f'db{N:02d}_wavelet.png'))
        plt.clf()

if __name__ == '__main__':
    main()
