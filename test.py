#!/usr/bin/env python3
import os
import sympy
import mpmath as sm
import scipy.signal
import matplotlib.pyplot as plt

# precision
sm.mp.prec = 512

def daubechies(N):
    # make polynomial
    q_y = [sm.binomial(N-1+k,k) for k in reversed(range(N))]

    # get polynomial roots y[k]
    y = sm.mp.polyroots(q_y, maxsteps=200, extraprec=64)

    z = []
    for yk in y:
        # subustitute y = -1/4z + 1/2 - 1/4/z to factor f(y) = y - y[k]
        f = [sm.mpf('-1/4'), sm.mpf('1/2') - yk, sm.mpf('-1/4')]

        # get polynomial roots z[k]
        z += sm.mp.polyroots(f)

    # make polynomial using the roots within unit circle
    h0z = sm.sqrt('2')
    for zk in z:
        if sm.fabs(zk) < 1:
            h0z *= sympy.sympify('(z-zk)/(1-zk)').subs('zk',zk)

    # adapt vanishing moments
    hz = (sympy.sympify('(1+z)/2')**N*h0z).expand()

    # get scaling coefficients
    return [sympy.re(hz.coeff('z',k)) for k in reversed(range(N*2))]

def main():
    for N in range(2,30):
        # get dbN coeffients
        dbN = daubechies(N)

        # write coeffients
        filename = os.path.join(os.getcwd(), 'coefficients/daub' + str(2*N).zfill(2) +'_coefficients.txt')
        print("Writing file {}".format(filename))
        with open(filename, 'w+') as f:
            f.write('# Daubechies ' + str(2*N) + ' scaling coefficients\n')
            f.write("        else if constexpr (N == " + str(2*N) + ")\n        {\n")
            f.write("            if constexpr (std::is_same<float, Real>::value) {\n                return {")
            for i, h in enumerate(dbN):
                f.write(sm.nstr(h, 9) + 'f, ')
            f.write("};\n            }\n")

            f.write("            else if constexpr (std::is_same<double, Real>::value) {\n                return {")
            for i, h in enumerate(dbN):
                f.write(sm.nstr(h, 17) + ', ')
            f.write("};\n            }\n")

            f.write("            else if constexpr (std::is_same<long double, Real>::value) {\n                return {")
            for i, h in enumerate(dbN):
                # log2(64) + some leeway
                f.write(sm.nstr(h, 22) + 'L, ')
            f.write("};\n            }\n")

            f.write("            #ifdef BOOST_HAS_FLOAT128\n")
            f.write("            else if constexpr (std::is_same<boost::multiprecision::float128, Real>::value) {\n                return {")
            for i, h in enumerate(dbN):
                # log10(2**123) + some leeway
                f.write(sm.nstr(h, 37) + 'Q,\n                        ')
            f.write("};\n            }\n")
            f.write("            #endif\n")
            f.write('            else { throw std::logic_error("Wavelet transform coefficients for this precision have not been implemented."); }\n')
            f.write("        }\n")

        # get an approximation of scaling function
        '''x, phi, psi = scipy.signal.cascade(dbN)

        # plot scaling function
        plt.plot(x, phi, 'k')
        plt.grid()
        plt.title('db' + str(2*N) + ' scaling function')
        plt.savefig('scaling_png/daub' + str(2*N).zfill(2) + '_scaling' + '.png')
        plt.clf()

        # plot wavelet
        plt.plot(x, psi, 'k')
        plt.grid()
        plt.title( 'db' + str(2*N) + " wavelet" )
        plt.savefig('wavelet_png/daub' + str(2*N).zfill(2) + '_wavelet' + '.png')
        plt.clf()'''

if __name__ == '__main__':
    main()
