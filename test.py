# -*- coding: utf-8 -*-
import sympy
import sympy.mpmath as sm
import scipy.signal
import matplotlib.pyplot as plt

# precition
sm.mp.prec = 256

def daubechis(N):
	q_z = [sm.binomial(N-1+k,k) for k in reversed(range(N))]

	q_sol,err = sm.mp.polyroots(q_z, maxsteps=100, error=True)
	#print N, sm.nstr(err,5)

	if err > 1.0e-60:
		raise Exception

	s_arr = []
	for q in q_sol:
		sol = sm.mp.polyroots([sm.mpf('-1/4'), sm.mpf('1/2') - q, sm.mpf('-1/4')]);
		if( sympy.Abs(sol[0]) < sympy.Abs(sol[1]) ):
			s_arr.append(sol[0])
		else:
			s_arr.append(sol[1])

	h0z = 1
	for s in s_arr:
		h0z *= sympy.sympify('(z-s)/(1-s)').subs('s',s)

	hz = (sympy.sympify('(1+z)/2')**N*h0z).expand()

	scaling_coeff = [sympy.re(hz.coeff('z',k)) for k in reversed(range(N*2))]

	#sm.nprint(sum(scaling_coeff)-1)
	sqrt2 = sm.sqrt('2.0')
	return [s*sqrt2 for s in scaling_coeff]

if __name__ == '__main__':
	for N in range(2,100):
		f = open('db' + str(N).zfill(2) +'_coefficients.txt', 'w')
		dbN = daubechis(N)
		f.write('# db' + str(N) + ' scaling coefficients\n')
		for i, c in enumerate(dbN):
			f.write('h['+ str(i) + ']='+ sm.nstr(c,40) + '\n')
		f.close()

		x, phi, psi = scipy.signal.cascade(dbN)

		plt.plot(x, phi, 'k')
		plt.grid()
		plt.title('db' + str(N) + ' scaling function')
		#plt.savefig('db' + str(N).zfill(2) + '_scaling' + '.svg')
		plt.savefig('db' + str(N).zfill(2) + '_scaling' + '.png')
		plt.clf()

		plt.plot(x, psi, 'k')
		plt.grid()
		plt.title( 'db' + str(N) + " wavelet" )
		#plt.savefig('db' + str(N).zfill(2) + '_wavelet' + '.svg')
		plt.savefig('db' + str(N).zfill(2) + '_wavelet' + '.png')
		plt.clf()
