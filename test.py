# -*- coding: utf-8 -*-
import scipy.signal, numpy, numpy.linalg as nl, sympy, sympy.mpmath as sm, matplotlib.pyplot as plt

# precition
sm.mp.prec = 256

def daubechis(N):
	q_z = []
	for k in range(N-1,-1,-1):
		q_z.append( sm.binomial(N-1+k,k) )

	q_sol = sm.mp.polyroots(q_z)

	s_arr = []
	for q in q_sol:
		sol = sm.mp.polyroots([sm.mpf('-1/4'), sm.mpf('1/2') - q, sm.mpf('-1/4')]);
		if( sympy.Abs(sol[0]) < sympy.Abs(sol[1]) ):
			s_arr.append(sol[0])
		else:
			s_arr.append(sol[1])

	z = sympy.symbols('z')
	h0z = 1
	for s in s_arr:
		h0z *= (z-s)/(sm.mpf('1')-s)

	hz = sympy.expand((sm.mpf('1/2')*z + sm.mpf('1/2'))**N*h0z)

	scaling_coeff = []
	for k in range(N*2-1, -1, -1):
		scaling_coeff.append(sympy.re(hz.coeff(z,k)))
		
	#sm.nprint(sum(scaling_coeff)-1)

	sqrt2 = sympy.sqrt(sm.mpf('2'))
	return map(lambda s:s*sqrt2, scaling_coeff)
	

if __name__ == '__main__':
	for N in range(2,51):
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
