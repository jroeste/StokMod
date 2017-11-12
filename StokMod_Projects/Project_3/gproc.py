import numpy as np
import matplotlib.pyplot as plt

def h(ta, tb):
	#t = np.linspace(tmin, tmax, n);
	na = len(ta);
	nb = len(tb);
	
	onesa = np.ones(na);
	onesb = np.ones(nb);
	return abs(np.outer(ta, onesb) - np.outer(onesa, tb));

def samplemodel(t, mu, sigma, phi):
	n = len(t);
	H = h(t, t);
	S = sigma**2 * np.exp(-phi*H);
	L = np.linalg.cholesky(S);

	z = np.random.normal(0, 1, n);
	return mu + np.matmul(L, z);

def samplecondmodel(ta, xb, tb, mua, sigma, phi):
	na = len(ta);
	mu, S = condexpvar(ta, xb, tb, sigma, phi);	
	L = np.linalg.cholesky(S);

	z = np.random.normal(0, 1, na);
	return mu + np.matmul(L, z);

def expocor(H, sigma, phi):
	return sigma**2 * np.exp(-phi*H);
	
def condexpvar(ta, xb, tb, sigma, phi):
	Ha = h(ta, ta);
	Hb = h(tb, tb);
	Hab = h(ta, tb);	
	Sa = expocor(Ha, sigma, phi);
	Sb = expocor(Hb, sigma, phi);
	Sab = expocor(Hab, sigma, phi);
	tmp = np.matmul(Sab, np.linalg.inv(Sb));
	mu = np.matmul(tmp, xb);
	S = Sa - np.matmul(tmp, np.transpose(Sab));
	return mu, S;

def testf1():
	mu = 0;
	sigma = 1;

	tmin = 1;
	tmax = 100;
	n = tmax - tmin + 1;
	t = np.linspace(tmin, tmax, tmax - tmin + 1);

	N = 10;
	z = np.zeros((N, n));
	for i in range(N):
		phi = 3/10;
		z[i,:] = samplemodel(t, mu, sigma, phi);
		plt.plot(t, z[i,:]);
	plt.show();
	for i in range(N):
		phi = 3/30;
		z[i,:] = samplemodel(t, mu, sigma, phi);
		plt.plot(t, z[i,:]);
	plt.show();
testf1();

def testf2():
	xb = np.array([0.58, -1.34, 0.61]);
	tb = np.array([11.2, 51.8, 81.4]);

	mua = 0;
	sigma = 1;
	phi = 3/10;

	tmin = 1;
	tmax = 100;
	ta = np.linspace(tmin, tmax, tmax - tmin + 1);
	n = len(ta);

	N = 10;
	z = np.zeros((N, n));
	for i in range(N):
		phi = 3/10;
		z[i,:] = samplecondmodel(ta, xb, tb, mua, sigma, phi);
		plt.plot(ta, z[i,:]);
	plt.show();
	for i in range(N):
		phi = 3/30;
		z[i,:] = samplecondmodel(ta, xb, tb, mua, sigma, phi);
		plt.plot(ta, z[i,:]);
	plt.show();
testf2();
