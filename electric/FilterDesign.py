from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import sys

file_object = open('filterDesign.txt', "w") 

w = 8  # wordlength of quantized coefficients
Astart = 1
Gstart = 17
Cstart = 33
Fstart = 49
Istart = 65
Dstart = 81
Bstart = 97
Estart = 113

def getBinaryNumberFromCapacitance(cap):
	cap = int(cap)
	return bin(cap)[2:].zfill(8)
	
def writeKeysForKoef(start, variable):
	keys = getBinaryNumberFromCapacitance(variable)
	for i in range(len(keys)):
		#print(Ikeys[i])
		if(keys[i] == '1'):
			file_object.write(str(start + i))
			file_object.write('\n')
			file_object.write(str(start + i + 8))
			file_object.write('\n')
		
def getKeysFromKoefArray(fullArrayRounded):
#not sure with equal coefficients
	for kArray in fullArrayRounded:
		# array 
		I = kArray[0]
		Ikeys = getBinaryNumberFromCapacitance(I)
		writeKeysForKoef(Istart, I)
		E = kArray[1]
		writeKeysForKoef(Estart, E)
		C = kArray[2]
		writeKeysForKoef(Cstart, C)
		J = kArray[3]
		writeKeysForKoef(Istart, J)
		#writeKeysForKoef(Jstart, J)
		G = kArray[4]
		writeKeysForKoef(Gstart, G)
		H = kArray[5]
		writeKeysForKoef(Gstart, H)
		#writeKeysForKoef(Hstart, H)
		file_object.write('_')
		file_object.write('\n')
	

def getMinExceptNull(array):
	min = 10000000
	for elem in array:
		if(elem != 0):
			if(elem<min):
				min = elem
	return min

def white_noize(sos):
	mean = 0
	std = 1 
	num_samples = 1000
	samples = np.random.normal(mean, std, size=num_samples)
	b, a = signal.sos2tf(sos)
	y = signal.lfilter(b, a, samples)
	t = np.linspace(-1, 1, 1000)
	plt.figure
	plt.plot(t, y, 'b', alpha=0.75)
	#plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')
	plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice','filtfilt'), loc='best')
	plt.grid(True)
	plt.show()

def uniform_midtread_quantizer(x, w, xmin=1):
	# quantization step
	Q = float(xmin)/(2**(w-1))
	# limiter
	x = np.copy(x)
	idx = np.where(x <= -xmin)

	x[idx] = -1
	idx = np.where(x > xmin - Q)
	x[idx] = 1 - Q
	for array in idx:
		print array
	# linear uniform quantization
	#xQ = Q * np.floor(x/Q + 1/2)
	xQ = np.floor(x/Q + 1/2)
	return xQ

def getSosFromCoefficients(fullArray):
	sos = []
	for kArray in fullArray:
		deviation = []
		sos_line = []
		koef=1
		#print kArray
		#print np.amax(kArray)
		
		minimum = getMinExceptNull(kArray)
		print minimum
		koef = (koef/minimum)
		while(np.amax(kArray/minimum)<64):
			#print kArray
			for index in range(len(kArray)):
				kArray[index] = kArray[index]*2
			
			koef = koef*2		
		
		I = round(kArray[0]/minimum)
		if(I == 0):
			deviation.append(0)
		else:
			deviation.append(kArray[0]/I/minimum)
		
		E = round(kArray[1]/minimum)
		if(E == 0):
			deviation.append(0)
		else:
			deviation.append(kArray[1]/E/minimum)
			
		C = round(kArray[2]/minimum)
		if(C == 0):
			deviation.append(0)
		else:
			deviation.append(kArray[2]/C/minimum)
		J = round(kArray[3]/minimum)
		if(J == 0):
			deviation.append(0)
		else:
			deviation.append(kArray[3]/J/minimum)
		G = round(kArray[4]/minimum)
		if(G == 0):
			deviation.append(0)
		else:
			deviation.append(kArray[4]/G/minimum)
		H = round(kArray[5]/minimum)
		if(H == 0):
			deviation.append(0)
		else:
			deviation.append(kArray[5]/H/minimum)
		#I = kArray[0]
		#E = kArray[1]
		#C = kArray[2]
		#J = kArray[3]
		#G = kArray[4]
		#H = kArray[5]
		print "kArray_norm"
		kArray_norm = [I, E, C, J, G, H]
		print kArray_norm
		print "deviation"
		print deviation
		for dev in deviation:
			if(dev == 0):
				continue
			elif(abs(dev-1)>0.015):
				print "bad characteristics of filter, next step"
				raise ValueError("bad filter, go next")
		
		sos_line.append(I/koef)
		sos_line.append((-I-J+G)/koef)
		sos_line.append((J-H)/koef)
		sos_line.append(1)
		sos_line.append((C+E)/koef-2)
		sos_line.append(1-(E)/koef)
		sos.append(sos_line)
	return sos
	
def getSosFromCoefficientsForElliptic(fullArray):
	sos = []
	for kArray in fullArray:
		deviation = []
		sos_line = []
		koef=1
		#print kArray
		#print np.amax(kArray)
		minimum = getMinExceptNull(kArray)
		koef = (koef/minimum)
		while(np.amax(kArray/minimum)<128):
			#print kArray
			for index in range(len(kArray)):
				kArray[index] = kArray[index]*2
			
			koef = koef*2		
		
		K = round(kArray[0]/minimum)
		deviation.append(kArray[0]/K/minimum)
		G = round(kArray[1]/minimum)
		deviation.append(kArray[1]/G/minimum)
		E = round(kArray[2]/minimum)
		deviation.append(kArray[2]/E/minimum)
		C = round(kArray[3]/minimum)
		deviation.append(kArray[3]/C/minimum)
		#I = kArray[0]
		#E = kArray[1]
		#C = kArray[2]
		#J = kArray[3]
		#G = kArray[4]
		#H = kArray[5]
		print "kArray_norm"
		kArray_norm = [K, E, G, C]
		print kArray_norm
		print "deviation"
		print deviation
		for dev in deviation:
			if(abs(dev-1)>0.005):
				print "bad characteristics of filter, next step"
				raise ValueError("bad filter, go next")
		
		sos_line.append(K/koef)
		sos_line.append((2*K-G)/koef)
		sos_line.append((K)/koef)
		sos_line.append(1)
		sos_line.append((C+E)/koef-2)
		sos_line.append(1-(E)/koef)
		sos.append(sos_line)
	return sos

def getCoefficients(sos, quantized = False):
	fullArray = []
	if(quantized==True):
		sos = uniform_midtread_quantizer(sos, 8)
		for index in range(len(sos)):
			sos[index] = sos[index]/sos[index][3]
		print "sos"
		print sos
		
	for filter in sos:
		kArray = []
		b = [filter[0], filter[1], filter[2]]
		a = [filter[3], filter[4], filter[5]]
		print b
		#print a
		I = b[0]
		E = 1 - a[2]
		C = 2 + a[1] - E
		#J = b[2]*2
		J=I;
		eq = False;
		bias = 0.05
		while(eq != True):
			G = b[1] + I + J
			H = J - b[2]
			if(abs(H)<0.00001):
				H = 0;
			if(H<0 or G<0):
				#J=J+bias
				print "can't create filter"
				return
			else:
				eq = True
			if(J>10):
				print "can't create filter"
				return
		#in = np.array([[3,1], [1,2]])
		#out = np.array([9,8])
		#x = np.linalg.solve(a, b)
		array = [I, E, C, J, G, H]
		#minimum = np.amin(array)
		minimum = 1
		kArray.append(I/minimum)
		kArray.append(E/minimum)
		kArray.append(C/minimum)
		kArray.append(J/minimum)
		kArray.append(G/minimum)
		kArray.append(H/minimum)
		#print kArray
		fullArray.append(kArray)
	return fullArray

def getCoefficientsForElliptic(sos, quantized = False):
	if(quantized==True):
		sos = uniform_midtread_quantizer(sos, 8)
		print "sos"
		print sos
	fullArray = []
	for filter in sos:
		kArray = []
		b = [filter[0], filter[1], filter[2]]
		a = [filter[3], filter[4], filter[5]]
		#print b
		#print a
		I = 0
		H = 0
		L = 0
		J = 0
		K = b[0]
		G = 2*K-b[1]
		E = 1 - a[2]
		C = 2 - E + a[1]
		#in = np.array([[3,1], [1,2]])
		#out = np.array([9,8])
		#x = np.linalg.solve(a, b)
		array = [K, G, E, C]
		#minimum = np.amin(array)
		minimum = 1
		kArray.append(K/minimum)
		kArray.append(G/minimum)
		kArray.append(E/minimum)
		kArray.append(C/minimum)
		#print kArray
		fullArray.append(kArray)
	return fullArray

def plot_sosfreqz(w,h):
	plt.subplot(2, 1, 1)
	db = 20*np.log10(np.abs(h))
	plt.plot(w/np.pi, db)
	plt.ylim(-75, 5)
	plt.grid(True)
	plt.yticks([0, -20, -40, -60])
	plt.ylabel('Gain [dB]')
	plt.title('Frequency Response')
	plt.subplot(2, 1, 2)
	plt.plot(w/np.pi, np.angle(h))
	plt.grid(True)
	plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
	plt.ylabel('Phase [rad]')
	plt.xlabel('Normalized frequency (1.0 = Nyquist)')
	plt.savefig('filterDesignResult.png')
	#plt.show()
	
	
def plot_2nd_order_from_sos(sos):
	for filter in sos:
		b = [filter[0],filter[1],filter[2]]
		a = [filter[3],filter[4],filter[5]]
		w, h = signal.freqz(b, a, 1000000)
		#plot_sosfreqz(w,h)
		
def getRoundCoefficients(fullArray):
	fullArrayRounded = []
	for kArray in fullArray:
		koef=1
		#print kArray
		#print np.amax(kArray)
		minimum = getMinExceptNull(kArray)
		koef = (koef/minimum)
		while(np.amax(kArray/minimum)<64):
			#print kArray
			for index in range(len(kArray)):
				kArray[index] = kArray[index]*2
			
			koef = koef*2		
		
		I = round(kArray[0]/minimum)
		E = round(kArray[1]/minimum)
		C = round(kArray[2]/minimum)
		J = round(kArray[3]/minimum)
		G = round(kArray[4]/minimum)
		H = round(kArray[5]/minimum)
		print "kArray_norm"
		kArray_norm = [I, E, C, J, G, H]
		print kArray_norm
		fullArrayRounded.append(kArray_norm)
	return fullArrayRounded
		
def make_digital_filter(N, rp, rs, Wn, btype='low'):
	finish = False
	WnAdd = Wn
	rpAdd = rp
	rsAdd = rs
	fullArray = [];
	try:
		while(finish == False):
			if(WnAdd<0.5):
				WnAdd = WnAdd+0.001
			elif(abs(rpAdd-rp)/rp<0.05):
				rpAdd = rpAdd+0.01
			elif(abs(rsAdd-rs)/rs<0.05):
				rsAdd = rsAdd+0.01
			else:
				raise Exception("There is no filters with these characteristics.")
			b, a = signal.ellip(N, rpAdd, rsAdd, WnAdd, btype, analog=False)
			sos = signal.tf2sos(b,a)
			#print sos
			w, h = signal.sosfreqz(sos, worN=1000000)
			#plot_2nd_order_from_sos(sos)
			makeCallibration(sos)
			#plot_2nd_order_from_sos(sos)
			#print "sos"
			print sos
			w, h = signal.sosfreqz(sos, worN=1000000)
			#plot_sosfreqz(w,h)
			
			#fullArray = getCoefficientsForElliptic(sos, True)
			fullArray = getCoefficients(sos)
			
			#print fullArray
			try:
				new_sos = getSosFromCoefficients(fullArray)
				#new_sos = getSosFromCoefficientsForElliptic(fullArray)
				finish = True
			except ValueError as e:
				pass
	except Exception as e:
		print e
		print "There is no filters with these characteristics."
		return
				
	w, h = signal.sosfreqz(sos, worN=1000000)
	#plot_sosfreqz(w,h)
	print "new_sos"
	print new_sos
	print "general_coefficients"
	fullArrayRounded = getRoundCoefficients(fullArray)
	getKeysFromKoefArray(fullArrayRounded)
	w, h = signal.sosfreqz(new_sos, worN=1000000)
	plot_sosfreqz(w,h)
	#white_noize(new_sos)
	
	
def makeCallibration(sos):
	coef = 1
	for filter in sos:
		#print "filter " + str(filter)
		b = [filter[0],filter[1],filter[2]]
		a = [filter[3],filter[4],filter[5]]
		sys = signal.TransferFunction(b, a, dt=0.000001)
		w, mag, phase = signal.dbode(sys)
		maximum = np.amax(mag)
		#print "max " + str(maximum)
		if(maximum>5):
			divide = abs(maximum)/5
			coef = coef*divide
			filter[0] = filter[0]/divide
			filter[1] = filter[1]/divide
			filter[2] = filter[2]/divide
			#print "/" + str(divide)
		else:
			if(abs(maximum)>5):
				divide = abs(maximum)/5
				coef=coef*divide
				filter[0] = filter[0]/divide
				filter[1] = filter[1]/divide
				filter[2] = filter[2]/divide
				#print "*" + str(divide)
				
	#print "coef" + str(coef)
	b = [sos[0][0],sos[0][1],sos[0][2]]
	a = [sos[0][3],sos[0][4],sos[0][5]]
	#print b
	#print a
	sys = signal.TransferFunction(b, a, dt=0.000001)
	w, mag, phase = signal.dbode(sys)
	#print np.amax(mag)
	#print "*" + str(coef)
	sos[0][0] = sos[0][0]*coef
	sos[0][1] = sos[0][1]*coef
	sos[0][2] = sos[0][2]*coef
	
	
if __name__ == "__main__":
	# order, max_db_dist, db-, f, 
	#make_digital_filter(6,0.1,60,0.05, "low")
	order = float(sys.argv[1])
	rp = float(sys.argv[2])
	rs = float(sys.argv[3])
	Wn = float(sys.argv[4])
	type = sys.argv[5]
	make_digital_filter(order, rp, rs, Wn, type)