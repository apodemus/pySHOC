#from numpy import *
import uncertainties as u
from uncertainties.unumpy import *
import numpy as np
import matplotlib.pyplot as plt
from SHOC_readnoise import ReadNoiseTable

print '\n'*3, '*'*88
print 'This script does a SNR comparison between SHOC EM and CON modes\n\n'
print '*'*88, '\n'*3

fn_rnt = '/home/hannes/iraf-work/SHOC_ReadoutNoiseTable_new'

################################################################################################################################################################################


################################################################################################################################################################################

def user_input_str(query, default=None, **kw):								#NEED RETURN TYPE ARGUMENT!
  validity_test =  kw['val_test'] 	if kw.has_key('val_test') 	else lambda x: True
  verify = kw['verify']		 	if kw.has_key('verify') 	else False
  what = kw['what'] 			if kw.has_key('what') 		else 'output string'
  example = kw['example'] 		if kw.has_key('example') 	else False
  
  repeat = True
  while repeat:
    if default in ['', None]:
      default_str = ''
    else:
      default_str = ' or CR for the default %s.\n' %repr(default)
    
    instr = raw_input(query + default_str)
    
    if not instr:
      if default:
	instr = default
	repeat = False
      else:
	pass	#returns empty instr = ''
    else:
      if verify:
	while not(instr == ''):
	  print "You have entered '%s' as the %s" %(instr, what)
	  #print "Example output filename:\t %s_0001.fits" %instr
	  new_instr = raw_input('CR to accept or re-input.\n')
	  if new_instr == '':
	    break
	  else:
	    instr = new_instr
      if validity_test(instr):
	repeat = False
      else:
	print 'Invalid input!! Please try again.\n'
	repeat = True
	  
  return instr
  
rnt = ReadNoiseTable(fn_rnt)

#print rnt.__dict__
  
t = np.linspace(0,0.5,100)
r_ap = 9.			#pixels
bin = int(user_input_str('binning? ', 1 ))
s = 0.076/6*4			#arcsec per pixel				#TELESCOPE??
#print bin, np.pi, r_ap, s
#print type(bin), type(np.pi), type(r_ap), type(s)
A = np.pi*r_ap*r_ap*bin*s
B = 21.7			#dark sky surface brightness mag/arcsec^2	#ASK!
Msky = B - 2.5*log10(A);	#Sky mag
Ms = float( user_input_str('Object magnitude = ', 15)	)		#Obj mag

#SHOC Coefficients	(See Rocco's thesis
sf = -0.4;		sigma_sf = 0.001;	s = u.ufloat((sf,sigma_sf)) 
cf = 9.75;		sigma_cf = 0.004;	c = u.ufloat((cf,sigma_cf)) 
sigma_Fsky = 0.2					#Assumed flux error (%) in Sky
Msky_up =  u.ufloat((Msky,-2.5 + 2.5*log10(10+10*sigma_Fsky)));		Msky_low = u.ufloat((Msky,2.5 - 2.5*log10(10-10*sigma_Fsky)))
sigma_Fs = 0.2						#Assumed flux error (%) in Source
Ms_up = u.ufloat((Ms,-2.5 + 2.5*log10(10+10*sigma_Fs)));		Ms_low = u.ufloat((Ms, 2.5 - 2.5*log10(10-10*sigma_Fs)))

#print Ms_low, Ms_up
#print Msky_low, Msky_up


npix = np.pi*r_ap*r_ap		#number of pixels inside aperture


ron, sens = rnt.get_readnoise()
#print ron
#print sens
#raw_input()

#Conventional mode
NR_CON = 7.49		#Read Noise
G_CON = 2.4		#Gain

#EM mode
NR_EM = 29.99		#Read Noise
G_EM = 50		#Gain

def SNR_CON(Ms,Msky):
  fn = t*10.**(s*Ms+c)
  fd = t*(10.**c)*(10.**(s*Msky)+10.**(s*Ms)) + npix*NR_CON**2.
  SNR_CON = fn/sqrt(fd)			#f
  return SNR_CON

def SNR_EM(Ms,Msky):
  fn = t*10.**(s*Ms+c)
  gd = 2.*t*(10.**c)*10.**(s*Msky) + npix*(NR_EM/G_EM)**2.
  SNR_EM = fn/sqrt(gd)				#g
  return SNR_EM
  
SNR_CON_nv = nominal_values(SNR_CON(Ms,Msky))
SNR_CON_up = SNR_CON(Ms_up,Msky_up)
SNR_CON_low = SNR_CON(Ms_low,Msky_low)

SNR_EM_nv = nominal_values(SNR_EM(Ms,Msky))
SNR_EM_up = SNR_EM(Ms_up,Msky_up)
SNR_EM_low = SNR_EM(Ms_low,Msky_low)



fig = plt.figure(1)
ax1 = fig.add_subplot(111)

ax1.plot(t,SNR_CON_nv,'b-',label='CON')
ax1.fill_between(t, SNR_CON_nv+std_devs(SNR_CON_up), SNR_CON_nv-std_devs(SNR_CON_low), facecolor='b',alpha=0.25)
#ax1.fill_between(t,1.2*SNR_CON_nv,0.8*SNR_CON_nv,facecolor='grey',alpha=0.25)

ax1.plot(t,SNR_EM_nv,'r-',label='EM')
ax1.fill_between(t, SNR_EM_nv+std_devs(SNR_EM_up), SNR_EM_nv-std_devs(SNR_EM_low), facecolor='r',alpha=0.25)
#ax1.fill_between(t,1.2*SNR_EM_nv,0.8*SNR_EM_nv,facecolor='grey',alpha=0.25)

plt.title('3 MHz 14bit, 4.9 preamp gain')
ax1.set_xlabel('t')
ax1.set_ylabel('SNR')
ax1.legend()

plt.show()