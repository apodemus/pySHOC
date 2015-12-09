#from numpy import *
import uncertainties as u
from uncertainties.unumpy import nominal_values, std_devs
import numpy as np
import matplotlib.pyplot as plt

#from SHOC_readnoise import ReadNoiseTable

import argparse
from sys import argv


print('\n'*3, '*'*88)
print('This script does a SNR comparison between SHOC EM and CON modes\n\n')
print('*'*88, '\n'*3)

main_parser = argparse.ArgumentParser( description='SNR comparison between SHOC EM and CON modes.' )
main_parser.add_argument( '-b', '--binning', default=4, type=int )
main_parser.add_argument( '-m', '--magnitude', type=float, default=15, help='Object magnitude = ?' )
main_parser.add_argument( '-s', '--sky-mag', type=float, default=21.7, help='Sky surface brightness mag/arcsec^2. default=21.7 for dark sky.' )
main_parser.add_argument( '-r',  type=float, default=10, help='Aperture radius in pixels.' )
main_parser.add_argument( '-t',  '--exposure', type=float, default=10, help='Exposure time. (upper limit)' )


main_parser.add_argument( '-tel',  type=str, default='1.9', help='Which telescope' )
#main_parser.add_argument( '-shoc',  type=int, default=1, help='Which SHOC camera?' )
##main_parser.add_argument( '-mode',  type=str, default='CON', help='SHOC Mode: EM / CON.' )
#main_parser.add_argument( '-pre-amp', type=float, default=2.4, help='Pre-amp setting.' )
#main_parser.add_argument( '-freq',  type=int, default=1, help='Readout frequency in MHz.' )
args = main_parser.parse_args( argv[1:] )

#RNT = ReadNoiseTable()
pixscale = {'1.9'     :       0.076,
            '1.9+'    :       0.163,            #with focal reducer
            '1.0'     :       0.167,
            '0.75'    :       0.218   }


t       = np.linspace(0, args.exposure, 100)       #range of exposure times to plot for
r_ap    = args.r                        #Aperture radius in pixels
Npix    = np.pi * r_ap*r_ap             #number of pixels inside aperture
binning = args.binning
scale   = pixscale[args.tel]            #arcsec per pixel

A = Npix * binning * scale              #Aperture are on sky
Msky = args.sky_mag - 2.5*log10(A);     #Sky mag
Ms = args.magnitude                     #Obj mag

#SNR Coefficients, uncertainty      (See Rocco's thesis)
SF =    {       '1.9'     :       (-0.4, 0.001)
                '1.9+'    :       (-0.4, 0.001),            #with focal reducer
                '1.0'     :       (-0.4, 0.001),
                '0.75'    :       (-0.4, 0.002)   }
CF =    {       '1.9'     :       (9.750, 0.004)
                '1.9+'    :       (9.483, 0.006),            #with focal reducer
                '1.0'     :       (9.277, 0.005),
                '0.75'    :       (9.140, 0.186)   }

sf = u.ufloat( SF[args.tel] )
cf = u.ufloat( CF[args.tel] )


#TODO:  Proper error propagation
sigma_Fsky = 0.2                                        #Assumed flux error (%) in Sky
Msky_up =  u.ufloat( (Msky,-2.5 + 2.5*log10(10+10*sigma_Fsky)) )
Msky_low = u.ufloat( (Msky,2.5 - 2.5*log10(10-10*sigma_Fsky)) )
sigma_Fs = 0.2                                          #Assumed flux error (%) in Source
Ms_up = u.ufloat( (Ms,-2.5 + 2.5*log10(10+10*sigma_Fs)) )
Ms_low = u.ufloat( (Ms, 2.5 - 2.5*log10(10-10*sigma_Fs)) )


#ron, sens, saturation = RNT.get_readnoise_from_kw( mode = args.mode,
                                                    #shoc = args.shoc,
                                                    #preamp = args.pre_amp,
                                                    #freq = args.freq )
      


#Conventional mode
NR_CON = 7.49           #Read Noise
G_CON = 2.4             #Gain

#EM mode
NR_EM = 29.99           #Read Noise
G_EM = 50               #Gain

def SNR_CON(Ms,Msky):
    fn = t * np.power(10, Ms*sf + cf)
    fd = t*(10.**cf)*(10.**(sf*Msky)+10.**(sf*Ms)) + Npix*NR_CON**2.
    SNR_CON = fn/sqrt(fd)                 #f
    return SNR_CON

def SNR_EM(Ms,Msky):
    fn = t* 10.**(sf*Ms+cf)
    gd = 2.*t*(10.**cf)*10.**(sf*Msky) + Npix*(NR_EM/G_EM)**2.
    SNR_EM = fn/sqrt(gd)                          #g
    return SNR_EM

SNR_CON_nv = nominal_values(SNR_CON(Ms,Msky))
SNR_CON_up = SNR_CON(Ms_up,Msky_up)
SNR_CON_low = SNR_CON(Ms_low,Msky_low)

SNR_EM_nv = nominal_values(SNR_EM(Ms,Msky))
SNR_EM_up = SNR_EM(Ms_up,Msky_up)
SNR_EM_low = SNR_EM(Ms_low,Msky_low)



fig, ax1  = plt.subplots( figsize=(18,8), tight_layout=True)

ax1.plot(t,SNR_CON_nv,'b-',label='CON')
ax1.fill_between(t, SNR_CON_nv+std_devs(SNR_CON_up), SNR_CON_nv-std_devs(SNR_CON_low), facecolor='b',alpha=0.25)
#ax1.fill_between(t,1.2*SNR_CON_nv,0.8*SNR_CON_nv,facecolor='grey',alpha=0.25)

ax1.plot(t,SNR_EM_nv,'r-',label='EM')
ax1.fill_between(t, SNR_EM_nv+std_devs(SNR_EM_up), SNR_EM_nv-std_devs(SNR_EM_low), facecolor='r',alpha=0.25)
#ax1.fill_between(t,1.2*SNR_EM_nv,0.8*SNR_EM_nv,facecolor='grey',alpha=0.25)

plt.title('3 MHz 14bit, Pre-amp  ')
ax1.set_xlabel('t (s)')
ax1.set_ylabel('SNR')
ax1.legend()
ax1.grid()

plt.show()
