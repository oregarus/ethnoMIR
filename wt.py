'''
Wavelet spectrogram for audio signal

@author: Ata
'''

import pywt
import wave
import struct
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def mirCWt(trackTitle, startSec, lengthSec, sRatio, wtMom, bDebug, bSaveAudio, saveWtSpectra, bCompressWtSpectra):
    
    # read audio file
    audioSig, sampleRate = sf.read('audiosrc/' + trackTitle)
    
    # local definitions
    minFreq = 117 # Hz
    maxFreq = 500 # Hz
    scaleFactor = 2*sRatio
    dt = sRatio/sampleRate # seconds
    audioSigLength = int(lengthSec * sampleRate)
    startPoint = int(startSec * sampleRate)
    scales = np.arange(sampleRate/(maxFreq*scaleFactor), sampleRate/(minFreq*scaleFactor), int(8/scaleFactor))
    
    #save audio file
    if bSaveAudio:
        saveAudio(trackTitle, startPoint, audioSigLength, sampleRate, audioSig)
    
    # cut and/or simplify audioSignal
    audioSig = audioSig[startPoint:startPoint+audioSigLength:sRatio, 1]
    
    # debug
    if bDebug:
        print('sample rate = %f\n' %(sampleRate/sRatio)) # 48000
        print('audio sample length = %f\n' %len(audioSig))
        frequencies = pywt.scale2frequency(wtMom, scales) / dt # frequency
        print('frequencies:\n')
        print(frequencies)

    # wavelet spectra calculation
    [cfs, wtFreqs] = pywt.cwt(audioSig, scales, wtMom, dt)
    wtSpectra = (abs(cfs)) ** 2
    
    # save detailed wavelet spectrogram data to file
    if (saveWtSpectra & 1) == 1:
        strAdd = ''
        saveWtSpectraFile(trackTitle[:-4], dt, strAdd, wtFreqs, wtSpectra)
    
    # compress spectrogram for CNN
    if bCompressWtSpectra:          
        fig, axs = plt.subplots(2)
        fig.suptitle('1d size reduction')
        axs[0].contourf(np.arange(0, len(wtSpectra[1,:]))*dt, wtFreqs, wtSpectra)
        wtSpectra = wtSpectra.reshape(-1, 120).mean(axis=1)
        wtSpectra = np.reshape(a=wtSpectra, newshape=(40,400)) # matrix with 40 rows and 400 columns
        dt = dt*120
        axs[1].contourf(np.arange(0, 400)*dt, wtFreqs, wtSpectra)
        plt.show()

    # save compressed wavelet spectrogram data to file
    if (saveWtSpectra & 2) == 2:
        strAdd = 'Comp'
        saveWtSpectraFile(trackTitle[:-4], dt, strAdd, wtFreqs, wtSpectra)
    
def cWtFromFile(trackTtile, bDisplay):
    
    # read data from file
    wtFreqs = np.loadtxt("wtresult/freqs_" + trackTtile) # read freqs
    dt = wtFreqs[len(wtFreqs)-2] # extract dt
    sigLen = wtFreqs[len(wtFreqs)-1] # extract signal length info
    wtFreqs = wtFreqs[0:len(wtFreqs)-2] # delete dt and signal length
    wtSpectra = np.loadtxt("wtresult/wtPower_" + trackTtile) # read spectra data
    
    # display wavelet spectra
    if bDisplay:
        #'''
        plt.figure
        plt.contourf(np.arange(0, sigLen)*dt, wtFreqs, wtSpectra)
        plt.title('Complex Morlet spectrogram - '+trackTtile)
        plt.yscale('log')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.show()
        #'''
        '''
        # zoom to specific area
        plt.figure(2)
        plt.contourf(np.arange(0, int(2400/2))*dt, wtFreqs, wtSpectra[:,int(13440/2):int(15840/2)])
        plt.title('cmor spectrogram')
        plt.yscale('log')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.show()
        '''
    
def saveAudio(trackTitle, startPoint, audioSigLength, sampleRate, audioSig=[]):
    # save audio file
    wavFile = wave.open('wtresult/audio_'+trackTitle+str(startSec)+'to'+str(startSec+lengthSec)+'sec.wav', 'w') # open wav file
    wavFile.setparams((1, 2, sampleRate, audioSigLength, "NONE", "not compressed"))
    for sample in audioSig[startPoint:startPoint+audioSigLength,1]:
        wavFile.writeframes(struct.pack('h', int( sample * 32767.0 )))
    wavFile.close()

def saveWtSpectraFile(trackTitle, dt, strAdd, wtFreqs=[], wtSpectra=[]):
    fileName = trackTitle+str(startSec)+'to'+str(startSec+lengthSec)+'secSRatio'+str(sRatio)+strAdd
    wtFreqs = np.append(wtFreqs, [dt, len(wtSpectra[1,:])]) # add dt to wtFreqs array
    np.savetxt("wtresult/freqs_"+fileName+".txt", wtFreqs, newline=" ")
    np.savetxt("wtresult/wtPower_"+fileName+".txt", wtSpectra)

'''
Program sequence:
'''

# predefinitions
trackTitle = 'csardas_Icsan1969.wav'  
startSec = 1 # start of input data
lengthSec = 4 # length of input data
sRatio = 4 # sampleRate = sampleRate/sRatio
wtMom = 'cmor' # mother wavelet used


mirCWt(trackTitle, startSec, lengthSec, sRatio, wtMom, bDebug=False, bSaveAudio=True, saveWtSpectra=3, bCompressWtSpectra=True)

wtFileTitle = trackTitle[:-4]+str(startSec)+'to'+str(startSec+lengthSec)+'secSRatio'+str(sRatio)+'Comp.txt'
cWtFromFile(wtFileTitle, bDisplay=True)