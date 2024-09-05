import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
import scipy.signal
from scipy.signal import convolve
# Lista de nombres de archivos
filenames = ['terremoto1.txt', 'terremoto2.txt']
#Se utiliza enumerate para obtener el indice y el valor de cada elem de la lista

'''
Ejercicio a)
Calcule los coeficientes de la serie de Fourier discreta y determine la TFD a partir de los valores obtenidos
anteriormente.
'''

for i, filename in enumerate(filenames):
    data = np.loadtxt(filename,usecols=[0,1],unpack= True)
    ecd_x = data[0].tolist()
    ecd_y = data[1].tolist() 
    # Calcula los coeficientes de Fourier
    fft_y = fft(ecd_y)
    # Calcula las frecuencias para el eje x del gráfico de Fourier
    n = len(ecd_y)
    timestep = ecd_x[1] - ecd_x[0]  # Asume un espaciado uniforme
    freq = np.fft.fftfreq(n, 1/timestep)
    # Grafica los coeficientes de Fourier en una nueva figura
    fig, graf = plt.subplots()
    graf.plot(freq, np.abs(fft_y))
    graf.set_title('Coeficientes de Fourier de ' + filename)
    graf.set_xlabel('Frecuencia (Hz)')
    graf.set_ylabel('Amplitud')
    plt.show()  # Muestra el gráfico en una nueva ventana


'''
Ejercicio b)
Suavice las altas frecuencias de las señales utilizando una convolucion con la ventana que considere
adecuada.
'''

for i, filename in enumerate(filenames):
    data = np.loadtxt(filename,usecols=[0,1],unpack= True)
    ecd_x = data[0].tolist()
    ecd_y = data[1].tolist()
    # Define el tamaño de la ventana
    window_size = 200
    # Crea la ventana de Hamming
    window = scipy.signal.windows.hamming(window_size)
    # Normaliza la ventana
    window /= np.sum(window)
    # Suaviza la señal mediante la convolución con la ventana de Hamming
    smoothed_signal = convolve(ecd_y, window, mode='same')
    # Grafica la señal suavizada
    fig, graf = plt.subplots(2)
    graf[0].plot(ecd_x, smoothed_signal)
    graf[0].set_title('Señal suavizada de ' + filename)
    graf[0].set_xlabel('Tiempo (s)')
    graf[0].set_ylabel('Amplitud')
    graf[1].plot(ecd_x, ecd_y)
    graf[1].set_title('Señal original de ' + filename)
    graf[1].set_xlabel('Tiempo (s)')
    graf[1].set_ylabel('Amplitud')
    plt.show()  # Muestra el gráfico en una nueva ventana

'''
Ejercicio c)
Determine cuales son las frecuencias más afectadas por el terremoto en cada una de las señales, utilizando
el resultado del inciso 1. Nota algun cambio luego del filtrado anterior?
'''
for filename in filenames:
    data = np.loadtxt(filename,usecols=[0,1],unpack= True)
    ecd_x = data[0].tolist()
    ecd_y = data[1].tolist()
    # Calcula los coeficientes de Fourier
    fft_result = fft(ecd_y)
    # Calcula las frecuencias para el eje x del gráfico de Fourier
    n = len(ecd_y)
    timestep = ecd_x[1] - ecd_x[0]  # Asume un espaciado uniforme
    frequencies = np.fft.fftfreq(n, d=timestep)
    # Toma los valores absolutos de los coeficientes de Fourier
    magnitudes = np.abs(fft_result)
    # Encuentra la frecuencia con la amplitud más alta
    max_freq = frequencies[np.argmax(magnitudes)]
    # Imprime la frecuencia con la amplitud más alta
    print('La frecuencia más afectada por el terremoto en  ' + filename + ' es ' + str(max_freq) + ' Hz')
    fig, graf = plt.subplots()
    graf.set_title('Coeficientes de Fourier de ' + filename)
    graf.set_xlabel('Frecuencia (Hz)')
    graf.set_ylabel('Amplitud')
    graf.stem(frequencies, magnitudes)
    # 'max_freq' es la frecuencia que se marcará con una línea roja
    graf.axvline(x=max_freq, color='r', linestyle='--')
    graf.annotate(f'Max Freq: {max_freq}', xy=(max_freq, max(magnitudes)), xytext=(max_freq, max(magnitudes) + 5),arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()

for filename in filenames:
    data = np.loadtxt(filename,usecols=[0,1],unpack= True)
    ecd_x = data[0].tolist()
    ecd_y = data[1].tolist()
    smoothed_signal = convolve(ecd_y, window, mode='same')
    # Calcula los coeficientes de Fourier
    fft_result = fft(smoothed_signal)
    # Calcula las frecuencias para el eje x del gráfico de Fourier
    n = len(ecd_y)
    timestep = ecd_x[1] - ecd_x[0]  # Asume un espaciado uniforme
    frequencies = np.fft.fftfreq(n, d=timestep)
    # Toma los valores absolutos de los coeficientes de Fourier
    magnitudes = np.abs(fft_result)
    # Encuentra la frecuencia con la magnitud más alta
    max_freq = frequencies[np.argmax(magnitudes)]
    # Imprime la frecuencia con la magnitud más alta
    print('La frecuencia más afectada por el terremoto en ' + filename + ' suavizada es ' + str(max_freq) + ' Hz')
    fig, graf = plt.subplots()
    graf.set_title('Coeficientes de Fourier de ' + filename + ' suavizada')
    graf.set_xlabel('Frecuencia (Hz)')
    graf.set_ylabel('Amplitud')
    graf.stem(frequencies, magnitudes)
    # 'max_freq' es la frecuencia que se marcará con una linea roja
    graf.axvline(x=max_freq, color='r', linestyle='--')
    graf.annotate(f'Max Freq: {max_freq}', xy=(max_freq, max(magnitudes)), xytext=(max_freq, max(magnitudes) + 5),arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()

'''
d) Teniendo en cuenta ambas señales ¿Cual es la frecuencia que más se aceleró? Obtenga este resultado
de 2 maneras diferentes.
'''
import numpy as np
from scipy.fft import fft, fftfreq

# Carga los datos
data1 = np.loadtxt('terremoto1.txt', usecols=[1], unpack=True)
data2 = np.loadtxt('terremoto2.txt', usecols=[1], unpack=True)

# Aplica FFT
fft1 = fft(data1)
fft2 = fft(data2)

# Obtiene los valores absolutos para encontrar las amplitudes
magnitudes1 = np.abs(fft1)
magnitudes2 = np.abs(fft2)

# Obtiene las frecuencias acordes a los resultados de la FFT
sampling_rate1 =0.01  # Reemplaza con las frecuencias de muestreo en terremoto 1
sampling_rate2 =0.005 # Reemplaza con las frecuencias de muestreo en terremoto 2
freqs1 = fftfreq(len(data1),sampling_rate1)
freqs2 = fftfreq(len(data2), sampling_rate2)

# Encuentra la frecuencia con la amplitud más alta en cada señal 
max_freq1 = freqs1[np.argmax(magnitudes1)]
max_freq2 = freqs2[np.argmax(magnitudes2)]

# Imprime las frecuencias máximas en Hz
print("La frecuencia más acelerada en terremoto1 es:",max_freq1, "Hz")
print("La frecuencia más acelerada en terremoto2 es:",max_freq2, "Hz")

# Compara las frecuencias máximas
if max_freq1 > max_freq2:
    print("La frecuencia más acelerada se encuentra en terremoto 1")
elif max_freq1 < max_freq2:
    print("La frecuencia más acelerada se encuentra en terremoto 2")
else:
    print("La frecuencia más acelerada es igual en ambas señales")

#---------------------------------------------------------------------------
#forma 2
import numpy as np
from scipy.signal import periodogram

# Carga los datos
data1 = np.loadtxt('terremoto1.txt', usecols=[1], unpack=True)
data2 = np.loadtxt('terremoto2.txt', usecols=[1], unpack=True)

# Calcula los periodogramas
freqs1, psd1 = periodogram(data1, fs=1/sampling_rate1)
freqs2, psd2 = periodogram(data2, fs=1/sampling_rate2)

# Encuentra la frecuencia con amplitud más alta en cada señal
max_freq1 = freqs1[np.argmax(psd1)]
max_freq2 = freqs2[np.argmax(psd2)]

# Muestra las frecuencias más altas en Hz
print("La frecuencia más acelerada en terremoto1 es:",max_freq1, "Hz")
print("La frecuencia más acelerada en terremoto2 es:",max_freq2, "Hz")

# Compara las frecuencias más altas
if max_freq1 > max_freq2:
    print("La frecuencia más acelerada se encuentra en terremoto 1")
elif max_freq1 < max_freq2:
    print("La frecuencia más acelerada se encuentra en terremoto 2")
else:
    print("La frecuencia más acelerada es igual en ambas señales")

'''
e)
Se midio la señal terremoto3.txt pero se perdió el registro de la ubicación. Determine cual de los
detectores de las dos primeras señales estaría mas próximo. Justifique.
'''
from scipy.signal import correlate

# Carga los datos de los terremotos
data1 = np.loadtxt('terremoto1.txt', usecols=[1], unpack=True)
data2 = np.loadtxt('terremoto2.txt', usecols=[1], unpack=True)
data3 = np.loadtxt('terremoto3.txt', usecols=[1], unpack=True)

# Calcula la correlación cruzada entre los terremotos
corr1 = correlate(data1, data3)
corr2 = correlate(data2, data3)

# Encuentra el valor máximo de la correlación cruzada
max_corr1 = np.max(corr1)
max_corr2 = np.max(corr2)

# Imprime los resultados
print('La correlación máxima entre terremoto 1 y terremoto 3 es ' + str(max_corr1))
print('La correlación máxima entre terremoto 2 y terremoto 3 es ' + str(max_corr2))

# Compara las correlaciones
if max_corr1 > max_corr2:
    print('terremoto1 está más cerca de terremoto3')
else:
    print('terremoto2 está más cerca de terremoto3')


# Crea una figura
fig, ax = plt.subplots()

# Grafica la correlación cruzada entre terremoto1 y terremoto3
#ax.plot(corr1, label='Terremoto1')

# Grafica la correlación cruzada entre terremoto2 y terremoto3
ax.plot(corr2, label='Terremoto2')
#
# Añade una leyenda
ax.legend()

# Muestra la figura
plt.show()

"""
Los picos en el gráfico indican los desplazamientos donde las dos señales son más similares
Al comparar los picos de las correlaciones cruzadas, puedes ver visualmente cuál terremoto está más cerca de "terremoto3".
"""
