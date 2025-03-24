# Señales electromiográficas EMG  
 LABORATORIO - 4 PROCESAMIENTO DIGITAL DE SEÑALES


## Requisitos
- Python 3.9
- Bibliotecas necesarias:
  - wfdb
  - numpy
  - matplotlib
  - seaborn

Instalar dependencias:
```python
pip install wfdb numpy matplotlib seaborn
```
----
## Introducción

En este laboratorio  se observo cómo se comportan las señales tanto en el tiempo como en la frecuencia. Lo haremos aplicando tres técnicas fundamentales: la convolución, la correlación y la transformada de Fourier. Además del análisis de una señal electromiografía (EMG).

-----

## Convolución
La convolución es una operación matemática que combina dos funciones para describir la superposición entre ambas. La convolución toma dos funciones, “desliza” una sobre la otra, multiplica los valores de las funciones en todos los puntos de superposición, y suma los productos para crear una nueva función. Este proceso crea una nueva función que representa cómo interactúan las dos funciones originales entre sí.
La convolución se utiliza en el procesamiento digital de señales para estudiar y diseñar sistemas lineales de tiempo invariante (LTI), como los filtros digitales.
La señal de salida de un sistema LTI,y[n], es la convolución de la señal de entrada x[n] y la respuesta al impulso h[n] del sistema.[1]
### Fórmula de la convolución discreta:

$$
y[n] = \sum_{k=0}^{M-1} x[k] h[n-k]
$$

Donde:

- \(y[n]\) es la señal de salida.
- \(x[k]\) es la señal de entrada.
- \(h[n-k]\) es la respuesta al impulso del sistema desplazada en el tiempo.
- \(M\) es la longitud de la señal de entrada.

### 1. Convolución entre la señal x[n] y del sistema h[n]
```python
h = [5,6,0,0,7,7,5]
x = [1,0,1,4,6,6,0,7,0,8]
y = np.convolve(x,h,mode='full')
print('h[n] =', h)
print('x[n] =',x)
print('y[n] =',y)
```
$$
h[n] = \begin{bmatrix}
5 & 6 & 0 & 0 & 7 & 7 & 5
\end{bmatrix}
$$

$$
x[n] = \begin{bmatrix}
1 & 0 & 1 & 4 & 6 & 6 & 0 & 7 & 0 & 8
\end{bmatrix}
$$

$$
y[n] = \begin{bmatrix}
5 & 6 & 5 & 26 & 61 & 73 & 48 & 70 & 117 & 144 & 120 & 79 & 49 & 91 & 56 & 40
\end{bmatrix}
$$

Este código en Python calcula la convolución discreta entre dos señales utilizando la función np.convolve() de NumPy. Primero, se definen dos listas, h y x, que representan la respuesta al impulso de un sistema y una señal de entrada, respectivamente. Luego, se aplica la convolución entre estas dos señales usando np.convolve(x, h, mode='full'), lo que genera una nueva señal y cuya longitud es la suma de las longitudes de x y h menos uno. La convolución es una operación fundamental en procesamiento de señales, ya que permite analizar cómo una señal se ve afectada por un sistema. Finalmente, el código imprime las señales h, x y y para visualizar los datos y el resultado de la convolución.

---

### 2. Grafico de la señal x[n] y del sistema h[n]
```python
fig = plt.figure(figsize=(10, 5)) 
plt.plot(h,color='g')
plt.stem(range(len(h)), h)
plt.title("Sistema (santiago)")  
plt.xlabel("(n)") 
plt.ylabel("h [n]") 
plt.grid()
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/b400a6c8-f58f-4757-a74c-ed36a19d3d59" alt="imagen" width="450">
</p>

```python
fig = plt.figure(figsize=(10, 5)) 
plt.plot(x,color='g')
plt.stem(range(len(x)), x)
plt.title("Señal (santiago)")  
plt.xlabel("(n)") 
plt.ylabel("x [n]") 
plt.grid()  
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/fa6848b8-bb89-4478-9399-7f6207653284" alt="imagen" width="450">
</p>

Este código genera dos gráficos para representar la respuesta al impulso h[n] y la señal de entrada x[n]. Para cada una, se crea una figura de 10x5 y se trazan dos representaciones: una línea verde (plt.plot()) y un gráfico de tipo stem (plt.stem()) para resaltar los valores discretos.

---

### 3. Grafico de la convolución
```python
fig = plt.figure(figsize=(10, 5)) 
plt.plot(y,color='g')
plt.title("Señal Resultante (santiago)")  
plt.xlabel("(n)") 
plt.ylabel("y [n]") 
plt.grid() 
plt.stem(range(len(y)), y)
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/030a2690-8be0-4fd6-bf8f-c76ff7ca80f9" alt="imagen" width="450">
</p>



Este fragmento de código genera un gráfico de la señal resultante y[n], que es el resultado de la convolución entre x[n] y h[n]. Se traza la señal con una línea verde usando plt.plot(y, color='g'). Luego, se superpone un gráfico de tipo stem con plt.stem(range(len(y)), y), resaltando los valores discretos de la señal.

---


## Correlación
La correlación en señales mide estadísticamente cómo dos señales varían de manera conjunta, evaluando su similitud o relación lineal. Es clave en el procesamiento de señales, ya que permite analizar sincronización, patrones y dependencias entre flujos de datos o formas de onda. Prácticamente, se usa para detectar similitudes, identificar patrones, filtrar ruido y extraer información relevante.[2]
### Fórmula de la correlación cruzada:

$$
R_{xy}[n] = \sum_{k=-\infty}^{\infty} x[k] y[k+n]
$$

Donde:
- \(R_{xy}[n]\) es la correlación cruzada entre \(x\) y \(y\).
- \(x[k]\) y \(y[k+n]\) representan las señales en diferentes desplazamientos temporales.


### 1. Señal Cosenoidal
```python
Ts = 1.25e-3
n = np.arange(0, 9) #valores enteros
x1 = np.cos(2*np.pi*100*n*Ts)
fig = plt.figure(figsize=(10, 5)) 
plt.plot(n, x1, label="", color='black')
plt.title("Señal Cosenoidal")  
plt.xlabel("(n)") 
plt.ylabel("x1 [nTs]") 
plt.grid()
plt.stem(range(len(x1)), x1)
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/a6b5c5b2-e536-418f-9f35-36c75dd033bd" alt="imagen" width="450">
</p>

Se genera y grafica una señal cosenoidal muestreada. Primero, se define un periodo de muestreo Ts = 1.25e-3, y luego se crea un arreglo n con valores enteros de 0 a 8 usando np.arange(0, 9). La función np.arange(inicio, fin) genera una secuencia de números desde inicio hasta fin-1 con un paso de 1 por defecto. En este caso, n representa los instantes de muestreo en el dominio discreto.

A partir de n, se calcula la señal x1 como un coseno de 100 Hz evaluado en los instantes n * Ts. Para la visualización, se crea una figura de tamaño 10x5, donde plt.plot(n, x1, color='black') traza la señal con una línea negra, y plt.stem(range(len(x1)), x1) resalta los valores discretos.


---

### 2. Señal Senoidal
```python
x2 = np.sin(2*np.pi*100*n*Ts)
fig = plt.figure(figsize=(10, 5)) 
plt.plot(n, x2, label="", color='black')
plt.title("Señal Senoidal")  
plt.xlabel("(n)") 
plt.ylabel("x2 [nTs]") 
plt.grid()
plt.stem(range(len(x2)), x2)
```
<p align="center">
    <img src="https://github.com/user-attachments/assets/0926d754-2336-4313-8858-af1f28e19ed2" alt="imagen" width="450">
</p>

Al igual que en la gráfica anterior, este código genera y visualiza una señal, pero en este caso es una señal senoidal en lugar de una cosenoidal. Se usa el mismo conjunto de valores n = np.arange(0, 9), generado con np.arange(), y se calcula x2 como un seno de 100 Hz evaluado en los instantes n * Ts.

---

### 3. Correlación de las Señales y Representación Grafica
```python
correlacion = np.correlate(x1,x2,mode='full')
print('Correlación =',correlacion)
fig = plt.figure(figsize=(10, 5)) 
plt.plot(correlacion, color='black')
plt.stem(range(len(correlacion)), correlacion)
plt.title("Correlación")  
plt.xlabel("(n)") 
plt.ylabel("R[n]") 
plt.grid()
```
Se calcula y grafica la correlación cruzada entre las señales x1 y x2. La correlación mide la similitud entre dos señales a diferentes desplazamientos en el tiempo, lo que permite identificar patrones compartidos o desfases entre ellas.

Primero, np.correlate(x1, x2, mode='full') computa la correlación cruzada, generando una nueva señal correlacion, cuya longitud es len(x1) + len(x2) - 1. Luego, el resultado se imprime en la consola.

$$
\text{Correlación} = \begin{bmatrix}
-2.44929360 \times 10^{-16} & -7.07106781 \times 10^{-1} & -1.50000000 & -1.41421356 \\
-1.93438661 \times 10^{-16} & 2.12132034 \times 10^{0} & 3.50000000 & 2.82842712 \\
8.81375476 \times 10^{-17} & -2.82842712 \times 10^{0} & -3.50000000 & -2.12132034 \\
3.82856870 \times 10^{-16} & 1.41421356 \times 10^{0} & 1.50000000 & 7.07106781 \times 10^{-1} \\
0.00000000 \times 10^{0}
\end{bmatrix}
$$

<p align="center">
    <img src="https://github.com/user-attachments/assets/c0028249-0f51-430a-bebc-44794b47bfc0" alt="imagen" width="450">
</p>

Para visualizar la correlación, se crea una figura de 10x5 donde plt.plot(correlacion, color='black') dibuja la señal con una línea negra, mientras que plt.stem(range(len(correlacion)), correlacion) resalta sus valores discretos. 

---
## Transformación (Señal Electromiografica)
### 1. Caracterizacion en Función del Tiempo 
```python
datos = wfdb.rdrecord('session1_participant1_gesture10_trial1') 
t = 1500
señal = datos.p_signal[:t, 0] 
fs = datos.fs
```
Se carga una señal de electromiografía (EMG) y se extraen los primeros 1500 puntos.

#### 1.1. Estadisticos Descriptivos y frecuencia de muestreo
Se calculan los siguientes estadísticos:
- *Media (μ):* Valor promedio de la señal.
- *Desviación Estándar (σ):* Medida de la dispersión de los datos respecto a la media.
- *Coeficiente de Variación (CV):* Relación entre desviación estándar y media, expresada en porcentaje.
```python
def caracterizacion():
    print()
    print()
    media = np.mean(señal)
    desvesta = np.std(señal)
    print('Media de la señal:',np.round(media,6))
    print('Desviación estándar:',np.round(desvesta,6))
    print("Coeficiente de variación:",np.round((media/desvesta),6))
    print('Frecuencia de muestreo:',fs,'Hz')
    
    fig = plt.figure(figsize=(8, 4))
    sns.histplot(señal, kde=True, bins=30, color='black')
    plt.hist(señal, bins=30, edgecolor='blue')
    plt.title('Histograma de Datos')
    plt.xlabel('datos')
    plt.ylabel('Frecuencia')

caracterizacion()
```
- Media de la señal: 0.000131
- Desviación estándar: 0.071519
- Coeficiente de variación: 0.001834
- Frecuencia de muestreo: 2048 Hz
<p align="center">
    <img src="https://github.com/user-attachments/assets/d64b9102-821a-4754-a102-2a5977baab0c" alt="imagen" width="450">
</p>

- Histograma:El histograma resultante tiene una distribución que se asemeja a una campana de Gauss, lo cual es un fuerte indicativo de una distribución normal en los datos. Esto significa que la mayoría de los valores están concentrados alrededor del promedio, mientras que las frecuencias disminuyen gradualmente hacia ambos extremos.
  
#### 1.2. Grafica de Electromiografía
```python
fig = plt.figure(figsize=(10, 5)) 
plt.plot(señal, color='m')
plt.title("Electromiografía [EMG]")  
plt.xlabel("muestras[n]") 
plt.ylabel("voltaje [mv]") 
plt.grid()
```
<p align="center">
    <img src="https://github.com/user-attachments/assets/9ea890d5-b5b2-46e3-aaf5-dd1f556bec0b" alt="imagen" width="450">
</p>

### 2. Descripción la señal en cuanto a su clasificación 

La señal electromiográfica (EMG) es el registro de la actividad eléctrica generada por los músculos esqueléticos. Se clasifica como una señal biomédica no estacionaria y altamente variable debido a factores como la activación muscular, la fatiga, la calidad de los electrodos y el ruido ambiental. Su análisis en los dominios temporal y espectral permite extraer información relevante para aplicaciones como el control de prótesis, el diagnóstico de trastornos neuromusculares y el estudio del rendimiento deportivo. En el análisis temporal, se evalúan parámetros como la amplitud y la duración de los potenciales de acción de las unidades motoras. En el análisis espectral, técnicas como la transformada de Fourier y el análisis wavelet permiten descomponer la señal y caracterizar su dinámica.[3][4]

### 3. Tranformada de Fourier
La transformada de Fourier permite convertir una señal del dominio del tiempo al dominio de la frecuencia.

### Fórmula de la Transformada de Fourier Discreta (DFT):

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2 \pi k n / N}
$$

Donde:

- \(X[k]\) es la representación en frecuencia de la señal.
- \(x[n]\) es la señal original en el dominio del tiempo.
- \(N\) es el número total de muestras.
- \(e^{-j 2 \pi k n / N}\) representa la base exponencial compleja.

La DFT utiliza una suma ponderada de las muestras de la señal con bases exponenciales complejas para transformar la señal desde el tiempo hacia el dominio de la frecuencia.

#### 3.1. Grafica de la transformada de fourier
El siguiente código muestra cómo calcular y graficar la transformada de Fourier de una señal:
```python
N = len(señal)
frecuencias = np.fft.fftfreq(N, 1/fs)
transformada = np.fft.fft(señal) / N
magnitud = (2 * np.abs(transformada[:N//2]))**2

plt.figure(figsize=(10, 5))
plt.plot(frecuencias[:N//2], np.abs(transformada[:N//2]), color='black')
plt.title("Transformada de Fourier de la Señal")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()
```
- np.fft.fft: Calcula la transformada de Fourier de la señal.
- np.fft.fftfreq: Devuelve las frecuencias correspondientes a cada componente de la transformada.
- N//2: Se utiliza para considerar únicamente las frecuencias positivas.
- plt.plot: Genera una gráfica de la magnitud de la transformada.

Esta gráfica muestra las frecuencias presentes en la señal y su magnitud asociada.

<p align="center">
    <img src="https://github.com/user-attachments/assets/bbd94695-07fa-475c-8b62-6d6f98bfd948" alt="imagen" width="450">
</p>

#### 3.2. Grafica de la densidad espectral
En la práctica, para señales discretas y de duración finita, la DEP se estima utilizando la transformada de Fourier discreta (DFT). Al calcular la DFT de una señal y normalizar adecuadamente, se obtiene una estimación de su densidad espectral de potencia. Esta estimación permite identificar las frecuencias predominantes y analizar cómo se distribuye la energía de la señal en el dominio de la frecuencia.
```python
plt.figure(figsize=(10, 5))
plt.plot(frecuencias[:N//2], magnitud, color='black')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia')
plt.title('Densidad espectral de la señal')
plt.grid()

plt.show()
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/d5f9a88b-1baf-47e8-86c6-3f8b68610271" alt="imagen" width="450">
</p>

- La Densidad Espectral de Potencia (DSP) mide la potencia de una señal en función de la frecuencia.[5]
- magnitud: Representa la potencia de cada frecuencia, calculada como el cuadrado de la magnitud de la transformada de Fourier.
Ambas gráficas son fundamentales para comprender el comportamiento de la señal en el dominio de la frecuencia. La primera da información sobre las frecuencias presentes, mientras que la segunda muestra cómo se distribuye la energía de la señal en esas frecuencias.

----
## Conclusión

- La comparación entre correlación y convolución resaltó que la correlación mide la similitud sin invertir la señal, mostrando la independencia entre señales senoidales y cosenoidales.
- El análisis de señales EMG en los dominios de tiempo y frecuencia nos permite caracterizar su comportamiento, y la DFT es crucial para identificar las distribuciones de frecuencia y potencia dominantes.

----
## Bibliografias
- [1] https://la.mathworks.com/discovery/convolution.html
- [2] https://radartopix.com/es/que-es-la-correlacion-en-las-senales/
- [3] https://link.springer.com/chapter/10.1007/978-3-540-74471-9_31
- [4] https://scielo.isciii.es/scielo.php?script=sci_arttext&pid=S1137-66272009000600003
- [5] https://prezi.com/p/cwcmwut1n1fx/densidad-espectral-de-potencia/#:~:text=La%20Densidad%20Espectral%20de%20Potencia%20(DSP)%20mide%20la%20potencia%20de,frecuencias%20en%20una%20se%C3%B1al%20analizada
----
## Autores 
- Samuel Peña
- Ana Abril
- Santiago Mora
