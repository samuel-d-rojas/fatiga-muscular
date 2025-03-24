# Señales Electromiográficas EMG  
 LABORATORIO - 4 PROCESAMIENTO DIGITAL DE SEÑALES


## Requisitos
- Python 3.9
- Bibliotecas necesarias:
  - wfdb
  - numpy
  - matplotlib
  - seaborn
## Introducción
En este laboratorio, se realizó la adquisición y análisis de señales electromiográficas (EMG) con el objetivo de estudiar la fatiga muscular a través del procesamiento digital de señales. Para ello, se utilizaron electrodos de superficie y un sistema de adquisición de datos (DAQ), permitiendo registrar la actividad eléctrica de los músculos durante una contracción sostenida. Posteriormente, se aplicaron técnicas de filtrado y análisis espectral mediante la Transformada de Fourier (FFT) para identificar cambios en la frecuencia de la señal, lo que permitió evaluar la fatiga en el musculo estudiado.

_ _ _

## 1) Preparación del Sujeto
Se analizó el músculo flexor común de los dedos, encargado de la flexión de los dedos de la mano y fundamental en la prensión y manipulación de objetos. Desde el punto de vista de la fisiología muscular, su frecuencia de contracción se encuentra en un rango de 10 a 50 Hz, lo que indica la actividad eléctrica generada durante su activación. Para analizar esta actividad, se utilizó un sensor de electromiografía (EMG), específicamente el Muscle Sensor v3, junto con un sistema de adquisición de datos (DAQ). Este sistema permitió registrar y almacenar la señal electromiográfica, para posteriormente realizar su correspondiente procesamiento digital. 

<p align="center">
    <img src="https://github.com/user-attachments/assets/d0a7cfdc-6328-43ff-8884-5fc5ccb0735b" alt="imagen" width="200" height="200">
    <img src="https://github.com/user-attachments/assets/841d468d-fa1d-41aa-a9ed-9167ff048171" alt="imagen" width="350" height="200">
</p>

Se empleó una configuración diferencial, que consiste en colocar dos electrodos activos sobre el mismo músculo y un tercer electrodo en una zona de referencia. En este método, se contrasta la señal registrada por los dos electrodos activos para suprimir interferencias y ruidos comunes, como el ambiental o el generado por músculos adyacentes. El electrodo de referencia, ubicado en una región eléctricamente estable (por ejemplo, en un punto óseo), El electrodo de referencia se utiliza para establecer un potencial de base o cero contra el cual se comparan las señales de los electrodos activos. 

<p align="center">
    <img src="https://github.com/user-attachments/assets/fbf058b1-05ab-48ff-bd4d-951703bb5857" alt="imagen" width="200">
</p>

Para determinar la frecuencia de muestreo, se siguió el teorema de Nyquist. Dado que la frecuencia máxima en este caso es de 50 Hz, la frecuencia de muestreo debe ser mayor o igual a 100 Hz para garantizar una correcta reconstrucción de la señal. Para esta caso se utilizo 1000 Hz de frecuencia.

$$
f_s \geq 2f_{\text{max}}
$$

$$
100 \geq 2(100)
$$


_ _ _

## 2) Adquisición de la Señal EMG:

La adquisición de la señal se realizó utilizando Python, a través de un repositorio de GitHub [2].

```python
# Importamos las librerías necesarias
import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
```
```python
# Definimos la tasa de muestreo en Hz (muestras por segundo)
sample_rate = 1000  # 1000 muestras por segundo (1 kHz)

# Definimos la duración de la adquisición en minutos
duration_minutes = 2  

# Convertimos la duración a segundos
duration_seconds = duration_minutes * 60  

# Calculamos el número total de muestras necesarias
num_samples = int(sample_rate * duration_seconds)  

# Creamos una tarea para la adquisición de datos con NI-DAQmx
with nidaqmx.Task() as task:
    
    # Agregamos un canal de entrada analógica para medir voltaje en "Dev3/ai0"
    task.ai_channels.add_ai_voltage_chan("Dev3/ai0")
    
    # Configuramos el temporizador de muestreo:
    task.timing.cfg_samp_clk_timing(
        sample_rate,  # Tasa de muestreo en Hz
        sample_mode=AcquisitionType.FINITE,  # Adquisición finita (un número fijo de muestras)
        samps_per_chan=num_samples  # Número total de muestras a adquirir
    )
    
    # Iniciamos la adquisición de datos
    task.start()

    # Esperamos hasta que la tarea termine (con un margen extra de 10 segundos)
    task.wait_until_done(timeout=duration_seconds + 10)

    # Leemos los datos adquiridos del canal
    data = task.read(number_of_samples_per_channel=num_samples)

# Creamos un eje de tiempo que va desde 0 hasta la duración total en segundos
time_axis = np.linspace(0, duration_seconds, num_samples, endpoint=False)
```

## 3) Filtrado de la Señal:

## 4) Aventanamiento:

## 5) Análisis Espectral:

## Conclusión

## Bibliografias
[1] Pololu, "Muscle Sensor v3 User’s Manual," [Online]. Available: https://www.pololu.com/file/0J745/Muscle_Sensor_v3_users_manual.pdf. [Accessed: 24-Mar-2025].

[2] National Instruments, "NI-DAQmx Python API," GitHub repository, [Online]. Available: https://github.com/ni/nidaqmx-python. [Accessed: 24-Mar-2025].

_ _ _

## Autores 
- Samuel Joel Peña Rojas
