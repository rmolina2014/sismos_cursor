# Validador de Archivos SeisBench

Este programa valida si los archivos en la carpeta `seisbench_data` cumplen con el formato estándar de SeisBench.

## Características

- ✅ Valida la existencia de archivos requeridos (`metadata.csv` y `waveforms.hdf5`)
- 📊 Verifica la estructura del archivo `metadata.csv`
- 🔍 Valida el contenido y tipos de datos del metadata
- 📈 Analiza la estructura del archivo `waveforms.hdf5`
- 📊 Valida el contenido de las formas de onda
- 🔗 Verifica la consistencia entre metadata y waveforms
- 📋 Genera un reporte completo de validación

## Instalación

1. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Ejecuta el validador desde la línea de comandos:

```bash
python validar_seisbench.py
```

## Estructura de Archivos Esperada

```
seisbench_data/
├── metadata.csv      # Metadatos de las trazas sísmicas
└── waveforms.hdf5    # Formas de onda en formato HDF5
```

## Columnas Requeridas en metadata.csv

- `trace_name`: Nombre único de la traza
- `split`: División del dataset (train/validation/test)
- `station`: Código de la estación sísmica
- `channel`: Canal de la estación (ej: BHZ, BHE, BHN)
- `start_time`: Tiempo de inicio de la traza
- `end_time`: Tiempo de fin de la traza
- `sampling_rate`: Tasa de muestreo en Hz

## Columnas Opcionales

- `latitude`: Latitud de la estación
- `longitude`: Longitud de la estación
- `magnitude`: Magnitud del evento sísmico
- `depth`: Profundidad del evento
- `distance`: Distancia epicentral
- `source_depth`: Profundidad de la fuente
- `source_latitude`: Latitud de la fuente
- `source_longitude`: Longitud de la fuente

## Ejemplo de Salida

```
🚀 Iniciando validación completa de archivos SeisBench...
📁 Directorio: seisbench_data

🔍 Validando existencia de archivos...
✅ metadata.csv encontrado: seisbench_data/metadata.csv
✅ waveforms.hdf5 encontrado: seisbench_data/waveforms.hdf5

📊 Validando estructura de metadata.csv...
✅ Columna requerida 'trace_name' encontrada
✅ Columna requerida 'split' encontrada
✅ Columna requerida 'station' encontrada
✅ Columna requerida 'channel' encontrada
✅ Columna requerida 'start_time' encontrada
✅ Columna requerida 'end_time' encontrada
✅ Columna requerida 'sampling_rate' encontrada
✅ Columna opcional 'latitude' encontrada
✅ Columna opcional 'longitude' encontrada
✅ Columna opcional 'magnitude' encontrada

📋 Primeras filas del metadata:
                    trace_name split station channel                    start_time                      end_time  sampling_rate  latitude  longitude  magnitude
0  2024-01-01-0033-42L_trace train     SJA    BHZ  2025-06-18T12:22:44.435611  2025-06-18T12:22:44.435611          100.0   -30.564    -69.712        2.4
1  2024-01-01-0100-06R_trace train     SJA    BHZ  2025-06-18T12:22:44.435611  2025-06-18T12:22:44.435611          100.0   -30.564    -69.712        2.4

🔍 Validando contenido de metadata.csv...
✅ sampling_rate: valores válidos (rango: 100.0-100.0 Hz)
✅ start_time: formato de fecha válido
✅ end_time: formato de fecha válido
✅ latitude: valores válidos
✅ longitude: valores válidos
✅ magnitude: valores válidos (rango: 2.4-2.4)
✅ Total de registros válidos: 2
✅ Splits encontrados: ['train']
✅ Estaciones encontradas: ['SJA']
✅ Canales encontrados: ['BHZ']

📊 Validando estructura de waveforms.hdf5...
📁 Grupos encontrados: ['2024-01-01-0033-42L_trace', '2024-01-01-0100-06R_trace']
📊 Dataset '2024-01-01-0033-42L_trace': forma (1000,), tipo float32
📊 Dataset '2024-01-01-0100-06R_trace': forma (1000,), tipo float32
✅ Datasets de formas de onda encontrados: 2

🔍 Validando contenido de waveforms.hdf5...

📊 Analizando dataset: 2024-01-01-0033-42L_trace
   Forma: (1000,)
   Tipo: float32
   Rango: [-0.123456, 0.123456]
   Media: 0.000000
   Desv. Est.: 0.045678
   ✅ Forma de onda 1D válida

📊 Analizando dataset: 2024-01-01-0100-06R_trace
   Forma: (1000,)
   Tipo: float32
   Rango: [-0.234567, 0.234567]
   Media: 0.000000
   Desv. Est.: 0.056789
   ✅ Forma de onda 1D válida

✅ No se encontraron problemas en los datos

🔗 Validando consistencia entre metadata y waveforms...
📋 Trazas en metadata: 2
📊 Datasets en HDF5: 2
✅ Correspondencia perfecta entre metadata y waveforms

============================================================
📋 REPORTE COMPLETO DE VALIDACIÓN SEISBENCH
============================================================

🎯 RESULTADO GENERAL: ✅ VÁLIDO

📊 FILE EXISTENCE:
   Estado: ✅ VÁLIDO

📊 METADATA STRUCTURE:
   Estado: ✅ VÁLIDO

📊 METADATA CONTENT:
   Estado: ✅ VÁLIDO

📊 WAVEFORMS STRUCTURE:
   Estado: ✅ VÁLIDO

📊 WAVEFORMS CONTENT:
   Estado: ✅ VÁLIDO

📊 CONSISTENCY:
   Estado: ✅ VÁLIDO

💡 RECOMENDACIONES:
   ✅ Los archivos cumplen con el formato SeisBench
   ✅ Los datos están listos para usar con SeisBench
```

## Códigos de Salida

- `0`: Validación exitosa - Los archivos cumplen con el formato SeisBench
- `1`: Validación fallida - Se encontraron problemas que deben corregirse

## Notas

- El programa valida tanto la estructura como el contenido de los archivos
- Se verifica la consistencia entre los metadatos y las formas de onda
- Se generan estadísticas detalladas de los datos
- El reporte incluye recomendaciones específicas para corregir problemas
