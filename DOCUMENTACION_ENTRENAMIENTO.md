# Documentación: Entrenamiento de PhaseNet con SeisBench

## Resumen del Proceso

Este documento describe el proceso completo de validación y entrenamiento de un modelo PhaseNet usando SeisBench con datos locales.

## 1. Validación de Datos

### Archivos Creados

- `validar_seisbench.py` - Validador completo de archivos SeisBench
- `explorar_hdf5.py` - Explorador de archivos HDF5
- `convertir_a_seisbench.py` - Convertidor de formato

### Proceso de Validación

1. **Validación inicial**: Se detectó que el archivo `waveforms.hdf5` no tenía la estructura esperada
2. **Exploración**: Se analizó la estructura interna del HDF5
3. **Conversión**: Se generó un nuevo archivo compatible con SeisBench
4. **Validación final**: Se confirmó que los datos cumplen con el estándar

### Resultado

```
🎯 RESULTADO GENERAL: ✅ VÁLIDO
- metadata.csv: ✅ VÁLIDO
- waveforms.hdf5: ✅ VÁLIDO
- Consistencia: ✅ VÁLIDO
```

## 2. Entrenamiento de PhaseNet

### Archivos Creados

- `entrenar_phasenet.py` - Versión inicial (requiere ajustes)
- `entrenar_phasenet_simple.py` - Versión funcional para un canal
- `explorar_seisbench.py` - Explorador de funcionalidades de SeisBench

### Problemas Encontrados y Soluciones

#### Problema 1: Importación de LocalSeismicDataset

**Error**: `ImportError: cannot import name 'LocalSeismicDataset'`
**Solución**: Usar `WaveformDataset` en su lugar

#### Problema 2: Importación de pick_generator

**Error**: `ImportError: cannot import name 'pick_generator'`
**Solución**: Usar `PickLabeller` o entrenamiento manual

#### Problema 3: Número de componentes

**Error**: `Number of source and target components needs to be identical. Got 3!=2`
**Solución**: Ajustar `component_order="Z"` para datos de un solo canal

### Configuración Final

```python
# Configuración exitosa
DATA_DIR = "seisbench_data"
BATCH_SIZE = 1
EPOCHS = 10
DEVICE = "cpu"  # o "cuda" si hay GPU

# Modelo PhaseNet para un canal
model = PhaseNet(
    in_channels=1,  # Un solo canal
    classes=2,      # P y S
    phases=['P', 'S']
)
```

## 3. Resultados del Entrenamiento

### Estadísticas Finales

```
📊 Resumen del entrenamiento:
   - Épocas completadas: 10
   - Pérdida final: 0.974422
   - Dispositivo usado: cpu
   - Tamaño de batch: 1
   - Datos de entrenamiento: 1
   - Datos de validación: 1
```

### Archivo del Modelo

- **Ubicación**: `phasenet_trained.pt`
- **Contenido**:
  - Estado del modelo
  - Estado del optimizador
  - Configuración del modelo
  - Métricas de entrenamiento

## 4. Uso del Modelo Entrenado

### Cargar el Modelo

```python
import torch
from seisbench.models import PhaseNet

# Cargar modelo entrenado
checkpoint = torch.load('phasenet_trained.pt')
model = PhaseNet(
    in_channels=1,
    classes=2,
    phases=['P', 'S']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Inferencia

```python
# Preparar datos
waveform = torch.FloatTensor(tu_forma_de_onda).unsqueeze(0).unsqueeze(0)
# [1, 1, samples] - [batch, channel, samples]

# Predicción
with torch.no_grad():
    predictions = model(waveform)
    # predictions: [1, 2, samples] - [batch, phases, samples]
```

## 5. Estructura de Archivos Final

```
3_validar/
├── seisbench_data/
│   ├── metadata.csv                    # Metadatos validados
│   ├── waveforms.hdf5                  # Formas de onda convertidas
│   └── waveforms_backup_*.hdf5         # Backup del archivo original
├── validar_seisbench.py                # Validador principal
├── explorar_hdf5.py                    # Explorador HDF5
├── convertir_a_seisbench.py            # Convertidor de formato
├── entrenar_phasenet_simple.py         # Entrenamiento funcional
├── explorar_seisbench.py               # Explorador SeisBench
├── phasenet_trained.pt                 # Modelo entrenado
├── requirements.txt                    # Dependencias
├── README.md                          # Documentación general
└── DOCUMENTACION_ENTRENAMIENTO.md     # Este archivo
```

## 6. Comandos de Ejecución

### Validación

```bash
python validar_seisbench.py
```

### Entrenamiento

```bash
python entrenar_phasenet_simple.py
```

### Exploración

```bash
python explorar_hdf5.py
python explorar_seisbench.py
```

## 7. Notas Importantes

### Limitaciones del Entrenamiento Actual

1. **Datos limitados**: Solo 2 trazas de entrenamiento
2. **Etiquetas dummy**: No hay picks reales P y S
3. **Un solo canal**: Solo datos BHZ

### Mejoras Recomendadas

1. **Más datos**: Agregar más trazas sísmicas
2. **Etiquetas reales**: Incluir picks P y S reales
3. **Múltiples canales**: Agregar BHE y BHN
4. **Validación cruzada**: Implementar k-fold cross-validation
5. **Métricas**: Agregar F1-score, precisión, recall

### Para Producción

1. **GPU**: Usar GPU para entrenamiento más rápido
2. **Batch size**: Aumentar batch size con más datos
3. **Early stopping**: Implementar parada temprana
4. **Checkpoints**: Guardar checkpoints durante entrenamiento
5. **Logging**: Implementar logging detallado

## 8. Troubleshooting

### Errores Comunes

1. **ImportError**: Verificar versión de SeisBench (0.9.0)
2. **Component mismatch**: Ajustar `component_order`
3. **Memory error**: Reducir batch size
4. **CUDA error**: Cambiar a CPU si no hay GPU

### Soluciones

1. **Reinstalar dependencias**: `pip install -r requirements.txt`
2. **Verificar datos**: Usar `validar_seisbench.py`
3. **Explorar estructura**: Usar `explorar_hdf5.py`
4. **Ajustar parámetros**: Modificar configuración según datos

## 9. Referencias

- [SeisBench Documentation](https://github.com/seisbench/seisbench)
- [PhaseNet Paper](https://doi.org/10.1029/2018GL080595)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [HDF5 Documentation](https://www.hdfgroup.org/solutions/hdf5/)

---

**Fecha de creación**: 18 de Junio, 2025  
**Versión**: 1.0  
**Autor**: Asistente IA  
**Estado**: Completado ✅
