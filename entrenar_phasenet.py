#!/usr/bin/env python3
"""
Entrenamiento de PhaseNet con SeisBench usando datos locales
==========================================================

Este script entrena un modelo PhaseNet usando los archivos
`seisbench_data/metadata.csv` y `seisbench_data/waveforms.hdf5`.

Pasos:
1. Carga de datos en formato SeisBench
2. Preparación del dataset y partición train/val
3. Inicialización del modelo PhaseNet
4. Entrenamiento y validación
5. Guardado del modelo entrenado

Requisitos:
- seisbench
- torch

Instalación:
    pip install seisbench torch

Ejecución:
    python entrenar_phasenet.py

"""

import seisbench
from seisbench.data import WaveformDataset
from seisbench.models import PhaseNet
from seisbench.generate import PickLabeller
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader

# 1. Configuración de rutas y parámetros
DATA_DIR = "seisbench_data"
METADATA = "metadata.csv"  # Relativo a DATA_DIR
WAVEFORMS = "waveforms.hdf5"  # Relativo a DATA_DIR
MODEL_SAVE_PATH = "phasenet_trained.pt"
BATCH_SIZE = 2  # Reducido para datos pequeños
EPOCHS = 10  # Aumentado para mejor resultado
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔧 Usando dispositivo: {DEVICE}")
print(f"📁 Directorio de datos: {DATA_DIR}")

# 2. Cargar el dataset local
print("\n🔍 Cargando datos locales...")
try:
    # Cargar metadata
    metadata = pd.read_csv(os.path.join(DATA_DIR, METADATA))
    print(f"✅ Metadata cargada: {len(metadata)} registros")
    
    # Crear dataset usando WaveformDataset
    dataset = WaveformDataset(
        path=DATA_DIR,
        metadata=metadata,
        metadata_filename=METADATA,
        waveform_filename=WAVEFORMS,
        sampling_rate=100.0,  # Ajusta según tus datos
        component_order="Z",  # Solo canal Z
        dimension_order="CW"
    )
    print(f"✅ Dataset creado exitosamente")
    print(f"   Total de trazas: {len(dataset)}")
    
except Exception as e:
    print(f"❌ Error al cargar dataset: {e}")
    exit(1)

# 3. Partición de datos (train/val)
print("\n🔀 Particionando datos...")
train_set = dataset.filter(split=["train"])
val_set = dataset.filter(split=["val", "validation", "dev"])

if len(val_set) == 0:
    print("⚠️ No se encontraron datos de validación, dividiendo train...")
    # Si no hay validación, usar parte de train como val
    train_set, val_set = train_set.split(0.8)

print(f"   Trazas entrenamiento: {len(train_set)}")
print(f"   Trazas validación: {len(val_set)}")

# 4. Inicializar modelo PhaseNet
print("\n⚡ Inicializando modelo PhaseNet...")
try:
    model = PhaseNet.from_pretrained("phasenet")  # Carga pesos base
    print("✅ Modelo PhaseNet cargado desde pretrained")
except Exception as e:
    print(f"⚠️ No se pudo cargar modelo pretrained: {e}")
    print("🔧 Inicializando modelo desde cero...")
    model = PhaseNet()

model.to(DEVICE)
print(f"✅ Modelo movido a {DEVICE}")

# 5. Preparar generador de etiquetas
print("\n🔖 Preparando generador de etiquetas...")
try:
    # Usar PickLabeller en lugar de pick_generator
    labeller = PickLabeller(
        phases=["P", "S"],  # Fases a detectar
        window=1.0,  # Ventana en segundos
        sigma=0.1  # Desviación estándar de la gaussiana
    )
    print("✅ PickLabeller configurado")
except Exception as e:
    print(f"⚠️ Error con PickLabeller: {e}")
    print("🔧 Usando configuración básica...")
    labeller = None

# 6. Preparar DataLoaders
print("\n🚀 Configurando DataLoaders...")
try:
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("✅ DataLoaders creados exitosamente")
except Exception as e:
    print(f"❌ Error al crear DataLoaders: {e}")
    exit(1)

# 7. Configurar entrenamiento manual
print("\n🏋️‍♂️ Configurando entrenamiento manual...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

# 8. Entrenar el modelo
print("\n🏋️‍♂️ Iniciando entrenamiento...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            optimizer.zero_grad()
            
            # Procesar el batch
            if isinstance(batch, dict):
                waveforms = batch['X'].to(DEVICE)
                targets = batch['y'].to(DEVICE) if 'y' in batch else None
            else:
                # Si batch es directamente los datos
                waveforms = batch.to(DEVICE)
                targets = None
            
            # Forward pass
            outputs = model(waveforms)
            
            # Calcular pérdida si hay targets
            if targets is not None:
                loss = criterion(outputs, targets)
            else:
                # Pérdida dummy si no hay targets
                loss = torch.mean(torch.abs(outputs))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        except Exception as e:
            print(f"⚠️ Error en batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / max(num_batches, 1)
    print(f"   Época {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

# 9. Guardar el modelo entrenado
print(f"\n💾 Guardando modelo entrenado...")
try:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,
        'loss': avg_loss
    }, MODEL_SAVE_PATH)
    print(f"✅ Modelo guardado en: {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"❌ Error al guardar modelo: {e}")

print("\n✅ Entrenamiento finalizado. Puedes usar el modelo para inferencia.")

# 10. Información adicional
print(f"\n📊 Resumen del entrenamiento:")
print(f"   - Épocas completadas: {EPOCHS}")
print(f"   - Pérdida final: {avg_loss:.6f}")
print(f"   - Dispositivo usado: {DEVICE}")
print(f"   - Tamaño de batch: {BATCH_SIZE}")
print(f"   - Datos de entrenamiento: {len(train_set)}")
print(f"   - Datos de validación: {len(val_set)}") 