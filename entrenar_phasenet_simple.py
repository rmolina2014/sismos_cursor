#!/usr/bin/env python3
"""
Entrenamiento simplificado de PhaseNet con datos de un solo canal
===============================================================

Este script entrena un modelo PhaseNet usando un enfoque más directo
para datos de un solo canal (BHZ).

"""

import torch
import torch.nn as nn
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# Configuración
DATA_DIR = "seisbench_data"
MODEL_SAVE_PATH = "phasenet_trained.pt"
BATCH_SIZE = 1
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔧 Usando dispositivo: {DEVICE}")

class SimpleWaveformDataset(Dataset):
    """Dataset simple para formas de onda de un solo canal"""
    
    def __init__(self, metadata_path, waveform_path):
        self.metadata = pd.read_csv(metadata_path)
        self.waveform_path = waveform_path
        
        # Cargar todas las formas de onda en memoria
        self.waveforms = {}
        with h5py.File(waveform_path, 'r') as f:
            for trace_name in self.metadata['trace_name']:
                if trace_name in f:
                    self.waveforms[trace_name] = f[trace_name][:]
        
        print(f"✅ Cargadas {len(self.waveforms)} formas de onda")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        trace_name = row['trace_name']
        
        # Obtener forma de onda
        waveform = self.waveforms[trace_name]
        
        # Normalizar
        waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-8)
        
        # Convertir a tensor y agregar dimensión de canal
        waveform = torch.FloatTensor(waveform).unsqueeze(0)  # [1, samples]
        
        # Crear etiquetas dummy (para entrenamiento básico)
        # En un caso real, necesitarías etiquetas de picks P y S
        labels = torch.zeros(2, len(waveform[0]))  # [2, samples] para P y S
        
        return {
            'X': waveform,
            'y': labels,
            'trace_name': trace_name
        }

def create_simple_phasenet():
    """Crear un modelo PhaseNet simple para un canal"""
    from seisbench.models import PhaseNet
    
    # Crear modelo PhaseNet
    model = PhaseNet(
        in_channels=1,  # Un solo canal
        classes=2,      # P y S
        phases=['P', 'S']
    )
    
    return model

def main():
    print("🚀 Iniciando entrenamiento simplificado de PhaseNet")
    
    # 1. Cargar datos
    print("\n📊 Cargando datos...")
    dataset = SimpleWaveformDataset(
        os.path.join(DATA_DIR, "metadata.csv"),
        os.path.join(DATA_DIR, "waveforms.hdf5")
    )
    
    # 2. Dividir en train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"   Datos de entrenamiento: {len(train_dataset)}")
    print(f"   Datos de validación: {len(val_dataset)}")
    
    # 3. Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Crear modelo
    print("\n⚡ Creando modelo PhaseNet...")
    model = create_simple_phasenet()
    model.to(DEVICE)
    print(f"✅ Modelo creado y movido a {DEVICE}")
    
    # 5. Configurar entrenamiento
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # 6. Entrenar
    print("\n🏋️‍♂️ Iniciando entrenamiento...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                optimizer.zero_grad()
                
                # Obtener datos
                waveforms = batch['X'].to(DEVICE)
                targets = batch['y'].to(DEVICE)
                
                # Forward pass
                outputs = model(waveforms)
                
                # Calcular pérdida
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"   Batch {batch_idx}, Loss: {loss.item():.6f}")
                
            except Exception as e:
                print(f"⚠️ Error en batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"   Época {epoch+1}/{EPOCHS}, Loss promedio: {avg_loss:.6f}")
    
    # 7. Guardar modelo
    print(f"\n💾 Guardando modelo...")
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': EPOCHS,
            'loss': avg_loss,
            'model_config': {
                'in_channels': 1,
                'classes': 2,
                'phases': ['P', 'S']
            }
        }, MODEL_SAVE_PATH)
        print(f"✅ Modelo guardado en: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"❌ Error al guardar: {e}")
    
    print("\n✅ Entrenamiento completado!")
    print(f"\n📊 Resumen:")
    print(f"   - Épocas: {EPOCHS}")
    print(f"   - Pérdida final: {avg_loss:.6f}")
    print(f"   - Dispositivo: {DEVICE}")
    print(f"   - Modelo: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main() 