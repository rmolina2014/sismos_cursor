#!/usr/bin/env python3
"""
Explorador de SeisBench
"""

import seisbench
print(f"SeisBench versión: {seisbench.__version__}")

print("\n=== Módulos disponibles ===")
import seisbench.data
print("seisbench.data:", [x for x in dir(seisbench.data) if not x.startswith('_')])

import seisbench.models
print("seisbench.models:", [x for x in dir(seisbench.models) if not x.startswith('_')])

import seisbench.generate
print("seisbench.generate:", [x for x in dir(seisbench.generate) if not x.startswith('_')])

import seisbench.util
print("seisbench.util:", [x for x in dir(seisbench.util) if not x.startswith('_')])

print("\n=== Ejemplo de uso básico ===")
try:
    from seisbench.models import PhaseNet
    print("✅ PhaseNet importado correctamente")
    
    model = PhaseNet()
    print("✅ Modelo PhaseNet creado")
    
except Exception as e:
    print(f"❌ Error con PhaseNet: {e}")

print("\n=== Explorando WaveformDataset ===")
try:
    from seisbench.data import WaveformDataset
    print("✅ WaveformDataset importado")
    
    # Intentar crear un dataset básico
    import pandas as pd
    metadata = pd.read_csv("seisbench_data/metadata.csv")
    print(f"✅ Metadata cargada: {len(metadata)} registros")
    
except Exception as e:
    print(f"❌ Error con WaveformDataset: {e}") 