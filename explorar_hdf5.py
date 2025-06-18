#!/usr/bin/env python3
"""
Explorador detallado del archivo HDF5
"""

import h5py
import numpy as np

def explore_hdf5_structure(filename):
    """Explora la estructura completa del archivo HDF5"""
    print(f"🔍 Explorando estructura de {filename}")
    print("="*50)
    
    with h5py.File(filename, 'r') as f:
        print(f"📁 Archivo: {filename}")
        print(f"📊 Tamaño: {f.id.get_filesize()} bytes")
        
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 Dataset: {name}")
                print(f"{indent}   Forma: {obj.shape}")
                print(f"{indent}   Tipo: {obj.dtype}")
                print(f"{indent}   Tamaño: {obj.size}")
                print(f"{indent}   Atributos: {dict(obj.attrs)}")
                
                # Mostrar algunos valores si el dataset no es muy grande
                if obj.size <= 20:
                    print(f"{indent}   Valores: {obj[:]}")
                elif len(obj.shape) == 1 and obj.size <= 100:
                    print(f"{indent}   Primeros 10 valores: {obj[:10]}")
                else:
                    print(f"{indent}   Primeros valores: {obj.flatten()[:5]}")
                    
            elif isinstance(obj, h5py.Group):
                print(f"{indent}📁 Grupo: {name}")
                print(f"{indent}   Elementos: {len(obj.keys())}")
                print(f"{indent}   Atributos: {dict(obj.attrs)}")
        
        f.visititems(print_structure)

if __name__ == "__main__":
    explore_hdf5_structure("seisbench_data/waveforms.hdf5") 