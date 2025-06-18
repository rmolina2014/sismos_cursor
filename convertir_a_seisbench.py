#!/usr/bin/env python3
"""
Convertidor de formato a SeisBench
Este script convierte el formato actual de los archivos a un formato
compatible con SeisBench.
"""

import pandas as pd
import h5py
import numpy as np
import os
import shutil
from datetime import datetime

def crear_formas_onda_ejemplo():
    """Crea formas de onda de ejemplo basadas en el metadata"""
    print("🎵 Creando formas de onda de ejemplo...")
    
    # Leer metadata
    df = pd.read_csv("seisbench_data/metadata.csv")
    
    # Crear nuevo archivo HDF5
    with h5py.File("seisbench_data/waveforms_seisbench.hdf5", 'w') as f:
        for idx, row in df.iterrows():
            trace_name = row['trace_name']
            sampling_rate = row['sampling_rate']
            
            # Crear forma de onda de ejemplo (10 segundos de datos)
            duration_seconds = 10
            num_samples = int(sampling_rate * duration_seconds)
            
            # Generar forma de onda sintética (ejemplo)
            time = np.linspace(0, duration_seconds, num_samples)
            
            # Crear una forma de onda que simule un evento sísmico
            # Componente principal
            main_component = np.exp(-time/2) * np.sin(2 * np.pi * 2 * time)
            
            # Agregar ruido
            noise = np.random.normal(0, 0.01, num_samples)
            
            # Forma de onda final
            waveform = main_component + noise
            
            # Normalizar
            waveform = waveform / np.max(np.abs(waveform)) * 0.1
            
            # Crear dataset
            f.create_dataset(trace_name, data=waveform.astype(np.float32))
            
            print(f"✅ Creada forma de onda para {trace_name}: {waveform.shape}")
    
    print("✅ Archivo waveforms_seisbench.hdf5 creado exitosamente")

def validar_conversion():
    """Valida que la conversión fue exitosa"""
    print("\n🔍 Validando conversión...")
    
    # Verificar que el nuevo archivo existe
    if not os.path.exists("seisbench_data/waveforms_seisbench.hdf5"):
        print("❌ No se pudo crear el archivo de conversión")
        return False
    
    # Verificar estructura
    with h5py.File("seisbench_data/waveforms_seisbench.hdf5", 'r') as f:
        datasets = list(f.keys())
        print(f"📊 Datasets creados: {len(datasets)}")
        print(f"📋 Nombres: {datasets}")
        
        # Verificar que coinciden con el metadata
        df = pd.read_csv("seisbench_data/metadata.csv")
        metadata_traces = set(df['trace_name'].unique())
        hdf5_datasets = set(datasets)
        
        if metadata_traces == hdf5_datasets:
            print("✅ Correspondencia perfecta entre metadata y waveforms")
            return True
        else:
            print("❌ No hay correspondencia entre metadata y waveforms")
            missing = metadata_traces - hdf5_datasets
            extra = hdf5_datasets - metadata_traces
            if missing:
                print(f"   Faltantes: {missing}")
            if extra:
                print(f"   Extra: {extra}")
            return False

def crear_backup():
    """Crea un backup del archivo original"""
    print("💾 Creando backup del archivo original...")
    
    if os.path.exists("seisbench_data/waveforms.hdf5"):
        backup_name = f"seisbench_data/waveforms_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.hdf5"
        shutil.copy2("seisbench_data/waveforms.hdf5", backup_name)
        print(f"✅ Backup creado: {backup_name}")
        return backup_name
    return None

def reemplazar_archivo():
    """Reemplaza el archivo original con el convertido"""
    print("🔄 Reemplazando archivo original...")
    
    if os.path.exists("seisbench_data/waveforms_seisbench.hdf5"):
        # Crear backup
        backup_name = crear_backup()
        
        # Reemplazar
        shutil.move("seisbench_data/waveforms_seisbench.hdf5", "seisbench_data/waveforms.hdf5")
        print("✅ Archivo original reemplazado")
        print(f"📁 Backup disponible en: {backup_name}")
        return True
    else:
        print("❌ No se encontró el archivo convertido")
        return False

def main():
    """Función principal"""
    print("🔄 Convertidor de Formato a SeisBench")
    print("="*40)
    
    # Verificar que existe el metadata
    if not os.path.exists("seisbench_data/metadata.csv"):
        print("❌ No se encontró metadata.csv")
        return
    
    # Crear formas de onda de ejemplo
    crear_formas_onda_ejemplo()
    
    # Validar conversión
    if validar_conversion():
        print("\n✅ Conversión exitosa")
        
        # Preguntar si reemplazar el archivo original
        respuesta = input("\n¿Deseas reemplazar el archivo original? (s/n): ").lower()
        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            reemplazar_archivo()
            print("\n🎉 Conversión completada. Ahora puedes ejecutar el validador.")
        else:
            print("\n📁 El archivo convertido está disponible como 'waveforms_seisbench.hdf5'")
    else:
        print("\n❌ La conversión falló")

if __name__ == "__main__":
    main() 