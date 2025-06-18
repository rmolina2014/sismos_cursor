#!/usr/bin/env python3
"""
Validador de archivos SeisBench
Este programa valida si los archivos en la carpeta seisbench_data
cumplen con el formato estándar de SeisBench.
"""

import pandas as pd
import h5py
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SeisBenchValidator:
    def __init__(self, data_dir="seisbench_data"):
        self.data_dir = data_dir
        self.metadata_file = os.path.join(data_dir, "metadata.csv")
        self.waveforms_file = os.path.join(data_dir, "waveforms.hdf5")
        
        # Columnas requeridas en metadata.csv según SeisBench
        self.required_columns = [
            'trace_name', 'split', 'station', 'channel', 
            'start_time', 'end_time', 'sampling_rate'
        ]
        
        # Columnas opcionales pero comunes
        self.optional_columns = [
            'latitude', 'longitude', 'magnitude', 'depth', 
            'distance', 'source_depth', 'source_latitude', 
            'source_longitude'
        ]
        
        self.validation_results = {}
    
    def validate_file_existence(self):
        """Valida que los archivos requeridos existan"""
        print("🔍 Validando existencia de archivos...")
        
        results = {}
        
        # Verificar metadata.csv
        if os.path.exists(self.metadata_file):
            print(f"✅ metadata.csv encontrado: {self.metadata_file}")
            results['metadata_exists'] = True
        else:
            print(f"❌ metadata.csv no encontrado: {self.metadata_file}")
            results['metadata_exists'] = False
        
        # Verificar waveforms.hdf5
        if os.path.exists(self.waveforms_file):
            print(f"✅ waveforms.hdf5 encontrado: {self.waveforms_file}")
            results['waveforms_exists'] = True
        else:
            print(f"❌ waveforms.hdf5 no encontrado: {self.waveforms_file}")
            results['waveforms_exists'] = False
        
        self.validation_results['file_existence'] = results
        return results
    
    def validate_metadata_structure(self):
        """Valida la estructura del archivo metadata.csv"""
        print("\n📊 Validando estructura de metadata.csv...")
        
        if not os.path.exists(self.metadata_file):
            print("❌ metadata.csv no existe, saltando validación")
            return {'valid': False, 'error': 'Archivo no existe'}
        
        try:
            # Leer el archivo CSV
            df = pd.read_csv(self.metadata_file)
            
            results = {
                'valid': True,
                'total_records': len(df),
                'columns_found': list(df.columns),
                'missing_required': [],
                'data_types': {},
                'sample_data': {}
            }
            
            # Verificar columnas requeridas
            for col in self.required_columns:
                if col in df.columns:
                    print(f"✅ Columna requerida '{col}' encontrada")
                    results['data_types'][col] = str(df[col].dtype)
                    results['sample_data'][col] = df[col].iloc[0] if len(df) > 0 else None
                else:
                    print(f"❌ Columna requerida '{col}' NO encontrada")
                    results['missing_required'].append(col)
                    results['valid'] = False
            
            # Verificar columnas opcionales
            optional_found = []
            for col in self.optional_columns:
                if col in df.columns:
                    optional_found.append(col)
                    print(f"✅ Columna opcional '{col}' encontrada")
            
            results['optional_columns_found'] = optional_found
            
            # Mostrar información básica
            print(f"📈 Total de registros: {len(df)}")
            print(f"📋 Columnas encontradas: {len(df.columns)}")
            print(f"🔧 Columnas requeridas: {len(self.required_columns)}")
            print(f"➕ Columnas opcionales encontradas: {len(optional_found)}")
            
            # Mostrar primeras filas
            print("\n📋 Primeras filas del metadata:")
            print(df.head().to_string())
            
            self.validation_results['metadata_structure'] = results
            return results
            
        except Exception as e:
            error_msg = f"Error al leer metadata.csv: {str(e)}"
            print(f"❌ {error_msg}")
            return {'valid': False, 'error': error_msg}
    
    def validate_metadata_content(self):
        """Valida el contenido y tipos de datos del metadata"""
        print("\n🔍 Validando contenido de metadata.csv...")
        
        if not os.path.exists(self.metadata_file):
            return {'valid': False, 'error': 'Archivo no existe'}
        
        try:
            df = pd.read_csv(self.metadata_file)
            results = {'valid': True, 'issues': []}
            
            # Validar tipos de datos
            if 'sampling_rate' in df.columns:
                if not pd.to_numeric(df['sampling_rate'], errors='coerce').notna().all():
                    results['issues'].append("sampling_rate contiene valores no numéricos")
                    results['valid'] = False
                else:
                    print(f"✅ sampling_rate: valores válidos (rango: {df['sampling_rate'].min()}-{df['sampling_rate'].max()} Hz)")
            
            # Validar fechas
            for time_col in ['start_time', 'end_time']:
                if time_col in df.columns:
                    try:
                        pd.to_datetime(df[time_col])
                        print(f"✅ {time_col}: formato de fecha válido")
                    except:
                        results['issues'].append(f"{time_col} contiene fechas inválidas")
                        results['valid'] = False
            
            # Validar coordenadas si existen
            for coord_col in ['latitude', 'longitude']:
                if coord_col in df.columns:
                    if not pd.to_numeric(df[coord_col], errors='coerce').notna().all():
                        results['issues'].append(f"{coord_col} contiene valores no numéricos")
                        results['valid'] = False
                    else:
                        print(f"✅ {coord_col}: valores válidos")
            
            # Validar magnitud si existe
            if 'magnitude' in df.columns:
                if not pd.to_numeric(df['magnitude'], errors='coerce').notna().all():
                    results['issues'].append("magnitude contiene valores no numéricos")
                    results['valid'] = False
                else:
                    print(f"✅ magnitude: valores válidos (rango: {df['magnitude'].min()}-{df['magnitude'].max()})")
            
            # Verificar consistencia de datos
            if len(df) > 0:
                print(f"✅ Total de registros válidos: {len(df)}")
                
                # Verificar splits
                if 'split' in df.columns:
                    splits = df['split'].unique()
                    print(f"✅ Splits encontrados: {list(splits)}")
                
                # Verificar estaciones
                if 'station' in df.columns:
                    stations = df['station'].unique()
                    print(f"✅ Estaciones encontradas: {list(stations)}")
                
                # Verificar canales
                if 'channel' in df.columns:
                    channels = df['channel'].unique()
                    print(f"✅ Canales encontrados: {list(channels)}")
            
            if results['issues']:
                print("⚠️ Problemas encontrados:")
                for issue in results['issues']:
                    print(f"   - {issue}")
            
            self.validation_results['metadata_content'] = results
            return results
            
        except Exception as e:
            error_msg = f"Error al validar contenido: {str(e)}"
            print(f"❌ {error_msg}")
            return {'valid': False, 'error': error_msg}
    
    def validate_waveforms_structure(self):
        """Valida la estructura del archivo waveforms.hdf5"""
        print("\n📊 Validando estructura de waveforms.hdf5...")
        
        if not os.path.exists(self.waveforms_file):
            print("❌ waveforms.hdf5 no existe, saltando validación")
            return {'valid': False, 'error': 'Archivo no existe'}
        
        try:
            with h5py.File(self.waveforms_file, 'r') as f:
                results = {
                    'valid': True,
                    'groups': list(f.keys()),
                    'datasets': {},
                    'attributes': {}
                }
                
                print(f"📁 Grupos encontrados: {list(f.keys())}")
                
                # Explorar estructura
                def explore_group(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        results['datasets'][name] = {
                            'shape': obj.shape,
                            'dtype': str(obj.dtype),
                            'size': obj.size
                        }
                        print(f"📊 Dataset '{name}': forma {obj.shape}, tipo {obj.dtype}")
                    elif isinstance(obj, h5py.Group):
                        print(f"📁 Grupo '{name}': {len(obj.keys())} elementos")
                
                f.visititems(explore_group)
                
                # Verificar que hay datasets con formas de onda
                waveform_datasets = [k for k, v in results['datasets'].items() 
                                   if len(v['shape']) >= 1]
                
                if waveform_datasets:
                    print(f"✅ Datasets de formas de onda encontrados: {len(waveform_datasets)}")
                    results['waveform_datasets'] = waveform_datasets
                else:
                    print("⚠️ No se encontraron datasets de formas de onda")
                    results['valid'] = False
                
                self.validation_results['waveforms_structure'] = results
                return results
                
        except Exception as e:
            error_msg = f"Error al leer waveforms.hdf5: {str(e)}"
            print(f"❌ {error_msg}")
            return {'valid': False, 'error': error_msg}
    
    def validate_waveforms_content(self):
        """Valida el contenido de las formas de onda"""
        print("\n🔍 Validando contenido de waveforms.hdf5...")
        
        if not os.path.exists(self.waveforms_file):
            return {'valid': False, 'error': 'Archivo no existe'}
        
        try:
            with h5py.File(self.waveforms_file, 'r') as f:
                results = {'valid': True, 'issues': [], 'statistics': {}}
                
                # Analizar cada dataset
                for dataset_name, dataset in f.items():
                    if isinstance(dataset, h5py.Dataset):
                        print(f"\n📊 Analizando dataset: {dataset_name}")
                        
                        # Obtener datos
                        data = dataset[:]
                        
                        # Estadísticas básicas
                        stats = {
                            'shape': data.shape,
                            'dtype': str(data.dtype),
                            'min': float(np.min(data)),
                            'max': float(np.max(data)),
                            'mean': float(np.mean(data)),
                            'std': float(np.std(data)),
                            'size': data.size
                        }
                        
                        results['statistics'][dataset_name] = stats
                        
                        print(f"   Forma: {data.shape}")
                        print(f"   Tipo: {data.dtype}")
                        print(f"   Rango: [{stats['min']:.6f}, {stats['max']:.6f}]")
                        print(f"   Media: {stats['mean']:.6f}")
                        print(f"   Desv. Est.: {stats['std']:.6f}")
                        
                        # Validaciones específicas
                        if np.isnan(data).any():
                            results['issues'].append(f"Dataset {dataset_name} contiene valores NaN")
                            results['valid'] = False
                        
                        if np.isinf(data).any():
                            results['issues'].append(f"Dataset {dataset_name} contiene valores infinitos")
                            results['valid'] = False
                        
                        # Verificar que los datos parecen ser formas de onda sísmicas
                        if len(data.shape) == 1:
                            # Forma de onda unidimensional
                            if data.size < 100:
                                results['issues'].append(f"Dataset {dataset_name} parece muy corto para ser una forma de onda")
                            print(f"   ✅ Forma de onda 1D válida")
                        elif len(data.shape) == 2:
                            # Múltiples formas de onda
                            print(f"   ✅ {data.shape[0]} formas de onda, cada una con {data.shape[1]} muestras")
                        else:
                            print(f"   ⚠️ Forma inesperada: {data.shape}")
                
                if results['issues']:
                    print("\n⚠️ Problemas encontrados:")
                    for issue in results['issues']:
                        print(f"   - {issue}")
                else:
                    print("\n✅ No se encontraron problemas en los datos")
                
                self.validation_results['waveforms_content'] = results
                return results
                
        except Exception as e:
            error_msg = f"Error al validar contenido de waveforms: {str(e)}"
            print(f"❌ {error_msg}")
            return {'valid': False, 'error': error_msg}
    
    def validate_consistency(self):
        """Valida la consistencia entre metadata y waveforms"""
        print("\n🔗 Validando consistencia entre metadata y waveforms...")
        
        if not (os.path.exists(self.metadata_file) and os.path.exists(self.waveforms_file)):
            print("❌ No se pueden validar ambos archivos")
            return {'valid': False, 'error': 'Archivos faltantes'}
        
        try:
            # Leer metadata
            df = pd.read_csv(self.metadata_file)
            
            # Leer waveforms
            with h5py.File(self.waveforms_file, 'r') as f:
                results = {'valid': True, 'issues': []}
                
                # Verificar que los nombres de trazas en metadata existen en waveforms
                if 'trace_name' in df.columns:
                    metadata_traces = set(df['trace_name'].unique())
                    hdf5_datasets = set(f.keys())
                    
                    print(f"📋 Trazas en metadata: {len(metadata_traces)}")
                    print(f"📊 Datasets en HDF5: {len(hdf5_datasets)}")
                    
                    # Verificar correspondencia
                    missing_in_hdf5 = metadata_traces - hdf5_datasets
                    extra_in_hdf5 = hdf5_datasets - metadata_traces
                    
                    if missing_in_hdf5:
                        results['issues'].append(f"Trazas en metadata pero no en HDF5: {len(missing_in_hdf5)}")
                        results['valid'] = False
                        print(f"❌ Trazas faltantes en HDF5: {list(missing_in_hdf5)[:5]}...")
                    
                    if extra_in_hdf5:
                        print(f"⚠️ Datasets extra en HDF5: {list(extra_in_hdf5)[:5]}...")
                    
                    if not missing_in_hdf5 and not extra_in_hdf5:
                        print("✅ Correspondencia perfecta entre metadata y waveforms")
                    elif not missing_in_hdf5:
                        print("✅ Todas las trazas del metadata están en waveforms")
                
                self.validation_results['consistency'] = results
                return results
                
        except Exception as e:
            error_msg = f"Error al validar consistencia: {str(e)}"
            print(f"❌ {error_msg}")
            return {'valid': False, 'error': error_msg}
    
    def generate_report(self):
        """Genera un reporte completo de validación"""
        print("\n" + "="*60)
        print("📋 REPORTE COMPLETO DE VALIDACIÓN SEISBENCH")
        print("="*60)
        
        # Resumen ejecutivo
        all_valid = True
        for test_name, result in self.validation_results.items():
            if isinstance(result, dict) and 'valid' in result:
                if not result['valid']:
                    all_valid = False
        
        print(f"\n🎯 RESULTADO GENERAL: {'✅ VÁLIDO' if all_valid else '❌ INVÁLIDO'}")
        
        # Detalles por prueba
        for test_name, result in self.validation_results.items():
            print(f"\n📊 {test_name.upper().replace('_', ' ')}:")
            if isinstance(result, dict):
                if 'valid' in result:
                    status = "✅ VÁLIDO" if result['valid'] else "❌ INVÁLIDO"
                    print(f"   Estado: {status}")
                
                if 'issues' in result and result['issues']:
                    print("   Problemas:")
                    for issue in result['issues']:
                        print(f"     - {issue}")
                
                if 'statistics' in result:
                    print("   Estadísticas:")
                    for dataset, stats in result['statistics'].items():
                        print(f"     {dataset}: {stats['shape']}, rango [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        if all_valid:
            print("   ✅ Los archivos cumplen con el formato SeisBench")
            print("   ✅ Los datos están listos para usar con SeisBench")
        else:
            print("   ⚠️ Se encontraron problemas que deben corregirse")
            print("   📖 Revisa los problemas listados arriba")
        
        return all_valid
    
    def run_full_validation(self):
        """Ejecuta todas las validaciones"""
        print("🚀 Iniciando validación completa de archivos SeisBench...")
        print(f"📁 Directorio: {self.data_dir}")
        
        # Ejecutar todas las validaciones
        self.validate_file_existence()
        self.validate_metadata_structure()
        self.validate_metadata_content()
        self.validate_waveforms_structure()
        self.validate_waveforms_content()
        self.validate_consistency()
        
        # Generar reporte
        return self.generate_report()

def main():
    """Función principal"""
    print("🔬 Validador de Archivos SeisBench")
    print("="*40)
    
    # Crear validador
    validator = SeisBenchValidator()
    
    # Ejecutar validación completa
    is_valid = validator.run_full_validation()
    
    # Retornar código de salida
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main() 