#!/usr/bin/env python3
"""
Ejemplo de uso del modelo PhaseNet entrenado
===========================================

Este script muestra cómo cargar el modelo entrenado y usarlo
para hacer predicciones en nuevas formas de onda.

"""

import torch
import numpy as np
import h5py
from seisbench.models import PhaseNet
import matplotlib.pyplot as plt

def cargar_modelo(ruta_modelo="phasenet_trained.pt"):
    """Cargar el modelo entrenado"""
    print("🔍 Cargando modelo entrenado...")
    
    # Cargar checkpoint
    checkpoint = torch.load(ruta_modelo, map_location='cpu')
    
    # Crear modelo con la misma configuración
    model = PhaseNet(
        in_channels=1,
        classes=2,
        phases=['P', 'S']
    )
    
    # Cargar pesos entrenados
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Modelo cargado exitosamente")
    print(f"   Configuración: {checkpoint['model_config']}")
    print(f"   Épocas entrenadas: {checkpoint['epoch']}")
    print(f"   Pérdida final: {checkpoint['loss']:.6f}")
    
    return model

def predecir_forma_onda(modelo, forma_onda, tasa_muestreo=100.0):
    """Hacer predicción en una forma de onda"""
    print("🔮 Realizando predicción...")
    
    # Normalizar forma de onda
    forma_onda_norm = (forma_onda - np.mean(forma_onda)) / (np.std(forma_onda) + 1e-8)
    
    # Convertir a tensor
    tensor_onda = torch.FloatTensor(forma_onda_norm).unsqueeze(0).unsqueeze(0)
    # [1, 1, samples] - [batch, channel, samples]
    
    # Predicción
    with torch.no_grad():
        predicciones = modelo(tensor_onda)
    
    # Convertir a numpy
    predicciones = predicciones.squeeze(0).numpy()
    # [2, samples] - [phases, samples]
    
    print("✅ Predicción completada")
    print(f"   Forma de entrada: {tensor_onda.shape}")
    print(f"   Forma de salida: {predicciones.shape}")
    
    return predicciones

def visualizar_prediccion(forma_onda, predicciones, tasa_muestreo=100.0):
    """Visualizar forma de onda y predicciones"""
    tiempo = np.arange(len(forma_onda)) / tasa_muestreo
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Forma de onda original
    ax1.plot(tiempo, forma_onda, 'b-', linewidth=0.5)
    ax1.set_title('Forma de Onda Original')
    ax1.set_ylabel('Amplitud')
    ax1.grid(True, alpha=0.3)
    
    # Predicciones
    ax2.plot(tiempo, predicciones[0], 'r-', label='Fase P', linewidth=0.5)
    ax2.plot(tiempo, predicciones[1], 'g-', label='Fase S', linewidth=0.5)
    ax2.set_title('Predicciones PhaseNet')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Probabilidad')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediccion_phasenet.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Gráfico guardado como 'prediccion_phasenet.png'")

def encontrar_picks(predicciones, umbral=0.5, tasa_muestreo=100.0):
    """Encontrar picks basados en umbral de probabilidad"""
    print("🎯 Buscando picks...")
    
    picks = {}
    
    for i, fase in enumerate(['P', 'S']):
        # Encontrar picos por encima del umbral
        picos = []
        prediccion = predicciones[i]
        
        for j in range(1, len(prediccion) - 1):
            if prediccion[j] > umbral and prediccion[j] > prediccion[j-1] and prediccion[j] > prediccion[j+1]:
                tiempo_pick = j / tasa_muestreo
                probabilidad = prediccion[j]
                picos.append((tiempo_pick, probabilidad))
        
        picks[fase] = picos
        print(f"   {fase}: {len(picos)} picks encontrados")
    
    return picks

def main():
    """Función principal"""
    print("🚀 Ejemplo de uso del modelo PhaseNet entrenado")
    print("="*50)
    
    # 1. Cargar modelo
    modelo = cargar_modelo()
    
    # 2. Cargar datos de ejemplo
    print("\n📊 Cargando datos de ejemplo...")
    with h5py.File("seisbench_data/waveforms.hdf5", 'r') as f:
        # Usar la primera traza como ejemplo
        trazas = list(f.keys())
        if trazas:
            traza_ejemplo = trazas[0]
            forma_onda = f[traza_ejemplo][:]
            print(f"✅ Cargada traza: {traza_ejemplo}")
            print(f"   Muestras: {len(forma_onda)}")
            print(f"   Duración: {len(forma_onda)/100.0:.1f} segundos")
        else:
            print("❌ No se encontraron trazas")
            return
    
    # 3. Hacer predicción
    predicciones = predecir_forma_onda(modelo, forma_onda)
    
    # 4. Encontrar picks
    picks = encontrar_picks(predicciones)
    
    # 5. Mostrar resultados
    print("\n📋 Resultados:")
    for fase, picks_fase in picks.items():
        print(f"   {fase}:")
        for tiempo, prob in picks_fase:
            print(f"     Tiempo: {tiempo:.2f}s, Probabilidad: {prob:.3f}")
    
    # 6. Visualizar (opcional - requiere matplotlib)
    try:
        print("\n📊 Generando visualización...")
        visualizar_prediccion(forma_onda, predicciones)
    except ImportError:
        print("⚠️ matplotlib no disponible, saltando visualización")
    
    print("\n✅ Ejemplo completado!")

if __name__ == "__main__":
    main() 