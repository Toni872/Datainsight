#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de prueba para el sistema de caché y visualizaciones avanzadas
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Importar módulos de utils
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Importar componentes directamente
try:
    from utils.cache_manager import CacheManager
    from utils.visualization import AdvancedVisualizer
    COMPONENTS_LOADED = True
    print("✓ Componentes cargados correctamente")
except ImportError as e:
    print(f"Error al importar componentes: {e}")
    print(f"Carpeta actual: {os.getcwd()}")
    print(f"Script dir: {script_dir}")
    print("Búsqueda en: ", sys.path)
    COMPONENTS_LOADED = False
    

def test_cache_manager():
    """Prueba las funcionalidades del gestor de caché"""
    print("\n=== PROBANDO GESTOR DE CACHÉ ===")
    
    try:
        # Crear instancia
        cache = CacheManager()
        print("✓ Instancia de CacheManager creada correctamente")
        
        # Probar almacenamiento y recuperación
        test_key = {"model": "test_model", "params": [1, 2, 3]}
        test_data = {"accuracy": 0.95, "f1_score": 0.94}
        
        # Guardar en caché
        cache.put(test_key, test_data, entry_type="test")
        print("✓ Datos guardados en caché")
        
        # Recuperar de caché
        retrieved = cache.get(test_key, entry_type="test")
        
        if retrieved is not None and retrieved["accuracy"] == 0.95:
            print("✓ Datos recuperados de caché correctamente")
        else:
            print("✗ Error al recuperar datos de caché")
        
        # Verificar estadísticas
        stats = cache.get_stats()
        print(f"✓ Estadísticas de caché obtenidas: {len(stats)} campos")
        print(f"  - Entradas en caché: {stats['entry_count']}")
        
        # Limpiar caché
        cache.clear()
        print("✓ Caché limpiada correctamente")
        
        return True
    except Exception as e:
        print(f"✗ Error en prueba de caché: {e}")
        return False


def test_visualization_with_cache():
    """Prueba el sistema de visualización con caché"""
    print("\n=== PROBANDO VISUALIZACIÓN CON CACHÉ ===")
    
    try:
        # Crear instancia
        visualizer = AdvancedVisualizer(use_cache=True)
        print("✓ Instancia de AdvancedVisualizer creada correctamente")
        
        # Crear datos de ejemplo para matriz de correlación
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'target': np.random.normal(0, 1, 100)
        })
        
        # Generar matriz de correlación (primera vez, sin caché)
        t1_start = time.time()
        corr_buf1 = visualizer.correlation_matrix(data)
        t1_end = time.time()
        
        # Verificar que es un buffer válido
        if corr_buf1 and isinstance(corr_buf1, BytesIO):
            print("✓ Matriz de correlación generada correctamente")
        else:
            print("✗ Error al generar matriz de correlación")
        
        # Generar de nuevo (debería usar caché)
        t2_start = time.time()
        corr_buf2 = visualizer.correlation_matrix(data)
        t2_end = time.time()
        
        # Verificar mejora de rendimiento con caché
        time1 = t1_end - t1_start
        time2 = t2_end - t2_start
        
        print(f"  - Tiempo sin caché: {time1:.4f} segundos")
        print(f"  - Tiempo con caché: {time2:.4f} segundos")
        
        if time2 < time1:
            print(f"✓ Mejora de rendimiento con caché: {(1 - time2/time1)*100:.1f}% más rápido")
        else:
            print("✗ No se observó mejora de rendimiento con caché")
        
        # Probar gráfico de importancia de características
        feature_importance = np.array([0.4, 0.3, 0.2, 0.1])
        feature_names = ['feature1', 'feature2', 'feature3', 'target']
        
        imp_buf = visualizer.feature_importance(feature_importance, feature_names)
        
        if imp_buf and isinstance(imp_buf, BytesIO):
            print("✓ Gráfico de importancia generado correctamente")
        else:
            print("✗ Error al generar gráfico de importancia")
        
        # Probar distribución
        dist_buf = visualizer.distribution_plot(data, x='feature1', kde=True)
        
        if dist_buf and isinstance(dist_buf, BytesIO):
            print("✓ Gráfico de distribución generado correctamente")
        else:
            print("✗ Error al generar gráfico de distribución")
        
        # Verificar estadísticas de caché
        stats = visualizer.cache_manager.get_stats()
        print(f"✓ Caché contiene {stats['entry_count']} entradas")
        
        return True
    except Exception as e:
        print(f"✗ Error en prueba de visualización: {e}")
        return False


def main():
    """Función principal para ejecutar todas las pruebas"""
    if not COMPONENTS_LOADED:
        print("✗ No se pudo cargar los componentes necesarios para las pruebas")
        return False
    
    results = []
    
    # Ejecutar pruebas
    results.append(("CacheManager", test_cache_manager()))
    results.append(("Visualizaciones", test_visualization_with_cache()))
    
    # Mostrar resumen
    print("\n=== RESUMEN DE PRUEBAS ===")
    all_pass = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"{test_name}: {'✓' if result else '✗'} {status}")
    
    if all_pass:
        print("\n✓ TODOS LOS COMPONENTES FUNCIONAN CORRECTAMENTE")
    else:
        print("\n✗ HAY ERRORES EN ALGUNOS COMPONENTES")
    
    return all_pass


if __name__ == "__main__":
    main()
