#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de caché para visualizaciones y resultados de modelos
Este módulo proporciona funciones para almacenar en caché y recuperar visualizaciones
y resultados de modelos para mejorar el rendimiento del sistema.
"""
import os
import sys
import time
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from io import BytesIO
import matplotlib.pyplot as plt

# Asegurar que podemos importar desde directorio padre
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CacheManager:
    """Gestor de caché para visualizaciones y resultados de modelos"""
    
    def __init__(self, cache_dir=None, max_age_hours=24):
        """
        Inicializa el gestor de caché
        
        Args:
            cache_dir (str): Directorio donde almacenar los archivos de caché
                           Si es None, se usa directorio predeterminado
            max_age_hours (int): Tiempo máximo en horas que se mantiene la caché
        """
        if cache_dir is None:
            # Usar directorio predeterminado relativo al directorio de ejecución
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        'cache', 'visualizations')
        else:
            self.cache_dir = cache_dir
            
        # Crear directorio de caché si no existe
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Configuración
        self.max_age = timedelta(hours=max_age_hours)
        self.metadata_file = os.path.join(self.cache_dir, 'cache_metadata.json')
        
        # Cargar metadatos existentes o crear nuevos
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {'entries': {}}
        else:
            self.metadata = {'entries': {}}
            
        # Limpiar caché antigua al iniciar
        self._clean_old_cache()
    
    def _generate_key(self, params):
        """
        Genera una clave única para un conjunto de parámetros
        
        Args:
            params (dict): Parámetros de la visualización o modelo
            
        Returns:
            str: Clave hash única
        """
        # Convertir parámetros a string JSON ordenado para consistencia
        param_str = json.dumps(params, sort_keys=True)
        
        # Generar hash MD5 como clave
        hash_key = hashlib.md5(param_str.encode()).hexdigest()
        
        return hash_key
    
    def _clean_old_cache(self):
        """Elimina entradas de caché que superan el tiempo máximo"""
        now = datetime.now()
        entries_to_remove = []
        
        # Identificar entradas antiguas
        for key, entry in self.metadata['entries'].items():
            created = datetime.fromisoformat(entry['created'])
            if now - created > self.max_age:
                entries_to_remove.append(key)
                
                # Eliminar archivo de cache
                cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                if os.path.exists(cache_file):
                    try:
                        os.remove(cache_file)
                    except:
                        pass
        
        # Eliminar entradas de los metadatos
        for key in entries_to_remove:
            del self.metadata['entries'][key]
            
        # Guardar metadatos actualizados
        self._save_metadata()
        
        # Retornar número de entradas eliminadas
        return len(entries_to_remove)
    
    def _save_metadata(self):
        """Guarda los metadatos de caché en disco"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def get(self, params, entry_type='visualization'):
        """
        Obtiene un elemento de la caché si existe
        
        Args:
            params (dict): Parámetros que identifican el elemento
            entry_type (str): Tipo de entrada (visualization, model, etc)
            
        Returns:
            object or None: Elemento si existe en caché, None en caso contrario
        """
        # Generar clave
        key = self._generate_key(params)
        
        # Verificar si existe en metadatos
        if key in self.metadata['entries']:
            entry = self.metadata['entries'][key]
            
            # Verificar tipo
            if entry['type'] != entry_type:
                return None
            
            # Verificar edad
            created = datetime.fromisoformat(entry['created'])
            if datetime.now() - created > self.max_age:
                # Eliminar entrada antigua
                del self.metadata['entries'][key]
                self._save_metadata()
                return None
            
            # Cargar desde archivo
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    # Si hay error al cargar, eliminar entrada
                    del self.metadata['entries'][key]
                    self._save_metadata()
        
        return None
    
    def put(self, params, data, entry_type='visualization'):
        """
        Guarda un elemento en la caché
        
        Args:
            params (dict): Parámetros que identifican el elemento
            data (object): Datos a almacenar en caché
            entry_type (str): Tipo de entrada (visualization, model, etc)
            
        Returns:
            str: Clave generada para la caché
        """
        # Generar clave
        key = self._generate_key(params)
        
        # Guardar datos en archivo
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Actualizar metadatos
        self.metadata['entries'][key] = {
            'type': entry_type,
            'created': datetime.now().isoformat(),
            'params': params
        }
        
        # Guardar metadatos
        self._save_metadata()
        
        return key
    
    def clear(self):
        """Limpia toda la caché"""
        # Eliminar archivos
        for key in self.metadata['entries']:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except:
                    pass
        
        # Reiniciar metadatos
        self.metadata = {'entries': {}}
        self._save_metadata()
    
    def get_stats(self):
        """
        Retorna estadísticas de uso de la caché
        
        Returns:
            dict: Estadísticas de uso de la caché
        """
        entry_count = len(self.metadata['entries'])
        entry_types = {}
        oldest_entry = None
        newest_entry = None
        total_size = 0
        
        # Calcular estadísticas
        if entry_count > 0:
            for key, entry in self.metadata['entries'].items():
                # Contar por tipo
                entry_type = entry['type']
                if entry_type in entry_types:
                    entry_types[entry_type] += 1
                else:
                    entry_types[entry_type] = 1
                
                # Determinar entradas más antigua y reciente
                created = datetime.fromisoformat(entry['created'])
                if oldest_entry is None or created < oldest_entry:
                    oldest_entry = created
                if newest_entry is None or created > newest_entry:
                    newest_entry = created
                
                # Calcular tamaño
                cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                if os.path.exists(cache_file):
                    total_size += os.path.getsize(cache_file)
        
        # Formatear resultados
        return {
            'entry_count': entry_count,
            'entry_types': entry_types,
            'oldest_entry': oldest_entry.isoformat() if oldest_entry else None,
            'newest_entry': newest_entry.isoformat() if newest_entry else None,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': self.cache_dir
        }


# Ejemplo de uso
if __name__ == "__main__":
    # Crear gestor de caché
    cache = CacheManager()
    
    # Ejemplo de almacenamiento de una visualización
    key_params = {
        'model': 'RandomForest',
        'dataset': 'iris',
        'visualization_type': 'confusion_matrix'
    }
    
    # Verificar si existe en caché
    result = cache.get(key_params)
    
    if result is None:
        print("No existe en caché, creando visualización...")
        # Crear visualización (normalmente sería un proceso costoso)
        plt.figure(figsize=(8, 6))
        plt.title("Matriz de Confusión - RandomForest")
        plt.plot([1, 2, 3], [1, 4, 9])
        
        # Guardar en buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        # Guardar en caché
        cache.put(key_params, buf)
        
        print("Visualización guardada en caché")
    else:
        print("Visualización recuperada de caché")
    
    # Mostrar estadísticas
    print("\nEstadísticas de caché:")
    stats = cache.get_stats()
    for k, v in stats.items():
        print(f"- {k}: {v}")
