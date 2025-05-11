#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para verificar la integración entre Node.js/TypeScript y Python.
Este script recibe parámetros desde Node y devuelve una respuesta en formato JSON.
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

def main():
    """
    Función principal que recibe argumentos y devuelve información
    sobre el entorno Python y las librerías disponibles.
    """
    # Intentar leer argumentos
    args = sys.argv[1:]
    
    # Preparar información para devolver
    result = {
        "status": "success",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "timestamp": datetime.now().isoformat(),
        "received_args": args,
        "available_datasets": []
    }
    
    # Verificar datasets disponibles
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    if os.path.exists(datasets_dir):
        result["available_datasets"] = [f for f in os.listdir(datasets_dir) if f.endswith((".csv", ".json"))]
    
    # Devolver resultado en formato JSON
    print(json.dumps(result, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())