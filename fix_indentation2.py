import re

# Path del archivo
file_path = r'c:\Users\Toninopc\Desktop\Programacion\mi-proyecto\data_science_ml_learning\5_especializacion\series_temporales\time_series_analyzer.py'

# Leer el archivo
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Corregir la línea específica línea 1054
corrected_content = content.replace('plt.close()        return img_str', 'plt.close()\n        return img_str')

# Guardar el archivo corregido
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(corrected_content)

print("Archivo corregido correctamente.")
