Este script debe ejecutarse manualmente como administrador.

# Abre una nueva ventana PowerShell como administrador y ejecuta:
Start-Process powershell -Verb RunAs -ArgumentList "-ExecutionPolicy Bypass -Command "cd 'C:\Users\Toninopc\Desktop\Programacion\mi-proyecto'; .\start-https-server.bat""

