// Este archivo contiene el código JavaScript del lado del cliente.

// Función para mostrar mensajes de error o éxito
function showMessage(message, type = 'success') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    const container = document.querySelector('.container');
    const main = document.querySelector('main');
    
    container.insertBefore(alertDiv, main);
    
    // Eliminar mensaje después de 5 segundos
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Función para realizar solicitudes a la API
async function fetchAPI(url) {
    try {
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`Error de servidor: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error al realizar la solicitud:', error);
        showMessage(`Error: ${error.message}`, 'danger');
        return null;
    }
}

// Función para ejecutar análisis de datos
async function runDataAnalysis(datasetName) {
    const resultContainer = document.getElementById('result-container');
    resultContainer.innerHTML = '<p>Cargando resultados...</p>';
    
    const data = await fetchAPI(`/api/analyze/${datasetName}`);
    
    if (data && data.success) {
        // Formatear el texto para preservar espacios y saltos de línea
        const formattedResult = data.resultado
            .replace(/\n/g, '<br>')
            .replace(/\s{2,}/g, function(match) {
                return '&nbsp;'.repeat(match.length);
            });
            
        resultContainer.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h3>Resultados del análisis de datos</h3>
                </div>
                <div class="card-body">
                    <div class="result-text">${formattedResult}</div>
                    <p><small>Generado el: ${new Date(data.timestamp).toLocaleString('es-ES')}</small></p>
                </div>
            </div>
        `;
    } else {
        resultContainer.innerHTML = `
            <div class="alert alert-danger">
                No se pudieron obtener resultados. Intente nuevamente.
            </div>
        `;
    }
}

// Función para ejecutar modelo de machine learning
async function runMachineLearning(datasetName) {
    const resultContainer = document.getElementById('result-container');
    resultContainer.innerHTML = '<p>Ejecutando modelo de machine learning...</p>';
    
    const data = await fetchAPI(`/api/ml/classify/${datasetName}`);
    
    if (data && data.success) {
        const resultado = data.resultado;
        
        let featuresHTML = '';
        if (resultado.importancia_caracteristicas) {
            featuresHTML = '<h4>Importancia de características:</h4><ul>';
            for (const [feature, value] of Object.entries(resultado.importancia_caracteristicas)) {
                featuresHTML += `<li>${feature}: ${value}</li>`;
            }
            featuresHTML += '</ul>';
        }
        
        resultContainer.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h3>Resultados del modelo ${data.modelo}</h3>
                </div>
                <div class="card-body">
                    <p><strong>Dataset:</strong> ${resultado.dataset}</p>
                    <p><strong>Número de muestras:</strong> ${resultado.num_muestras}</p>
                    <p><strong>Número de características:</strong> ${resultado.num_caracteristicas}</p>
                    <p><strong>Precisión:</strong> ${resultado.precision}%</p>
                    
                    ${featuresHTML}
                    
                    <div class="chart-container" id="features-chart"></div>
                    
                    <p><small>Generado el: ${new Date(data.timestamp).toLocaleString('es-ES')}</small></p>
                </div>
            </div>
        `;
        
        // Visualizar la importancia de características (requeriría una librería como Chart.js)
        if (window.Chart && resultado.importancia_caracteristicas) {
            visualizeFeatureImportance(resultado.importancia_caracteristicas);
        }
    } else {
        resultContainer.innerHTML = `
            <div class="alert alert-danger">
                No se pudieron obtener resultados del modelo. Intente nuevamente.
            </div>
        `;
    }
}

// Función para visualizar la importancia de características (requiere Chart.js)
function visualizeFeatureImportance(featuresData) {
    const ctx = document.getElementById('features-chart').getContext('2d');
    
    const labels = Object.keys(featuresData);
    const data = Object.values(featuresData);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importancia de características',
                data: data,
                backgroundColor: 'rgba(52, 152, 219, 0.6)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Función para cargar un dataset personalizado
async function uploadCustomDataset(formData) {
    try {
        const response = await fetch('/api/datasets/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Error al subir el archivo: ${response.statusText}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error al subir el dataset:', error);
        showMessage(`Error: ${error.message}`, 'danger');
        return null;
    }
}

// Función para actualizar el selector de datasets
async function updateDatasetSelector() {
    const datasetSelect = document.getElementById('dataset-select');
    if (!datasetSelect) return;
    
    try {
        const response = await fetch('/api/datasets/list');
        if (!response.ok) {
            throw new Error('No se pudieron obtener los datasets');
        }
        
        const data = await response.json();
        
        // Guardar la opción seleccionada actualmente
        const currentValue = datasetSelect.value;
        
        // Limpiar todas las opciones excepto la primera (placeholder)
        while (datasetSelect.options.length > 1) {
            datasetSelect.remove(1);
        }
        
        // Añadir los datasets disponibles
        data.datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = dataset.name;
            datasetSelect.appendChild(option);
        });
        
        // Restaurar la selección si aún existe
        if (data.datasets.some(d => d.id === currentValue)) {
            datasetSelect.value = currentValue;
        }
        
    } catch (error) {
        console.error('Error al actualizar los datasets:', error);
    }
}

// Inicialización cuando el DOM está completamente cargado
document.addEventListener('DOMContentLoaded', () => {
    // Comprobar si estamos en la página de análisis de datos
    const datasetForm = document.getElementById('dataset-form');
    
    if (datasetForm) {
        datasetForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const datasetSelect = document.getElementById('dataset-select');
            const analysisType = document.querySelector('input[name="analysis-type"]:checked').value;
            
            if (datasetSelect.value) {
                if (analysisType === 'basic') {
                    runDataAnalysis(datasetSelect.value);
                } else if (analysisType === 'ml') {
                    runMachineLearning(datasetSelect.value);
                }
            } else {
                showMessage('Por favor, seleccione un dataset', 'danger');
            }
        });
    }
    
    // Verificar si estamos en la página de modelos y existe el formulario de subida
    const uploadForm = document.getElementById('upload-dataset-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const fileInput = document.getElementById('dataset-file');
            const nameInput = document.getElementById('dataset-name');
            
            if (!fileInput.files[0]) {
                showMessage('Por favor, seleccione un archivo CSV para subir', 'danger');
                return;
            }
            
            if (!nameInput.value.trim()) {
                showMessage('Por favor, proporcione un nombre para el dataset', 'danger');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('name', nameInput.value.trim());
            
            // Mostrar mensaje de carga
            showMessage('Subiendo dataset, por favor espere...', 'info');
            
            const result = await uploadCustomDataset(formData);
            
            if (result && result.success) {
                showMessage(`Dataset "${result.dataset.name}" subido correctamente`, 'success');
                
                // Limpiar el formulario
                fileInput.value = '';
                nameInput.value = '';
                
                // Actualizar la lista de datasets disponibles
                updateDatasetSelector();
            }
        });
    }
    
    // Actualizar la lista de datasets disponibles al cargar la página
    updateDatasetSelector();
});