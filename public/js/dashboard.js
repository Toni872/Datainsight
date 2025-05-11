/**
 * DataInsight AI - Dashboard Javascript
 * Script para generar gráficas interactivas y visualizaciones
 */

// Objeto que contendrá las funciones públicas del dashboard
window.dashboardFunctions = {};

// Esperar a que el DOM esté cargado
document.addEventListener('DOMContentLoaded', () => {
    // Inicializar todas las gráficas
    initializeCharts();

    // Configurar los listeners para filtros y controles interactivos
    setupEventListeners();
});

/**
 * Inicializa todas las gráficas del dashboard
 */
function initializeCharts() {
    createModelsPerformanceChart();
    createClassDistributionChart();
    createFeatureImportanceChart();
    createCorrelationHeatmap();
    createPredictionAccuracyTimeline();
}

/**
 * Configura los event listeners para interactividad
 */
function setupEventListeners() {
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleDarkMode);
    }

    const refreshButton = document.getElementById('refresh-datasets');
    if (refreshButton) {
        refreshButton.addEventListener('click', function() {
            updateDatasetSelector();
            showMessage('Lista de datasets actualizada', 'success');
        });
    }

    const modelSelector = document.getElementById('model-selector');
    if (modelSelector) {
        modelSelector.addEventListener('change', updateChartsBasedOnModel);
    }
}

/**
 * Crea la gráfica de rendimiento de modelos
 */
function createModelsPerformanceChart() {
    const ctx = document.getElementById('modelsPerformanceChart');
    if (!ctx) return;
    
    const context = ctx.getContext('2d');
    
    // Datos de ejemplo para la gráfica
    const data = {
        labels: ['Random Forest', 'SVM', 'Regresión Logística', 'KNN', 'XGBoost', 'Decision Tree'],
        datasets: [
            {
                label: 'Precisión',
                data: [98.3, 92.7, 87.5, 89.2, 97.8, 85.1],
                backgroundColor: 'rgba(67, 97, 238, 0.7)',
                borderColor: 'rgba(67, 97, 238, 1)',
                borderWidth: 2
            },
            {
                label: 'Recall',
                data: [97.1, 91.4, 85.3, 88.7, 96.5, 83.9],
                backgroundColor: 'rgba(76, 201, 240, 0.7)',
                borderColor: 'rgba(76, 201, 240, 1)',
                borderWidth: 2
            },
            {
                label: 'F1-Score',
                data: [97.7, 92.0, 86.4, 89.0, 97.1, 84.5],
                backgroundColor: 'rgba(247, 37, 133, 0.7)',
                borderColor: 'rgba(247, 37, 133, 1)',
                borderWidth: 2
            }
        ]
    };

    // Opciones para la gráfica
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            },
            tooltip: {
                mode: 'index',
                intersect: false,
            },
            title: {
                display: false,
                text: 'Rendimiento de Modelos'
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                min: 70,
                max: 100,
                ticks: {
                    callback: function(value) {
                        return value + '%';
                    }
                }
            }
        }
    };

    // Crear la gráfica
    new Chart(context, {
        type: 'bar',
        data: data,
        options: options
    });
}

/**
 * Crea la gráfica de distribución de clases
 */
function createClassDistributionChart() {
    const ctx = document.getElementById('classDistributionChart');
    if (!ctx) return;
    
    const context = ctx.getContext('2d');
    
    // Datos de ejemplo para la gráfica
    const data = {
        labels: ['Setosa', 'Versicolor', 'Virginica'],
        datasets: [{
            data: [50, 50, 50],
            backgroundColor: [
                'rgba(67, 97, 238, 0.7)',
                'rgba(76, 201, 240, 0.7)',
                'rgba(247, 37, 133, 0.7)'
            ],
            borderColor: [
                'rgba(67, 97, 238, 1)',
                'rgba(76, 201, 240, 1)',
                'rgba(247, 37, 133, 1)'
            ],
            borderWidth: 1
        }]
    };

    // Opciones para la gráfica
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'right',
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const label = context.label || '';
                        const value = context.raw || 0;
                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                        const percentage = Math.round((value / total) * 100);
                        return `${label}: ${value} (${percentage}%)`;
                    }
                }
            },
        },
    };

    // Crear la gráfica
    new Chart(context, {
        type: 'pie',
        data: data,
        options: options
    });
}

/**
 * Crea la gráfica de importancia de características
 */
function createFeatureImportanceChart() {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx) return;
    
    const context = ctx.getContext('2d');
    
    // Datos de ejemplo para la gráfica
    const data = {
        labels: ['petal_length', 'petal_width', 'sepal_length', 'sepal_width'],
        datasets: [{
            label: 'Importancia',
            data: [0.45, 0.41, 0.08, 0.06],
            backgroundColor: [
                'rgba(67, 97, 238, 0.7)',
                'rgba(76, 201, 240, 0.7)',
                'rgba(247, 37, 133, 0.7)',
                'rgba(148, 63, 219, 0.7)'
            ],
            borderColor: [
                'rgba(67, 97, 238, 1)',
                'rgba(76, 201, 240, 1)',
                'rgba(247, 37, 133, 1)',
                'rgba(148, 63, 219, 1)'
            ],
            borderWidth: 1
        }]
    };

    // Opciones para la gráfica
    const options = {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const value = context.raw || 0;
                        return `Importancia: ${(value * 100).toFixed(1)}%`;
                    }
                }
            }
        },
        scales: {
            x: {
                beginAtZero: true,
                max: 0.5,
                ticks: {
                    callback: function(value) {
                        return (value * 100).toFixed(0) + '%';
                    }
                }
            }
        }
    };

    // Crear la gráfica
    new Chart(context, {
        type: 'bar',
        data: data,
        options: options
    });
}

/**
 * Crea un heatmap de correlación entre características
 */
function createCorrelationHeatmap() {
    const ctx = document.getElementById('correlationHeatmap');
    if (!ctx) return;
    
    const context = ctx.getContext('2d');
    
    // Simular una matriz de correlación
    const correlationMatrix = [
        [1.00, 0.12, 0.87, 0.82],
        [0.12, 1.00, 0.29, 0.18],
        [0.87, 0.29, 1.00, 0.96],
        [0.82, 0.18, 0.96, 1.00]
    ];
    
    const features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'];
    
    // Preparar datos para el heatmap
    const datasets = [];
    
    for (let i = 0; i < correlationMatrix.length; i++) {
        const rowData = correlationMatrix[i].map(value => ({
            x: i,
            y: correlationMatrix.length - 1 - features.indexOf(features[i]), 
            v: value
        }));
        
        datasets.push(...rowData);
    }
    
    // Definir la escala de colores
    const getColor = value => {
        // Azul a rojo
        if (value <= 0) {
            return `rgba(76, 201, 240, ${Math.abs(value)})`;
        } else {
            return `rgba(247, 37, 133, ${value})`;
        }
    };
    
    // Crear el gráfico personalizado (renderizado manual)
    const cellSize = Math.min(ctx.width / features.length, ctx.height / features.length);
    
    // Limpiar el canvas
    context.clearRect(0, 0, ctx.width, ctx.height);
    
    // Dibujar las celdas
    for (let i = 0; i < features.length; i++) {
        for (let j = 0; j < features.length; j++) {
            const value = correlationMatrix[i][j];
            const x = j * cellSize;
            const y = i * cellSize;
            
            // Dibujar el fondo de la celda
            context.fillStyle = getColor(value);
            context.fillRect(x, y, cellSize, cellSize);
            
            // Dibujar el valor de correlación
            context.fillStyle = 'black';
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            context.font = '10px Arial';
            context.fillText(value.toFixed(2), x + cellSize / 2, y + cellSize / 2);
        }
    }
    
    // Dibujar etiquetas
    context.fillStyle = 'black';
    context.textAlign = 'center';
    context.font = '12px Arial';
    
    // Etiquetas de columnas
    for (let i = 0; i < features.length; i++) {
        context.save();
        context.translate(i * cellSize + cellSize / 2, features.length * cellSize + 10);
        context.fillText(features[i], 0, 0);
        context.restore();
    }
    
    // Etiquetas de filas
    for (let i = 0; i < features.length; i++) {
        context.save();
        context.translate(-10, i * cellSize + cellSize / 2);
        context.rotate(-Math.PI / 2);
        context.textAlign = 'center';
        context.fillText(features[i], 0, 0);
        context.restore();
    }
}

/**
 * Crea una línea de tiempo de precisión de predicciones
 */
function createPredictionAccuracyTimeline() {
    const ctx = document.getElementById('predictionTimeline');
    if (!ctx) return;
    
    const context = ctx.getContext('2d');
    
    // Generar fechas para los últimos 12 meses
    const labels = [];
    const currentDate = new Date();
    for (let i = 11; i >= 0; i--) {
        const date = new Date(currentDate);
        date.setMonth(currentDate.getMonth() - i);
        labels.push(date.toLocaleString('es-ES', { month: 'short', year: '2-digit' }));
    }
    
    // Datos simulados de precisión de predicción
    const data = {
        labels: labels,
        datasets: [
            {
                label: 'Random Forest',
                data: [94.5, 94.7, 95.2, 95.8, 95.9, 96.3, 96.5, 97.1, 97.4, 97.8, 98.0, 98.3],
                borderColor: 'rgba(67, 97, 238, 1)',
                backgroundColor: 'rgba(67, 97, 238, 0.2)',
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointHoverRadius: 6
            },
            {
                label: 'XGBoost',
                data: [93.8, 94.1, 94.5, 95.2, 95.6, 95.8, 96.2, 96.5, 96.9, 97.2, 97.5, 97.8],
                borderColor: 'rgba(76, 201, 240, 1)',
                backgroundColor: 'rgba(76, 201, 240, 0.2)',
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointHoverRadius: 6
            }
        ]
    };

    // Opciones para la gráfica
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                callbacks: {
                    label: function(context) {
                        return `${context.dataset.label}: ${context.raw}%`;
                    }
                }
            },
        },
        scales: {
            y: {
                beginAtZero: false,
                min: 90,
                max: 100,
                ticks: {
                    callback: function(value) {
                        return value + '%';
                    }
                },
                title: {
                    display: true,
                    text: 'Precisión (%)'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Mes'
                }
            }
        }
    };

    // Crear la gráfica
    new Chart(context, {
        type: 'line',
        data: data,
        options: options
    });
}

/**
 * Actualiza las gráficas basadas en el modelo seleccionado
 */
function updateChartsBasedOnModel() {
    const modelSelector = document.getElementById('model-selector');
    if (!modelSelector) return;
    
    const selectedModel = modelSelector.value;
    
    // Aquí iría la lógica para cargar datos del modelo seleccionado
    // Por ahora, simulamos una actualización visual
    
    // Actualización simulada para diferentes modelos
    if (selectedModel === 'random_forest') {
        updateChartData('featureImportanceChart', {
            data: [0.45, 0.41, 0.08, 0.06]
        });
    } else if (selectedModel === 'svm') {
        updateChartData('featureImportanceChart', {
            data: [0.40, 0.35, 0.15, 0.10]
        });
    } else if (selectedModel === 'logistic_regression') {
        updateChartData('featureImportanceChart', {
            data: [0.38, 0.36, 0.14, 0.12]
        });
    }
}

/**
 * Actualiza datos de una gráfica existente
 */
function updateChartData(chartId, newData) {
    const chartElement = document.getElementById(chartId);
    if (!chartElement) return;
    
    const chartInstance = Chart.getChart(chartElement);
    if (!chartInstance) return;
    
    // Actualizar los datos
    if (newData.data) {
        chartInstance.data.datasets[0].data = newData.data;
    }
    
    if (newData.labels) {
        chartInstance.data.labels = newData.labels;
    }
    
    // Redibujar la gráfica
    chartInstance.update();
}

/**
 * Alternancia entre modo claro y oscuro
 */
function toggleDarkMode() {
    const body = document.body;
    body.classList.toggle('dark-mode');
    
    const isDarkMode = body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDarkMode);
    
    // Cambiar texto del botón
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.innerHTML = isDarkMode ? 
            '<i class="fas fa-sun"></i> Modo Claro' : 
            '<i class="fas fa-moon"></i> Modo Oscuro';
    }
    
    // Actualizar todas las gráficas para el nuevo tema
    updateChartsTheme(isDarkMode);
}

/**
 * Actualiza el tema de todas las gráficas
 */
function updateChartsTheme(isDarkMode) {
    // Color de texto para etiquetas y títulos
    const textColor = isDarkMode ? '#FFFFFF' : '#333333';
    const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    
    // Actualizar configuración global de Chart.js
    Chart.defaults.color = textColor;
    Chart.defaults.borderColor = gridColor;
    
    // Actualizar cada gráfica
    document.querySelectorAll('canvas').forEach(canvas => {
        const chart = Chart.getChart(canvas);
        if (chart) {
            // Actualizar colores de ejes
            if (chart.options.scales) {
                Object.values(chart.options.scales).forEach(scaleOptions => {
                    scaleOptions.grid = {
                        ...scaleOptions.grid,
                        color: gridColor
                    };
                    scaleOptions.ticks = {
                        ...scaleOptions.ticks,
                        color: textColor
                    };
                    if (scaleOptions.title) {
                        scaleOptions.title.color = textColor;
                    }
                });
            }
            
            // Actualizar leyenda
            if (chart.options.plugins && chart.options.plugins.legend) {
                chart.options.plugins.legend.labels = {
                    ...chart.options.plugins.legend.labels,
                    color: textColor
                };
            }
            
            // Actualizar la gráfica
            chart.update();
        }
    });
}

// Verificar si el modo oscuro estaba activado anteriormente
document.addEventListener('DOMContentLoaded', function() {
    const darkMode = localStorage.getItem('darkMode') === 'true';
    
    if (darkMode) {
        document.body.classList.add('dark-mode');
        
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.innerHTML = '<i class="fas fa-sun"></i> Modo Claro';
        }
        
        // Actualizar temas de gráficos
        updateChartsTheme(true);
    }
});

/**
 * Función para mostrar mensajes de retroalimentación al usuario
 */
function showMessage(message, type = 'info') {
    // Verificar si ya existe un contenedor de mensajes
    let messageContainer = document.getElementById('message-container');
    
    if (!messageContainer) {
        // Crear el contenedor si no existe
        messageContainer = document.createElement('div');
        messageContainer.id = 'message-container';
        messageContainer.style.position = 'fixed';
        messageContainer.style.top = '20px';
        messageContainer.style.right = '20px';
        messageContainer.style.zIndex = '1000';
        document.body.appendChild(messageContainer);
    }
    
    // Crear el elemento de mensaje
    const messageElement = document.createElement('div');
    messageElement.className = `message ${type}`;
    messageElement.innerHTML = `
        <div class="message-content">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    messageElement.style.backgroundColor = type === 'success' ? '#4CAF50' : type === 'error' ? '#F44336' : '#2196F3';
    messageElement.style.color = 'white';
    messageElement.style.padding = '10px 15px';
    messageElement.style.borderRadius = '4px';
    messageElement.style.marginBottom = '10px';
    messageElement.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
    messageElement.style.display = 'flex';
    messageElement.style.alignItems = 'center';
    messageElement.style.animation = 'fadeIn 0.5s';
    
    // Añadir el mensaje al contenedor
    messageContainer.appendChild(messageElement);
    
    // Eliminar el mensaje después de 3 segundos
    setTimeout(() => {
        messageElement.style.animation = 'fadeOut 0.5s';
        setTimeout(() => {
            if (messageContainer.contains(messageElement)) {
                messageContainer.removeChild(messageElement);
            }
        }, 500);
    }, 3000);
}

/**
 * Actualiza el selector de datasets
 */
function updateDatasetSelector() {
    // Esta función se llamaría para cargar datos reales desde el backend
    console.log('Actualizando lista de datasets...');
    // Por ahora solo es un placeholder
}

// Exponer funciones públicas
window.dashboardFunctions = {
    refreshCharts: function() {
        console.log('Actualizando gráficas...');
        // Simular una actualización de datos
        setTimeout(() => {
            // Actualizar aleatoriamente algunos datos
            const newData = {
                labels: ['Random Forest', 'SVM', 'Regresión Logística', 'KNN', 'XGBoost', 'Decision Tree'],
                datasets: [
                    {
                        label: 'Precisión',
                        data: [
                            95 + Math.random() * 5, 
                            90 + Math.random() * 5, 
                            85 + Math.random() * 5, 
                            87 + Math.random() * 5, 
                            95 + Math.random() * 5, 
                            83 + Math.random() * 5
                        ]
                    }
                ]
            };
            
            const charts = ['modelsPerformanceChart', 'classDistributionChart', 'featureImportanceChart'];
            charts.forEach(chartId => {
                const chartElement = document.getElementById(chartId);
                if (chartElement) {
                    const chartInstance = Chart.getChart(chartElement);
                    if (chartInstance) {
                        // Actualizar datos aleatoriamente para simular cambios
                        chartInstance.data.datasets.forEach(dataset => {
                            dataset.data = dataset.data.map(() => 80 + Math.random() * 20);
                        });
                        chartInstance.update();
                    }
                }
            });
            
            showMessage('Datos actualizados correctamente', 'success');
        }, 500);
    },
    
    loadRealtimeData: function() {
        showMessage('Activando modo de datos en tiempo real', 'info');
        
        // Simular intervalos de datos en tiempo real
        const intervalId = setInterval(() => {
            window.dashboardFunctions.refreshCharts();
        }, 5000);
        
        // Almacenar el ID para poder detenerlo después
        window.realtimeInterval = intervalId;
        
        // Cambiar el texto del botón
        const button = document.querySelector('button.btn-secondary');
        if (button) {
            button.innerHTML = '<i class="fas fa-stop"></i> Detener datos en tiempo real';
            button.onclick = function() {
                clearInterval(window.realtimeInterval);
                showMessage('Modo de datos en tiempo real desactivado', 'info');
                button.innerHTML = '<i class="fas fa-clock"></i> Activar datos en tiempo real';
                button.onclick = window.dashboardFunctions.loadRealtimeData;
            };
        }
    },

    // Exportar también otras funciones útiles
    toggleDarkMode: toggleDarkMode,
    updateChartsBasedOnModel: updateChartsBasedOnModel
};

// Verificar si el modo oscuro estaba activado anteriormente
document.addEventListener('DOMContentLoaded', function() {
    const darkMode = localStorage.getItem('darkMode') === 'true';
    
    if (darkMode) {
        document.body.classList.add('dark-mode');
        
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.innerHTML = '<i class="fas fa-sun"></i> Modo Claro';
        }
        
        // Actualizar temas de gráficos
        updateChartsTheme(true);
    }
});