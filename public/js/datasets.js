/**
 * DataInsight AI - Módulo de Gestión de Datasets
 * Maneja la carga, visualización y administración de datasets
 */

document.addEventListener('DOMContentLoaded', function() {
    // Inicializar DataTables para la tabla de datasets
    const datasetsTable = $('#datasetsTable').DataTable({
        responsive: true,
        paging: true,
        searching: true,
        ordering: true,
        info: true,
        language: {
            search: "Buscar:",
            lengthMenu: "Mostrar _MENU_ entradas",
            info: "Mostrando _START_ a _END_ de _TOTAL_ entradas",
            infoEmpty: "Mostrando 0 a 0 de 0 entradas",
            infoFiltered: "(filtrado de _MAX_ entradas totales)",
            paginate: {
                first: "Primero",
                last: "Último",
                next: "Siguiente",
                previous: "Anterior"
            }
        }
    });

    // Manejar envío del formulario de carga de dataset
    document.getElementById('uploadDatasetForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Obtener valores del formulario
        const datasetName = document.getElementById('datasetName').value;
        const datasetFile = document.getElementById('datasetFile').files[0];
        const separator = document.getElementById('separator').value;
        const hasHeader = document.getElementById('hasHeader').checked;
        const datasetCategory = document.getElementById('datasetCategory').value;
        const isPublic = document.getElementById('isPublic').checked;
        const datasetDescription = document.getElementById('datasetDescription').value;
        
        if (!datasetName || !datasetFile) {
            showNotification('Por favor, complete todos los campos requeridos.', 'error');
            return;
        }
        
        // En un entorno real, aquí se haría una petición AJAX al servidor
        // para cargar el archivo y procesarlo.
        // Simulamos un tiempo de carga y mostramos un mensaje de éxito
        
        showNotification('Cargando dataset...', 'info');
        
        // Simular carga
        setTimeout(() => {
            // En un entorno real, aquí se procesaría la respuesta del servidor
            // Simulamos que la carga fue exitosa
            
            // Añadir a la tabla (en producción esto vendría del servidor)
            const newRow = datasetsTable.row.add([
                `<div class="dataset-info">
                    <div class="dataset-icon" data-type="csv"><i class="fas fa-table"></i></div>
                    <div>
                        <strong>${datasetName}</strong>
                        <div class="dataset-description">${datasetDescription || 'Sin descripción'}</div>
                    </div>
                </div>`,
                `<span class="badge badge-${datasetCategory}">${datasetCategory}</span>`,
                '---', // En producción, estos valores vendrían del análisis del archivo
                '---',
                `${Math.round(datasetFile.size / 1024)} KB`,
                new Date().toLocaleDateString(),
                `<div class="action-buttons">
                    <button class="btn-icon" title="Visualizar" onclick="viewDataset('${datasetName}')">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn-icon" title="Descargar" onclick="downloadDataset('${datasetName}')">
                        <i class="fas fa-download"></i>
                    </button>
                    <button class="btn-icon" title="Eliminar" onclick="deleteDataset('${datasetName}')">
                        <i class="fas fa-trash-alt"></i>
                    </button>
                </div>`
            ]).draw(false).node();
            
            // Añadir clases para animar la fila nueva
            $(newRow).addClass('highlight-row');
            setTimeout(() => {
                $(newRow).removeClass('highlight-row');
            }, 3000);
            
            // Resetear el formulario
            document.getElementById('uploadDatasetForm').reset();
            
            showNotification(`¡Dataset "${datasetName}" cargado exitosamente!`, 'success');
        }, 1500);
    });
    
    // Filtro de categoría
    document.getElementById('filterCategory').addEventListener('change', function() {
        const category = this.value;
        
        if (category === 'all') {
            datasetsTable.column(1).search('').draw();
        } else {
            datasetsTable.column(1).search(category).draw();
        }
    });
    
    // Ordenamiento
    document.getElementById('sortBy').addEventListener('change', function() {
        const sortOption = this.value.split('-');
        const column = (() => {
            switch (sortOption[0]) {
                case 'name': return 0;
                case 'date': return 5;
                case 'size': return 4;
                default: return 5;
            }
        })();
        
        const direction = sortOption[1] === 'asc' ? 'asc' : 'desc';
        
        datasetsTable.order([column, direction]).draw();
    });
    
    // Búsqueda
    document.getElementById('searchDataset').addEventListener('input', function() {
        datasetsTable.search(this.value).draw();
    });

    // Inicializar tabs en el modal
    document.querySelectorAll('.tab-btn').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.getAttribute('data-tab');
            
            // Desactivar todos los tabs y contenidos
            document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            
            // Activar el tab seleccionado
            tab.classList.add('active');
            document.getElementById('tab' + tabId.charAt(0).toUpperCase() + tabId.slice(1)).classList.remove('hidden');
        });
    });
});

/**
 * Muestra una notificación estilo toast
 * @param {string} message - Mensaje a mostrar
 * @param {string} type - Tipo de notificación: success, error, warning, info
 */
function showNotification(message, type = 'info') {
    // Buscar si ya existe un contenedor de notificaciones
    let notificationContainer = document.getElementById('notificationContainer');
    
    // Si no existe, crearlo
    if (!notificationContainer) {
        notificationContainer = document.createElement('div');
        notificationContainer.id = 'notificationContainer';
        notificationContainer.className = 'notification-container';
        document.body.appendChild(notificationContainer);
    }
    
    // Crear la notificación
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    // Icono según el tipo
    const iconClass = (() => {
        switch (type) {
            case 'success': return 'fas fa-check-circle';
            case 'error': return 'fas fa-exclamation-circle';
            case 'warning': return 'fas fa-exclamation-triangle';
            default: return 'fas fa-info-circle';
        }
    })();
    
    notification.innerHTML = `
        <i class="${iconClass}"></i>
        <span>${message}</span>
        <button class="close-notification">&times;</button>
    `;
    
    // Añadir al contenedor
    notificationContainer.appendChild(notification);
    
    // Añadir listener para cerrar
    notification.querySelector('.close-notification').addEventListener('click', () => {
        notification.classList.add('fade-out');
        setTimeout(() => {
            notification.remove();
        }, 300);
    });
    
    // Auto cerrar después de 5 segundos
    setTimeout(() => {
        if (notification.parentNode) {
            notification.classList.add('fade-out');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }
    }, 5000);
}

/**
 * Abre el modal para visualizar un dataset
 * @param {string} datasetName - Nombre del dataset a visualizar
 */
function viewDataset(datasetName) {
    // Establecer título del modal
    document.getElementById('datasetModalTitle').textContent = `Dataset: ${datasetName}`;
    
    // En un entorno real, aquí haríamos una petición AJAX para obtener los datos del dataset
    // Simulamos una carga de datos
    
    // Mostrar stats del dataset
    document.getElementById('statRows').textContent = datasetName === 'iris_sample' ? '150' : 
                                                    datasetName === 'demo_dataset' ? '1,240' : '3,750';
    document.getElementById('statCols').textContent = datasetName === 'iris_sample' ? '5' : 
                                                    datasetName === 'demo_dataset' ? '8' : '12';
    document.getElementById('statNulls').textContent = datasetName === 'iris_sample' ? '0' : 
                                                    datasetName === 'demo_dataset' ? '15' : '124';
    document.getElementById('statDups').textContent = datasetName === 'iris_sample' ? '0' : 
                                                    datasetName === 'demo_dataset' ? '3' : '17';
    
    // Datos de ejemplo para la vista previa
    let previewData;
    
    if (datasetName === 'iris_sample') {
        previewData = {
            headers: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
            rows: [
                [5.1, 3.5, 1.4, 0.2, 'setosa'],
                [4.9, 3.0, 1.4, 0.2, 'setosa'],
                [4.7, 3.2, 1.3, 0.2, 'setosa'],
                [4.6, 3.1, 1.5, 0.2, 'setosa'],
                [5.0, 3.6, 1.4, 0.2, 'setosa'],
                [5.4, 3.9, 1.7, 0.4, 'setosa'],
                [4.6, 3.4, 1.4, 0.3, 'setosa'],
                [5.0, 3.4, 1.5, 0.2, 'setosa'],
                [4.4, 2.9, 1.4, 0.2, 'setosa'],
                [4.9, 3.1, 1.5, 0.1, 'setosa']
            ]
        };
    } else if (datasetName === 'demo_dataset') {
        previewData = {
            headers: ['id', 'fecha', 'producto', 'categoria', 'precio', 'cantidad', 'total', 'tienda'],
            rows: [
                [1, '2025-01-01', 'Laptop', 'Electrónica', 1200.50, 1, 1200.50, 'Madrid'],
                [2, '2025-01-02', 'Teclado', 'Accesorios', 45.99, 2, 91.98, 'Barcelona'],
                [3, '2025-01-03', 'Mouse', 'Accesorios', 25.50, 3, 76.50, 'Valencia'],
                [4, '2025-01-04', 'Monitor', 'Electrónica', 350.75, 1, 350.75, 'Madrid'],
                [5, '2025-01-05', 'SSD 1TB', 'Componentes', 120.00, 2, 240.00, 'Barcelona'],
                [6, '2025-01-06', 'RAM 16GB', 'Componentes', 85.25, 4, 341.00, 'Madrid'],
                [7, '2025-01-07', 'Cables', 'Accesorios', 12.99, 5, 64.95, 'Sevilla'],
                [8, '2025-01-08', 'Laptop', 'Electrónica', 1100.00, 1, 1100.00, 'Valencia'],
                [9, '2025-01-09', 'Audífonos', 'Accesorios', 89.99, 2, 179.98, 'Madrid'],
                [10, '2025-01-10', 'Cargador', 'Accesorios', 30.50, 3, 91.50, 'Barcelona']
            ]
        };
    } else {
        previewData = {
            headers: ['fecha', 'region', 'producto', 'categoria', 'subcategoria', 'unidades', 'precio_unitario', 'costo_unitario', 'ingresos', 'costos', 'ganancia', 'vendedor'],
            rows: [
                ['2025-01-15', 'Norte', 'Producto A', 'Electrónica', 'Smartphones', 12, 599.99, 350.00, 7199.88, 4200.00, 2999.88, 'Ana López'],
                ['2025-01-17', 'Sur', 'Producto B', 'Hogar', 'Cocina', 5, 129.99, 70.00, 649.95, 350.00, 299.95, 'Carlos Ruiz'],
                ['2025-01-20', 'Este', 'Producto C', 'Electrónica', 'Audio', 8, 89.99, 45.00, 719.92, 360.00, 359.92, 'Elena Santos'],
                ['2025-01-22', 'Oeste', 'Producto D', 'Informática', 'Tablets', 3, 349.99, 210.00, 1049.97, 630.00, 419.97, 'Juan Martín'],
                ['2025-01-25', 'Norte', 'Producto E', 'Electrodomésticos', 'Lavadoras', 2, 499.99, 320.00, 999.98, 640.00, 359.98, 'Ana López'],
                ['2025-01-28', 'Centro', 'Producto F', 'Moda', 'Ropa', 15, 35.99, 18.00, 539.85, 270.00, 269.85, 'Pedro Vázquez'],
                ['2025-01-30', 'Sur', 'Producto G', 'Electrónica', 'TV', 4, 799.99, 500.00, 3199.96, 2000.00, 1199.96, 'Carlos Ruiz'],
                ['2025-02-03', 'Este', 'Producto H', 'Jardín', 'Herramientas', 6, 79.99, 40.00, 479.94, 240.00, 239.94, 'Elena Santos'],
                ['2025-02-05', 'Oeste', 'Producto I', 'Informática', 'Portátiles', 2, 899.99, 650.00, 1799.98, 1300.00, 499.98, 'Juan Martín'],
                ['2025-02-08', 'Norte', 'Producto J', 'Hogar', 'Muebles', 3, 249.99, 150.00, 749.97, 450.00, 299.97, 'Ana López']
            ]
        };
    }
    
    // Generar tabla de vista previa
    const previewTable = document.getElementById('previewTable');
    const tableContent = `
        <thead>
            <tr>
                ${previewData.headers.map(header => `<th>${header}</th>`).join('')}
            </tr>
        </thead>
        <tbody>
            ${previewData.rows.map(row => `
                <tr>
                    ${row.map(cell => `<td>${cell}</td>`).join('')}
                </tr>
            `).join('')}
        </tbody>
    `;
    previewTable.innerHTML = tableContent;
    
    // Generar estadísticas para cada columna (simplificado)
    let statsContent = '';
    previewData.headers.forEach((header, index) => {
        const values = previewData.rows.map(row => row[index]);
        const isNumeric = typeof values[0] === 'number';
        
        if (isNumeric) {
            const sum = values.reduce((acc, val) => acc + val, 0);
            const mean = sum / values.length;
            const sortedValues = [...values].sort((a, b) => a - b);
            const median = sortedValues.length % 2 === 0 ? 
                (sortedValues[sortedValues.length/2 - 1] + sortedValues[sortedValues.length/2]) / 2 :
                sortedValues[Math.floor(sortedValues.length/2)];
            const min = Math.min(...values);
            const max = Math.max(...values);
            
            statsContent += `
                <tr>
                    <td>${header}</td>
                    <td>Numérico</td>
                    <td>0</td>
                    <td>${mean.toFixed(2)}</td>
                    <td>${median.toFixed(2)}</td>
                    <td>${min}</td>
                    <td>${max}</td>
                </tr>
            `;
        } else {
            statsContent += `
                <tr>
                    <td>${header}</td>
                    <td>Texto</td>
                    <td>0</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            `;
        }
    });
    
    document.getElementById('statsTable').querySelector('tbody').innerHTML = statsContent;
    
    // En un entorno real, aquí cargaríamos los gráficos con datos reales
    // Simulamos la visualización con Chart.js
    initPreviewCharts(datasetName, previewData);
    
    // Mostrar el modal
    document.getElementById('viewDatasetModal').style.display = 'block';
}

/**
 * Inicializa gráficos para la previsualización de datos
 */
function initPreviewCharts(datasetName, previewData) {
    // Destruir charts anteriores si existen
    if (window.columnChart) window.columnChart.destroy();
    if (window.correlationChart) window.correlationChart.destroy();
    
    // Obtener el contexto de los canvas
    const columnCtx = document.getElementById('columnDistChart').getContext('2d');
    const correlationCtx = document.getElementById('correlationChart').getContext('2d');
    
    // Preparar datos para el chart de distribución de columnas
    // Tomamos la primera columna numérica para el ejemplo
    let numericColIndex = previewData.headers.findIndex((header, index) => 
        typeof previewData.rows[0][index] === 'number');
    
    if (numericColIndex === -1) numericColIndex = 0; // Fallback
    
    const numericHeader = previewData.headers[numericColIndex];
    const numericValues = previewData.rows.map(row => row[numericColIndex]);
    
    // Chart de distribución
    window.columnChart = new Chart(columnCtx, {
        type: 'bar',
        data: {
            labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
            datasets: [{
                label: `Distribución de ${numericHeader}`,
                data: generateHistogramData(numericValues),
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Distribución de valores de ${numericHeader}`
                }
            }
        }
    });
    
    // Chart de correlación (heatmap simulado)
    const numericCols = previewData.headers.filter((header, index) => 
        typeof previewData.rows[0][index] === 'number');
    
    const correlationData = [];
    for (let i = 0; i < numericCols.length; i++) {
        correlationData.push([]);
        for (let j = 0; j < numericCols.length; j++) {
            // Generar correlación simulada entre -1 y 1
            correlationData[i].push(Math.random() * 2 - 1);
        }
    }
    
    window.correlationChart = new Chart(correlationCtx, {
        type: 'heatmap',
        data: {
            labels: numericCols,
            datasets: numericCols.map((col, i) => ({
                label: col,
                data: correlationData[i].map((val, j) => ({
                    x: numericCols[j],
                    y: col,
                    v: val
                }))
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Matriz de Correlación'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    type: 'category',
                    labels: numericCols
                },
                y: {
                    type: 'category',
                    labels: numericCols
                }
            }
        }
    });
    
    // Nota: Chart.js no tiene un tipo 'heatmap' nativo, esta es una simplificación
    // En un entorno de producción, se debería usar una librería de extensión de Chart.js
    // o una librería específica para heatmaps como Plotly, D3, etc.
}

/**
 * Genera datos para un histograma simple dividido en 5 rangos
 * @param {Array<number>} values - Array de valores numéricos
 * @return {Array<number>} - Array con el conteo de valores en cada rango
 */
function generateHistogramData(values) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    const bins = 5;
    const binSize = range / bins;
    
    const histogramData = Array(bins).fill(0);
    
    values.forEach(value => {
        const binIndex = Math.min(Math.floor((value - min) / binSize), bins - 1);
        histogramData[binIndex]++;
    });
    
    return histogramData;
}

/**
 * Descarga un dataset
 * @param {string} datasetName - Nombre del dataset a descargar
 */
function downloadDataset(datasetName) {
    // En un entorno real, esta función descargaría el archivo
    // del servidor usando una petición AJAX
    
    showNotification(`Descargando ${datasetName}.csv...`, 'info');
    
    // Simular descarga
    setTimeout(() => {
        showNotification(`¡Dataset "${datasetName}" descargado!`, 'success');
    }, 1500);
}

/**
 * Elimina un dataset
 * @param {string} datasetName - Nombre del dataset a eliminar
 */
function deleteDataset(datasetName) {
    // Mostrar confirmación
    if (!confirm(`¿Estás seguro de que deseas eliminar el dataset "${datasetName}"?`)) {
        return;
    }
    
    showNotification(`Eliminando ${datasetName}...`, 'info');
    
    // En un entorno real, enviaríamos una petición AJAX para eliminar el dataset
    // Simulamos la eliminación
    
    setTimeout(() => {
        // Encontrar y eliminar la fila de la tabla
        const table = $('#datasetsTable').DataTable();
        
        table.rows().every(function() {
            const rowData = this.data();
            if (rowData[0].includes(datasetName)) {
                const row = this.node();
                $(row).addClass('fade-out');
                
                setTimeout(() => {
                    this.remove().draw(false);
                    showNotification(`¡Dataset "${datasetName}" eliminado!`, 'success');
                }, 300);
                
                return false; // Salir del bucle
            }
            return true; // Continuar bucle
        });
    }, 1000);
}

/**
 * Cierra el modal de visualización de dataset
 */
function closeModal() {
    document.getElementById('viewDatasetModal').style.display = 'none';
}

/**
 * Envía el dataset seleccionado a la página de análisis
 */
function loadDatasetForAnalysis() {
    const datasetName = document.getElementById('datasetModalTitle').textContent.replace('Dataset: ', '');
    
    // En un entorno real, guardaríamos el dataset seleccionado en localStorage o enviaríamos
    // su ID a la página de análisis mediante parámetros de URL
    
    localStorage.setItem('selectedDataset', datasetName);
    
    showNotification(`Redirigiendo a análisis con dataset "${datasetName}"...`, 'info');
    
    setTimeout(() => {
        window.location.href = '/analisis.html?dataset=' + encodeURIComponent(datasetName);
    }, 1000);
}