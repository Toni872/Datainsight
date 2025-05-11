/**
 * models.js - Funcionalidad específica para la página de modelos
 * Este archivo maneja la lógica de interacción con modelos de Machine Learning
 * @author DataInsight AI
 * @version 2.0.0
 */

// Clase para gestionar los modelos de Machine Learning
class MLModelManager {
    constructor() {
        this.selectedDataset = null;
        this.selectedModels = [];
        this.currentModelType = 'classification';
        this.trainedModels = {};
        this.datasetInfo = {};
        this.dataPreview = [];
        this.isAdvancedModeAvailable = false;
        this.advancedModels = {
            classification: [],
            regression: [],
            clustering: [],
            timeseries: []
        };
        
        // Inicializar la interfaz
        this.initializeUI();
    }
    
    /**
     * Inicializa todos los elementos de la interfaz y eventos
     */
    initializeUI() {
        // Selectores y elementos
        this.datasetSelect = document.getElementById('dataset-select');
        this.modelTypeSelect = document.getElementById('model-type');
        this.trainButton = document.getElementById('train-model');
        this.refreshDatasetsButton = document.getElementById('refresh-datasets');
        this.uploadDatasetForm = document.getElementById('upload-dataset-form');
        this.predictionForm = document.getElementById('prediction-form');
        this.testSizeRange = document.getElementById('test-size');
        this.toggleParamsButton = document.getElementById('toggle-params');
        this.advancedOptionsContainer = document.getElementById('advanced-options-container');
        
        // Establecer controladores de eventos
        this.setupEventListeners();
        
        // Verificar si está disponible el modo avanzado
        this.checkAdvancedModeAvailability();
        
        // Cargar datos iniciales
        this.loadDatasets();
    }
    
    /**
     * Verifica si el modo avanzado está disponible y carga sus opciones
     */
    async checkAdvancedModeAvailability() {
        try {
            const response = await fetch('/api/ml/status');
            const data = await response.json();
            
            if (data.success && data.isAdvancedApiAvailable) {
                this.isAdvancedModeAvailable = true;
                await this.loadAdvancedModels();
                
                // Actualizar la interfaz para mostrar opciones avanzadas
                this.showAdvancedModeOptions();
            } else {
                this.isAdvancedModeAvailable = false;
                console.log('API avanzada no disponible, usando modo básico');
            }
        } catch (error) {
            console.error('Error al verificar disponibilidad del modo avanzado:', error);
            this.isAdvancedModeAvailable = false;
        }
    }
    
    /**
     * Carga la lista de modelos avanzados disponibles
     */
    async loadAdvancedModels() {
        try {
            const response = await fetch('/api/ml/available-models');
            const data = await response.json();
            
            if (data.success) {
                // Almacenar la información de modelos disponibles por tipo
                if (data.models) {
                    if (data.models.classification) {
                        this.advancedModels.classification = data.models.classification;
                    }
                    if (data.models.regression) {
                        this.advancedModels.regression = data.models.regression;
                    }
                    if (data.models.clustering) {
                        this.advancedModels.clustering = data.models.clustering;
                    }
                    if (data.models.timeseries) {
                        this.advancedModels.timeseries = data.models.timeseries;
                    }
                }
            }
        } catch (error) {
            console.error('Error al cargar modelos avanzados:', error);
        }
    }
    
    /**
     * Actualiza la interfaz para mostrar opciones de modelos avanzados
     */
    showAdvancedModeOptions() {
        // Añadir indicador de modo avanzado
        const advancedBadge = document.createElement('span');
        advancedBadge.classList.add('badge', 'badge-success', 'ml-2');
        advancedBadge.innerText = 'Avanzado';
        
        const title = document.querySelector('h2.card-title');
        if (title) {
            title.appendChild(advancedBadge);
        }
        
        // Añadir botón para modo avanzado
        const advancedModeToggle = document.createElement('button');
        advancedModeToggle.id = 'toggle-advanced-mode';
        advancedModeToggle.classList.add('btn', 'btn-sm', 'btn-outline-primary', 'ml-2');
        advancedModeToggle.innerText = 'Usar Modelos Avanzados';
        
        const modelTypeContainer = document.querySelector('#model-type').parentElement;
        modelTypeContainer.appendChild(advancedModeToggle);
        
        // Añadir evento para cambiar entre modo básico y avanzado
        advancedModeToggle.addEventListener('click', () => {
            this.toggleAdvancedMode();
        });
        
        // Añadir sección para clustering avanzado
        this.createAdvancedClusteringSection();
        
        // Añadir sección para análisis de series temporales
        this.createTimeSeriesSection();
        
        // Añadir sección para visualizaciones avanzadas
        this.createAdvancedVisualizationsSection();
    }
    
    /**
     * Actualiza la interfaz según la disponibilidad del modo avanzado
     */
    updateUIForAdvancedMode() {
        const advancedElements = document.querySelectorAll('.advanced-feature');
        
        // Mostrar u ocultar elementos avanzados
        advancedElements.forEach(element => {
            element.style.display = this.isAdvancedModeAvailable ? 'block' : 'none';
        });
        
        // Actualizar textos informativos
        const apiVersionBadge = document.getElementById('api-version-badge');
        if (apiVersionBadge) {
            apiVersionBadge.textContent = this.isAdvancedModeAvailable ? 'API v2.0' : 'API v1.0';
            apiVersionBadge.className = this.isAdvancedModeAvailable ? 
                'badge bg-success' : 'badge bg-secondary';
        }
        
        // Añadir pestaña de modelos avanzados si está disponible
        if (this.isAdvancedModeAvailable) {
            this.loadAdvancedModelsList();
        }
    }
    
    /**
     * Carga la lista de modelos avanzados disponibles
     */
    async loadAdvancedModelsList() {
        try {
            const response = await fetch('/api/ml/available-models');
            const data = await response.json();
            
            if (data.availableTrainingModels) {
                // Guardar los modelos disponibles
                this.advancedModels.classification = data.availableTrainingModels.classification || [];
                this.advancedModels.regression = data.availableTrainingModels.regression || [];
                
                // Actualizar los selectores de modelos
                this.updateModelSelectors();
            }
        } catch (error) {
            console.error('Error al cargar lista de modelos avanzados:', error);
        }
    }
    
    /**
     * Actualiza los selectores de modelos con los modelos disponibles
     */
    updateModelSelectors() {
        const modelSelectors = document.querySelectorAll('.model-selector');
        
        modelSelectors.forEach(selector => {
            const modelType = selector.dataset.modelType;
            const currentValue = selector.value;
            
            // Limpiar selector
            selector.innerHTML = '';
            
            // Añadir modelos según el tipo
            let modelsList = [];
            
            if (modelType === 'classification') {
                modelsList = [
                    ...this.getBasicClassificationModels(),
                    ...this.advancedModels.classification
                ];
            } else if (modelType === 'regression') {
                modelsList = [
                    ...this.getBasicRegressionModels(),
                    ...this.advancedModels.regression
                ];
            } else if (modelType === 'clustering') {
                modelsList = this.getClusteringModels();
            } else if (modelType === 'timeseries') {
                modelsList = this.getTimeSeriesModels();
            }
            
            // Añadir opciones
            modelsList.forEach(model => {
                const option = document.createElement('option');
                option.value = model.value || model;
                option.textContent = model.label || this.formatModelName(model);
                option.dataset.isAdvanced = model.isAdvanced || false;
                selector.appendChild(option);
            });
            
            // Intentar restaurar valor
            if (currentValue && selector.querySelector(`option[value="${currentValue}"]`)) {
                selector.value = currentValue;
            }
        });
    }
    
    /**
     * Obtiene la lista de modelos básicos de clasificación
     */
    getBasicClassificationModels() {
        return [
            { value: 'random_forest', label: 'Random Forest', isAdvanced: false },
            { value: 'svm', label: 'Support Vector Machine', isAdvanced: false },
            { value: 'logistic_regression', label: 'Regresión Logística', isAdvanced: false },
            { value: 'knn', label: 'K-Nearest Neighbors', isAdvanced: false },
            { value: 'decision_tree', label: 'Árbol de Decisión', isAdvanced: false }
        ];
    }
    
    /**
     * Obtiene la lista de modelos básicos de regresión
     */
    getBasicRegressionModels() {
        return [
            { value: 'linear_regression', label: 'Regresión Lineal', isAdvanced: false },
            { value: 'ridge', label: 'Ridge', isAdvanced: false },
            { value: 'lasso', label: 'Lasso', isAdvanced: false },
            { value: 'svr', label: 'SVR', isAdvanced: false },
            { value: 'random_forest_regressor', label: 'Random Forest', isAdvanced: false }
        ];
    }
    
    /**
     * Obtiene la lista de modelos de clustering
     */
    getClusteringModels() {
        return [
            { value: 'kmeans', label: 'K-Means', isAdvanced: false },
            { value: 'dbscan', label: 'DBSCAN', isAdvanced: false },
            { value: 'agglomerative', label: 'Agrupamiento Jerárquico', isAdvanced: true },
            { value: 'gaussian_mixture', label: 'Gaussian Mixture', isAdvanced: true },
            { value: 'spectral_clustering', label: 'Agrupamiento Espectral', isAdvanced: true },
            { value: 'birch', label: 'BIRCH', isAdvanced: true }
        ];
    }
    
    /**
     * Obtiene la lista de modelos de series temporales
     */
    getTimeSeriesModels() {
        return [
            { value: 'arima', label: 'ARIMA', isAdvanced: false },
            { value: 'sarima', label: 'SARIMA (Estacional)', isAdvanced: true },
            { value: 'prophet', label: 'Prophet (Facebook)', isAdvanced: true },
            { value: 'auto_arima', label: 'Auto ARIMA', isAdvanced: true },
            { value: 'lstm', label: 'LSTM (Deep Learning)', isAdvanced: true }
        ];
    }
    
    /**
     * Da formato al nombre del modelo para mostrar
     */
    formatModelName(modelName) {
        return modelName
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    /**
     * Configura todos los listeners de eventos
     */
    setupEventListeners() {
        // Cambio de dataset
        this.datasetSelect.addEventListener('change', () => this.onDatasetChange());
        
        // Cambio de tipo de modelo
        this.modelTypeSelect.addEventListener('change', (e) => this.onModelTypeChange(e.target.value));
        
        // Botón de entrenamiento
        this.trainButton.addEventListener('click', (e) => this.onTrainModel(e));
        
        // Botón de actualizar datasets
        this.refreshDatasetsButton.addEventListener('click', () => this.loadDatasets());
        
        // Formulario de subida de dataset
        this.uploadDatasetForm.addEventListener('submit', (e) => this.onUploadDataset(e));
        
        // Formulario de predicción
        this.predictionForm.addEventListener('submit', (e) => this.onSubmitPrediction(e));
        
        // Cambio en el tamaño de test
        this.testSizeRange.addEventListener('input', (e) => {
            document.getElementById('test-size-display').textContent = `${e.target.value}%`;
        });
        
        // Botón de mostrar/ocultar parámetros
        this.toggleParamsButton.addEventListener('click', () => this.toggleParameters());
        
        // Configurar las pestañas de resultados
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchResultsTab(btn.dataset.tab));
        });
    }
    
    /**
     * Maneja el cambio de dataset seleccionado
     */
    async onDatasetChange() {
        const datasetId = this.datasetSelect.value;  
        
        try {
            // Mostrar indicador de carga
            this.setLoading(true, 'Cargando información del dataset...');
            
            // Obtener información del dataset desde la API
            const response = await fetch(`/api/datasets/${datasetId}`);
            
            if (!response.ok) {
                throw new Error(`Error al cargar el dataset: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Error al cargar el dataset');
            }
            
            this.selectedDataset = data;
            
            // Actualizar la información y vista previa del dataset
            this.updateDatasetInfo(data);
            
            // Actualizar campos del predictor basados en las características
            this.updatePredictorFields(data.features);
            
            // Ocultar indicador de carga
            this.setLoading(false);
            
        } catch (error) {
            console.error('Error al cargar el dataset:', error);
            showMessage(`Error: ${error.message}`, 'danger');
            this.setLoading(false);
        }
    }
    
    /**
     * Maneja el cambio de tipo de modelo
     * @param {string} modelType - El tipo de modelo seleccionado
     */
    onModelTypeChange(modelType) {
        this.currentModelType = modelType;
        
        // Actualizar UI según el tipo de modelo
        document.querySelectorAll('.model-type-container').forEach(container => {
            container.style.display = 'none';
        });
        
        const selectedContainer = document.getElementById(`${modelType}-container`);
        if (selectedContainer) {
            selectedContainer.style.display = 'block';
        }
        
        // Actualizar opciones específicas del tipo de modelo
        this.updateModelSpecificOptions(modelType);
    }
    
    /**
     * Actualiza opciones específicas según el tipo de modelo
     */
    updateModelSpecificOptions(modelType) {
        // Ocultar todos los contenedores de opciones específicas
        document.querySelectorAll('.model-specific-options').forEach(container => {
            container.style.display = 'none';
        });
        
        // Mostrar opciones específicas del modelo seleccionado
        const optionsContainer = document.getElementById(`${modelType}-options`);
        if (optionsContainer) {
            optionsContainer.style.display = 'block';
        }
        
        // Actualizar selectores para este tipo de modelo
        this.updateModelSelectors();
    }
    
    /**
     * Inicia el proceso de entrenamiento del modelo
     * @param {Event} e - El evento del click
     */
    async onTrainModel(e) {
        e.preventDefault();
        
        if (!this.selectedDataset) {
            showMessage('Por favor, seleccione un dataset primero', 'warning');
            return;
        }
        
        // Obtener los modelos seleccionados
        const modelCheckboxes = document.querySelectorAll(`#${this.currentModelType}-models input[type="checkbox"]:checked`);
        
        if (modelCheckboxes.length === 0) {
            showMessage('Por favor, seleccione al menos un modelo para entrenar', 'warning');
            return;
        }
        
        this.selectedModels = Array.from(modelCheckboxes).map(cb => cb.value);
        
        // Recopilar parámetros de entrenamiento
        const trainingParams = {
            datasetId: this.datasetSelect.value,
            modelType: this.currentModelType,
            models: this.selectedModels,
            testSize: parseInt(this.testSizeRange.value) / 100,
            cvFolds: parseInt(document.getElementById('cv-folds').value),
            scaling: document.getElementById('use-scaling').checked
        };
        
        // Mostrar sección de resultados y barra de progreso
        const trainingResults = document.getElementById('training-results');
        const statusIndicator = trainingResults.querySelector('.status-indicator');
        const resultsContainer = trainingResults.querySelector('.results-container');
        
        trainingResults.classList.remove('hidden');
        statusIndicator.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        
        try {
            this.setLoading(true, 'Entrenando modelos. Esto puede tardar unos minutos...');
            
            // Llamar a la API para entrenar los modelos
            const response = await fetch('/api/models/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(trainingParams)
            });
            
            if (!response.ok) {
                throw new Error(`Error en la solicitud: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.error || 'Error al entrenar los modelos');
            }
            
            // Guardar los resultados del entrenamiento
            this.trainedModels = result.results;
            
            // Actualizar el progreso al 100%
            const progressEl = document.getElementById('progress');
            progressEl.textContent = '100%';
            
            // Mostrar resultados
            statusIndicator.classList.add('hidden');
            resultsContainer.classList.remove('hidden');
            
            // Actualizar la tabla de métricas con los resultados reales
            this.updateMetricsTable();
            
            // Inicializar las gráficas con los resultados del entrenamiento
            this.initializeResultCharts();
            
            // Actualizar el selector de modelos para predicción
            this.updatePredictionModelSelector();
            
            this.setLoading(false);
            showMessage('Modelos entrenados correctamente', 'success');
            
        } catch (error) {
            console.error('Error durante el entrenamiento:', error);
            showMessage(`Error: ${error.message}`, 'danger');
            statusIndicator.classList.add('hidden');
            this.setLoading(false);
        }
    }
    
    /**
     * Entrena un modelo avanzado
     */
    async trainAdvancedModel() {
        // Verificar que se seleccionó un dataset y modelo
        if (!this.selectedDataset) {
            showToast('Error', 'Selecciona un dataset primero', 'error');
            return;
        }
        
        const modelType = this.currentModelType;
        const modelSelect = document.querySelector(`.model-selector[data-model-type="${modelType}"]`);
        
        if (!modelSelect || !modelSelect.value) {
            showToast('Error', 'Selecciona un modelo para entrenar', 'error');
            return;
        }
        
        const modelName = modelSelect.value;
        const testSize = this.testSizeRange ? parseFloat(this.testSizeRange.value) : 0.2;
        const optimizeHyperparams = document.getElementById('optimize-hyperparams')?.checked || false;
        
        // Recoger parámetros adicionales según el tipo de modelo
        const modelParams = this.getModelParameters(modelType, modelName);
        
        try {
            // Mostrar mensaje de carga
            showLoading('Entrenando modelo...');
            
            // Enviar solicitud de entrenamiento
            const response = await fetch('/api/ml/train-model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    dataset_id: this.selectedDataset,
                    model_type: modelType,
                    model_name: modelName,
                    test_size: testSize,
                    optimize_hyperparams: optimizeHyperparams,
                    model_params: modelParams
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            // Ocultar mensaje de carga
            hideLoading();
            
            // Mostrar resultados
            this.displayTrainingResults(result);
            
            showToast('Éxito', 'Modelo entrenado correctamente', 'success');
        } catch (error) {
            hideLoading();
            console.error('Error al entrenar modelo avanzado:', error);
            showToast('Error', `Error al entrenar modelo: ${error.message}`, 'error');
        }
    }
    
    /**
     * Obtiene los parámetros específicos del modelo seleccionado
     */
    getModelParameters(modelType, modelName) {
        const params = {};
        
        // Obtener parámetros del formulario según el tipo de modelo
        const paramContainer = document.getElementById(`${modelName}-params`);
        if (!paramContainer) return params;
        
        // Buscar todos los inputs en el contenedor
        const inputs = paramContainer.querySelectorAll('input, select');
        
        inputs.forEach(input => {
            const paramName = input.name;
            if (!paramName) return;
            
            let value;
            
            if (input.type === 'checkbox') {
                value = input.checked;
            } else if (input.type === 'number') {
                value = parseFloat(input.value);
            } else if (input.type === 'select-multiple') {
                value = Array.from(input.selectedOptions).map(option => option.value);
            } else {
                value = input.value;
            }
            
            // Asignar valor al parámetro
            params[paramName] = value;
        });
        
        return params;
    }
    
    /**
     * Muestra los resultados del entrenamiento
     */
    displayTrainingResults(result) {
        // Crear contenedor de resultados si no existe
        let resultsContainer = document.getElementById('training-results');
        if (!resultsContainer) {
            resultsContainer = document.createElement('div');
            resultsContainer.id = 'training-results';
            resultsContainer.className = 'mt-4 p-3 border rounded';
            document.getElementById('model-results-section').appendChild(resultsContainer);
        }
        
        // Limpiar contenedor
        resultsContainer.innerHTML = '';
        
        // Título
        const title = document.createElement('h4');
        title.textContent = `Resultados de Entrenamiento: ${this.formatModelName(result.model_name)}`;
        resultsContainer.appendChild(title);
        
        // Información del modelo
        const modelInfo = document.createElement('div');
        modelInfo.className = 'model-info mb-3';
        modelInfo.innerHTML = `
            <p><strong>ID del Modelo:</strong> ${result.model_id || 'No disponible'}</p>
            <p><strong>Dataset:</strong> ${this.selectedDataset}</p>
            <p><strong>Tipo de Modelo:</strong> ${this.currentModelType}</p>
        `;
        resultsContainer.appendChild(modelInfo);
        
        // Métricas
        if (result.metrics) {
            const metricsCard = document.createElement('div');
            metricsCard.className = 'card mb-3';
            
            let metricsHtml = '<div class="card-header"><h5>Métricas</h5></div><div class="card-body">';
            
            Object.entries(result.metrics).forEach(([metricName, value]) => {
                const formattedName = metricName
                    .split('_')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(' ');
                
                const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
                
                metricsHtml += `<p><strong>${formattedName}:</strong> ${formattedValue}</p>`;
            });
            
            metricsHtml += '</div>';
            metricsCard.innerHTML = metricsHtml;
            resultsContainer.appendChild(metricsCard);
        }
        
        // Importancia de características
        if (result.feature_importance) {
            const featureCard = document.createElement('div');
            featureCard.className = 'card mb-3';
            featureCard.innerHTML = '<div class="card-header"><h5>Importancia de Características</h5></div>';
            
            const featureBody = document.createElement('div');
            featureBody.className = 'card-body';
            
            // Crear gráfico de barras para importancia de características
            const chartCanvas = document.createElement('canvas');
            chartCanvas.id = 'feature-importance-chart';
            featureBody.appendChild(chartCanvas);
            
            featureCard.appendChild(featureBody);
            resultsContainer.appendChild(featureCard);
            
            // Procesar datos para el gráfico
            const featureNames = [];
            const importanceValues = [];
            
            Object.entries(result.feature_importance).forEach(([feature, value]) => {
                featureNames.push(feature);
                importanceValues.push(typeof value === 'object' ? value.importance_mean : value);
            });
            
            // Crear gráfico
            new Chart(chartCanvas, {
                type: 'bar',
                data: {
                    labels: featureNames,
                    datasets: [{
                        label: 'Importancia',
                        data: importanceValues,
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        title: { display: true, text: 'Importancia de Características' }
                    }
                }
            });
        }
        
        // Botones para acciones adicionales
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'd-flex justify-content-between mt-3';
        
        // Botón para evaluar modelo
        const evaluateButton = document.createElement('button');
        evaluateButton.textContent = 'Evaluar Modelo';
        evaluateButton.className = 'btn btn-info me-2';
        evaluateButton.onclick = () => this.evaluateModel(result.model_id);
        
        // Botón para guardar modelo
        const saveButton = document.createElement('button');
        saveButton.textContent = 'Guardar Modelo';
        saveButton.className = 'btn btn-success me-2';
        saveButton.onclick = () => this.saveModel(result.model_id);
        
        // Botón para descargar modelo
        const downloadButton = document.createElement('button');
        downloadButton.textContent = 'Descargar Modelo';
        downloadButton.className = 'btn btn-secondary';
        downloadButton.onclick = () => this.downloadModel(result.model_id);
        
        actionsDiv.appendChild(evaluateButton);
        actionsDiv.appendChild(saveButton);
        actionsDiv.appendChild(downloadButton);
        
        resultsContainer.appendChild(actionsDiv);
    }
    
    /**
     * Evalúa un modelo entrenado
     */
    async evaluateModel(modelId) {
        try {
            showLoading('Evaluando modelo...');
            
            const response = await fetch(`/api/ml/evaluate-model/${modelId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    evaluation_type: 'cross_validation',
                    params: {
                        cv_folds: 5
                    }
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            hideLoading();
            
            // Mostrar resultados de evaluación
            this.displayEvaluationResults(result);
        } catch (error) {
            hideLoading();
            console.error('Error al evaluar modelo:', error);
            showToast('Error', `Error al evaluar modelo: ${error.message}`, 'error');
        }
    }
    
    /**
     * Muestra los resultados de la evaluación del modelo
     */
    displayEvaluationResults(result) {
        // Implementar visualización de resultados de evaluación
    }
    
    /**
     * Guarda un modelo entrenado
     */
    async saveModel(modelId) {
        try {
            const modelName = prompt('Ingresa un nombre para guardar el modelo:');
            if (!modelName) return;
            
            showLoading('Guardando modelo...');
            
            const response = await fetch(`/api/ml/save-model/${modelId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: modelName
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            hideLoading();
            showToast('Éxito', 'Modelo guardado correctamente', 'success');
            
            // Actualizar lista de modelos guardados
            this.loadSavedModels();
        } catch (error) {
            hideLoading();
            console.error('Error al guardar modelo:', error);
            showToast('Error', `Error al guardar modelo: ${error.message}`, 'error');
        }
    }
    
    /**
     * Descarga un modelo entrenado
     */
    async downloadModel(modelId) {
        try {
            showLoading('Preparando descarga...');
            
            const response = await fetch(`/api/ml/download-model/${modelId}`);
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Error al descargar modelo');
            }
            
            hideLoading();
            
            // Crear enlace para descarga
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `model_${modelId}.joblib`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            hideLoading();
            console.error('Error al descargar modelo:', error);
            showToast('Error', `Error al descargar modelo: ${error.message}`, 'error');
        }
    }
    
    /**
     * Realiza clustering en el dataset seleccionado
     */
    async performClustering() {
        if (!this.selectedDataset) {
            showToast('Error', 'Selecciona un dataset primero', 'error');
            return;
        }
        
        const algorithmSelect = document.querySelector('.model-selector[data-model-type="clustering"]');
        if (!algorithmSelect || !algorithmSelect.value) {
            showToast('Error', 'Selecciona un algoritmo de clustering', 'error');
            return;
        }
        
        const algorithm = algorithmSelect.value;
        const nClusters = document.getElementById('n-clusters')?.value || 3;
        
        try {
            showLoading('Realizando clustering...');
            
            const response = await fetch('/api/ml/clustering', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    dataset_id: this.selectedDataset,
                    algorithm: algorithm,
                    n_clusters: parseInt(nClusters),
                    dimensionality_reduction: document.getElementById('dimensionality-reduction')?.value
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            hideLoading();
            
            // Mostrar resultados de clustering
            this.displayClusteringResults(result);
        } catch (error) {
            hideLoading();
            console.error('Error al realizar clustering:', error);
            showToast('Error', `Error al realizar clustering: ${error.message}`, 'error');
        }
    }
    
    /**
     * Muestra los resultados del clustering
     */
    displayClusteringResults(result) {
        // Implementar visualización de resultados de clustering
    }
    
    /**
     * Actualiza la tabla de métricas con los resultados del entrenamiento
     */
    updateMetricsTable() {
        if (!this.trainedModels || !this.trainedModels.models_performance) {
            return;
        }
        
        const tbody = document.querySelector('.metrics-table tbody');
        tbody.innerHTML = '';
        
        const modelType = this.trainedModels.model_type;
        const bestModel = this.trainedModels.best_model;
        
        // Crear filas para cada modelo entrenado
        for (const modelName of this.trainedModels.models_trained) {
            const performance = this.trainedModels.models_performance[modelName];
            const trainingTime = this.trainedModels.training_time[modelName];
            
            if (!performance || performance.error) continue;
            
            const row = document.createElement('tr');
            if (modelName === bestModel) {
                row.classList.add('best-model');
            }
            
            // Nombre del modelo (primera columna)
            const nameCell = document.createElement('td');
            nameCell.textContent = this.getModelDisplayName(modelName);
            row.appendChild(nameCell);
            
            // Métricas específicas según el tipo de modelo
            if (modelType === 'classification') {
                // Precision
                const precisionCell = document.createElement('td');
                precisionCell.textContent = (performance.precision * 100).toFixed(1) + '%';
                row.appendChild(precisionCell);
                
                // Recall
                const recallCell = document.createElement('td');
                recallCell.textContent = (performance.recall * 100).toFixed(1) + '%';
                row.appendChild(recallCell);
                
                // F1-Score
                const f1Cell = document.createElement('td');
                f1Cell.textContent = (performance.f1_score * 100).toFixed(1) + '%';
                row.appendChild(f1Cell);
            } else if (modelType === 'regression') {
                // R2 Score
                const r2Cell = document.createElement('td');
                r2Cell.textContent = performance.r2_score.toFixed(3);
                row.appendChild(r2Cell);
                
                // RMSE
                const rmseCell = document.createElement('td');
                rmseCell.textContent = performance.rmse.toFixed(3);
                row.appendChild(rmseCell);
                
                // MAE
                const maeCell = document.createElement('td');
                maeCell.textContent = performance.mae.toFixed(3);
                row.appendChild(maeCell);
            } else if (modelType === 'clustering') {
                // Silhouette Score
                const silhouetteCell = document.createElement('td');
                silhouetteCell.textContent = performance.silhouette_score.toFixed(3);
                row.appendChild(silhouetteCell);
                
                // Número de clusters
                const clustersCell = document.createElement('td');
                clustersCell.textContent = performance.n_clusters;
                row.appendChild(clustersCell);
            }
            
            // Tiempo de entrenamiento
            const timeCell = document.createElement('td');
            timeCell.textContent = trainingTime.toFixed(2) + 's';
            row.appendChild(timeCell);
            
            tbody.appendChild(row);
        }
    }
    
    /**
     * Actualiza el selector de modelos para predicción
     */
    updatePredictionModelSelector() {
        if (!this.trainedModels || !this.trainedModels.models_trained) {
            return;
        }
        
        const selector = document.getElementById('prediction-model');
        selector.innerHTML = '';
        
        // Opción para el mejor modelo
        const bestOption = document.createElement('option');
        bestOption.value = 'best';
        bestOption.textContent = `Mejor modelo (${this.getModelDisplayName(this.trainedModels.best_model)})`;
        selector.appendChild(bestOption);
        
        // Opciones para todos los modelos entrenados
        for (const modelName of this.trainedModels.models_trained) {
            const performance = this.trainedModels.models_performance[modelName];
            if (performance && !performance.error) {
                const option = document.createElement('option');
                option.value = modelName;
                option.textContent = this.getModelDisplayName(modelName);
                selector.appendChild(option);
            }
        }
    }
    
    /**
     * Envía los datos para obtener una predicción
     * @param {Event} e - El evento del formulario
     */
    async onSubmitPrediction(e) {
        e.preventDefault();
        
        if (!this.selectedDataset) {
            showMessage('Por favor, seleccione un dataset y entrene un modelo primero', 'warning');
            return;
        }
        
        if (!this.trainedModels || !this.trainedModels.best_model) {
            showMessage('Por favor, entrene al menos un modelo primero', 'warning');
            return;
        }
        
        // Recopilar los valores de las características
        const features = [];
        const featureInputs = document.querySelectorAll('.feature-input input');
        
        featureInputs.forEach(input => {
            features.push(parseFloat(input.value));
        });
        
        // Modelo seleccionado para la predicción
        const modelSelect = document.getElementById('prediction-model');
        let selectedModel = modelSelect.value;
        
        // Si se selecciona "mejor modelo", usar el mejor modelo del entrenamiento
        if (selectedModel === 'best') {
            selectedModel = this.trainedModels.best_model;
        }
        
        try {
            this.setLoading(true, 'Realizando predicción...');
            
            // Llamar a la API para realizar la predicción
            const response = await fetch('/api/models/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    modelName: selectedModel,
                    datasetId: this.datasetSelect.value,
                    features: features
                })
            });
            
            if (!response.ok) {
                throw new Error(`Error en la solicitud: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.error || 'Error al realizar la predicción');
            }
            
            // Mostrar resultado de la predicción
            const predictionResult = document.querySelector('.prediction-result');
            predictionResult.classList.remove('hidden');
            
            // Actualizar la predicción en la interfaz
            this.updatePredictionResult(result.prediction);
            
            this.setLoading(false);
            
        } catch (error) {
            console.error('Error al realizar la predicción:', error);
            showMessage(`Error: ${error.message}`, 'danger');
            this.setLoading(false);
        }
    }
    
    /**
     * Actualiza los elementos de la UI para mostrar el resultado de la predicción
     * @param {Object} prediction - El resultado de la predicción
     */
    updatePredictionResult(prediction) {
        if (!prediction) return;
        
        // Obtener el elemento donde mostrar la clase predicha
        const predictionClass = document.querySelector('.prediction-class');
        
        // Obtener el contenedor para las probabilidades
        const predictionProbs = document.querySelector('.prediction-probs');
        
        // Limpiar el contenedor de probabilidades
        predictionProbs.innerHTML = '';
        
        // Si estamos en una tarea de clasificación y hay probabilidades
        if (this.trainedModels.model_type === 'classification' && prediction.probabilities) {
            // Mostrar la clase predicha
            const pred = prediction.prediction;
            predictionClass.textContent = pred;
            
            // Mostrar las probabilidades
            const probs = prediction.probabilities;
            
            // Crear elementos para cada probabilidad
            for (const [cls, prob] of Object.entries(probs)) {
                const probValue = parseFloat(prob);
                const percentProb = (probValue * 100).toFixed(1);
                
                // Crear contenedor para esta probabilidad
                const probItem = document.createElement('div');
                probItem.className = 'prob-item';
                
                // Etiqueta de la clase
                const probLabel = document.createElement('span');
                probLabel.className = 'prob-label';
                probLabel.textContent = `${cls}:`;
                probItem.appendChild(probLabel);
                
                // Barra de probabilidad
                const probBar = document.createElement('div');
                probBar.className = 'prob-bar';
                
                // Relleno de la barra
                const probFill = document.createElement('div');
                probFill.className = 'prob-fill';
                probFill.style.width = `${percentProb}%`;
                probBar.appendChild(probFill);
                
                // Valor numérico
                const probValueElem = document.createElement('span');
                probValueElem.className = 'prob-value';
                probValueElem.textContent = `${percentProb}%`;
                probBar.appendChild(probValueElem);
                
                probItem.appendChild(probBar);
                predictionProbs.appendChild(probItem);
            }
        } else {
            // Para regresión, simplemente mostrar el valor predicho
            predictionClass.textContent = prediction.prediction.toFixed(4);
            
            // Para regresión no hay probabilidades, ocultar ese contenedor
            predictionProbs.style.display = 'none';
        }
    }
    
    /**
     * Maneja la subida de un nuevo dataset
     * @param {Event} e - El evento del formulario
     */
    async onUploadDataset(e) {
        e.preventDefault();
        
        const datasetName = document.getElementById('dataset-name').value;
        const datasetFile = document.getElementById('dataset-file').files[0];
        const targetColumn = document.getElementById('target-column').value;
        const description = document.getElementById('dataset-description').value;
        
        if (!datasetName || !datasetFile) {
            showMessage('Por favor, complete los campos requeridos', 'warning');
            return;
        }
        
        const formData = new FormData();
        formData.append('name', datasetName);
        formData.append('file', datasetFile);
        
        if (targetColumn) {
            formData.append('target_column', targetColumn);
        }
        
        if (description) {
            formData.append('description', description);
        }
        
        try {
            this.setLoading(true, 'Subiendo dataset...');
            
            // Enviar el dataset al servidor
            const response = await fetch('/api/datasets/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Error en la solicitud: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.error || 'Error al subir el dataset');
            }
            
            // Mostrar mensaje de éxito
            showMessage(`Dataset "${datasetName}" subido correctamente`, 'success');
            
            // Actualizar la lista de datasets disponibles
            await this.loadDatasets();
            
            // Seleccionar el nuevo dataset
            const newDatasetId = result.dataset.dataset_id;
            if (newDatasetId && this.datasetSelect) {
                this.datasetSelect.value = newDatasetId;
                // Disparar el evento change para cargar los datos del nuevo dataset
                this.datasetSelect.dispatchEvent(new Event('change'));
            }
            
            // Limpiar el formulario
            this.uploadDatasetForm.reset();
            
            this.setLoading(false);
            
        } catch (error) {
            console.error('Error al subir el dataset:', error);
            showMessage(`Error: ${error.message}`, 'danger');
            this.setLoading(false);
        }
    }
    
    /**
     * Muestra u oculta los parámetros avanzados de los modelos
     */
    toggleParameters() {
        const paramsContainer = document.getElementById('params-container');
        const isHidden = paramsContainer.classList.toggle('hidden');
        
        this.toggleParamsButton.innerHTML = isHidden ? 
            '<i class="fas fa-sliders-h"></i> Mostrar Parámetros' : 
            '<i class="fas fa-sliders-h"></i> Ocultar Parámetros';
    }
    
    /**
     * Cambia entre las pestañas de resultados
     * @param {string} tabName - Nombre de la pestaña a mostrar
     */
    switchResultsTab(tabName) {
        // Desactivar todas las pestañas
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        // Activar pestaña actual
        document.querySelector(`.tab-btn[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }
    
    /**
     * Carga la lista de datasets disponibles desde la API
     */
    async loadDatasets() {
        try {
            this.setLoading(true, 'Cargando datasets...');
            
            // Llamar a la API para obtener la lista de datasets
            const response = await fetch('/api/datasets/list');
            
            if (!response.ok) {
                throw new Error(`Error en la solicitud: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.error || 'Error al cargar los datasets');
            }
            
            const datasets = result.datasets;
            
            // Guardar la selección actual
            const currentValue = this.datasetSelect ? this.datasetSelect.value : '';
            
            // Limpiar el selector
            this.datasetSelect.innerHTML = '';
            
            // Añadir opciones al selector
            datasets.forEach(dataset => {
                const option = document.createElement('option');
                option.value = dataset.id;
                option.textContent = dataset.name;
                this.datasetSelect.appendChild(option);
            });
            
            // Restaurar la selección si existe
            if (datasets.some(d => d.id === currentValue)) {
                this.datasetSelect.value = currentValue;
            } else {
                // Seleccionar el primer dataset por defecto
                this.datasetSelect.value = datasets.length > 0 ? datasets[0].id : '';
            }
            
            // Disparar el evento change para cargar los datos iniciales
            this.datasetSelect.dispatchEvent(new Event('change'));
            
            this.setLoading(false);
            
        } catch (error) {
            console.error('Error al cargar los datasets:', error);
            showMessage(`Error: ${error.message}`, 'danger');
            this.setLoading(false);
        }
    }
    
    /**
     * Actualiza la información y vista previa del dataset
     * @param {Object} data - Datos del dataset
     */
    updateDatasetInfo(data) {
        // Para esta demostración, usamos datos simulados
        const datasetInfo = {
            name: data.id === 'iris' ? 'Iris' : data.id.charAt(0).toUpperCase() + data.id.slice(1),
            rows: data.id === 'iris' ? 150 : (data.id === 'wine' ? 178 : (data.id === 'breast_cancer' ? 569 : 442)),
            columns: data.id === 'iris' ? 5 : (data.id === 'wine' ? 14 : (data.id === 'breast_cancer' ? 31 : 11)),
            target: data.id === 'iris' ? 'species' : (data.id === 'wine' ? 'class' : (data.id === 'breast_cancer' ? 'target' : 'outcome'))
        };
        
        // Actualizar elementos en la UI
        document.getElementById('dataset-info-name').textContent = datasetInfo.name;
        document.getElementById('dataset-info-rows').textContent = datasetInfo.rows;
        document.getElementById('dataset-info-cols').textContent = datasetInfo.columns;
        document.getElementById('dataset-info-target').textContent = datasetInfo.target;
        
        // En un caso real, aquí actualizaríamos la tabla de vista previa
        this.updateDataPreview(data.id);
    }
    
    /**
     * Actualiza la vista previa de los datos del dataset
     * @param {string} datasetId - ID del dataset
     */
    updateDataPreview(datasetId) {
        // Esta es una simulación. En un caso real, obtendríamos estos datos de la API
        const previewData = {
            iris: {
                columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
                data: [
                    [5.1, 3.5, 1.4, 0.2, 'setosa'],
                    [4.9, 3.0, 1.4, 0.2, 'setosa'],
                    [4.7, 3.2, 1.3, 0.2, 'setosa'],
                    [4.6, 3.1, 1.5, 0.2, 'setosa'],
                    [5.0, 3.6, 1.4, 0.2, 'setosa']
                ]
            },
            wine: {
                columns: ['alcohol', 'malic_acid', 'ash', 'alcalinity', 'magnesium', 'class'],
                data: [
                    [14.23, 1.71, 2.43, 15.6, 127, 0],
                    [13.20, 1.78, 2.14, 11.2, 100, 0],
                    [13.16, 2.36, 2.67, 18.6, 101, 0],
                    [14.37, 1.95, 2.50, 16.8, 113, 0],
                    [13.24, 2.59, 2.87, 21.0, 118, 0]
                ]
            },
            breast_cancer: {
                columns: ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'target'],
                data: [
                    [17.99, 10.38, 122.8, 1001.0, 0],
                    [20.57, 17.77, 132.9, 1326.0, 0],
                    [19.69, 21.25, 130.0, 1203.0, 0],
                    [11.42, 20.38, 77.58, 386.1, 1],
                    [20.29, 14.34, 135.1, 1297.0, 0]
                ]
            },
            diabetes: {
                columns: ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'outcome'],
                data: [
                    [6, 148, 72, 35, 1],
                    [1, 85, 66, 29, 0],
                    [8, 183, 64, 0, 1],
                    [1, 89, 66, 23, 0],
                    [0, 137, 40, 35, 1]
                ]
            }
        };
        
        const preview = previewData[datasetId] || previewData.iris;
        
        // Actualizar encabezados de la tabla
        const tableHead = document.querySelector('#data-preview thead tr');
        tableHead.innerHTML = '';
        preview.columns.forEach(column => {
            const th = document.createElement('th');
            th.textContent = column;
            tableHead.appendChild(th);
        });
        
        // Actualizar filas de datos
        const tableBody = document.querySelector('#data-preview tbody');
        tableBody.innerHTML = '';
        
        preview.data.forEach(row => {
            const tr = document.createElement('tr');
            
            row.forEach(cell => {
                const td = document.createElement('td');
                td.textContent = cell;
                tr.appendChild(td);
            });
            
            tableBody.appendChild(tr);
        });
    }
    
    /**
     * Actualiza los campos del predictor basados en las características disponibles
     * @param {string[]} features - Lista de características
     */
    updatePredictorFields(features) {
        // En un caso real, obtendríamos las características del dataset
        // Por ahora, simulamos con datos estáticos según el dataset
        let featuresFields;
        
        switch(this.datasetSelect.value) {
            case 'iris':
                featuresFields = [
                    {id: 'sepal_length', label: 'sepal_length:', value: 5.1},
                    {id: 'sepal_width', label: 'sepal_width:', value: 3.5},
                    {id: 'petal_length', label: 'petal_length:', value: 1.4},
                    {id: 'petal_width', label: 'petal_width:', value: 0.2}
                ];
                break;
            case 'wine':
                featuresFields = [
                    {id: 'alcohol', label: 'alcohol:', value: 13.0},
                    {id: 'malic_acid', label: 'malic_acid:', value: 2.0},
                    {id: 'ash', label: 'ash:', value: 2.5},
                    {id: 'alcalinity', label: 'alcalinity:', value: 15.0},
                    {id: 'magnesium', label: 'magnesium:', value: 110}
                ];
                break;
            case 'breast_cancer':
                featuresFields = [
                    {id: 'mean_radius', label: 'mean_radius:', value: 15.0},
                    {id: 'mean_texture', label: 'mean_texture:', value: 12.0},
                    {id: 'mean_perimeter', label: 'mean_perimeter:', value: 100.0},
                    {id: 'mean_area', label: 'mean_area:', value: 800.0}
                ];
                break;
            case 'diabetes':
                featuresFields = [
                    {id: 'pregnancies', label: 'pregnancies:', value: 3},
                    {id: 'glucose', label: 'glucose:', value: 120},
                    {id: 'blood_pressure', label: 'blood_pressure:', value: 70},
                    {id: 'skin_thickness', label: 'skin_thickness:', value: 20}
                ];
                break;
            default:
                featuresFields = [
                    {id: 'feature1', label: 'Feature 1:', value: 0},
                    {id: 'feature2', label: 'Feature 2:', value: 0}
                ];
        }
        
        const featuresContainer = document.querySelector('.features-inputs');
        featuresContainer.innerHTML = '';
        
        featuresFields.forEach(field => {
            const div = document.createElement('div');
            div.className = 'feature-input';
            
            const label = document.createElement('label');
            label.setAttribute('for', field.id);
            label.textContent = field.label;
            
            const input = document.createElement('input');
            input.type = 'number';
            input.id = field.id;
            input.step = '0.1';
            input.value = field.value;
            
            div.appendChild(label);
            div.appendChild(input);
            featuresContainer.appendChild(div);
        });
    }
    
    /**
     * Actualiza los parámetros disponibles para el tipo de modelo seleccionado
     * @param {string} modelType - El tipo de modelo
     */
    updateModelParameters(modelType) {
        // Implementación futura: cargar dinámicamente los parámetros
        // según el tipo de modelo y los modelos seleccionados
    }
    
    /**
     * Inicializa las gráficas para mostrar los resultados del modelo
     */
    initializeResultCharts() {
        // Para esta demostración, usamos datos estáticos
        // Matriz de confusión
        this.initConfusionMatrix();
        
        // Curva ROC
        this.initROCCurve();
        
        // Importancia de características
        this.initFeatureImportance();
    }
    
    /**
     * Inicializa la gráfica de matriz de confusión
     */
    initConfusionMatrix() {
        const ctxConfusion = document.getElementById('confusionMatrix');
        
        // Limpiar cualquier gráfico anterior
        if (ctxConfusion.__chart__) {
            ctxConfusion.__chart__.destroy();
        }
        
        // Crear nuevos datos de matriz de confusión según el dataset
        let labels, data;
        
        switch (this.datasetSelect.value) {
            case 'iris':
                labels = ['setosa', 'versicolor', 'virginica'];
                data = [
                    {x: 'setosa', y: 'setosa', v: 48},
                    {x: 'setosa', y: 'versicolor', v: 1},
                    {x: 'setosa', y: 'virginica', v: 0},
                    {x: 'versicolor', y: 'setosa', v: 2},
                    {x: 'versicolor', y: 'versicolor', v: 47},
                    {x: 'versicolor', y: 'virginica', v: 1},
                    {x: 'virginica', y: 'setosa', v: 0},
                    {x: 'virginica', y: 'versicolor', v: 2},
                    {x: 'virginica', y: 'virginica', v: 49}
                ];
                break;
            case 'wine':
                labels = ['Clase 0', 'Clase 1', 'Clase 2'];
                data = [
                    {x: 'Clase 0', y: 'Clase 0', v: 45},
                    {x: 'Clase 0', y: 'Clase 1', v: 2},
                    {x: 'Clase 0', y: 'Clase 2', v: 1},
                    {x: 'Clase 1', y: 'Clase 0', v: 3},
                    {x: 'Clase 1', y: 'Clase 1', v: 55},
                    {x: 'Clase 1', y: 'Clase 2', v: 2},
                    {x: 'Clase 2', y: 'Clase 0', v: 1},
                    {x: 'Clase 2', y: 'Clase 1', v: 3},
                    {x: 'Clase 2', y: 'Clase 2', v: 46}
                ];
                break;
            default:
                labels = ['Clase 0', 'Clase 1'];
                data = [
                    {x: 'Clase 0', y: 'Clase 0', v: 85},
                    {x: 'Clase 0', y: 'Clase 1', v: 10},
                    {x: 'Clase 1', y: 'Clase 0', v: 15},
                    {x: 'Clase 1', y: 'Clase 1', v: 90}
                ];
        }
        
        // Crear la matriz de confusión personalizada usando Chart.js
        const chart = new Chart(ctxConfusion, {
            type: 'matrix',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Matriz de Confusión',
                    data: data,
                    backgroundColor: (context) => {
                        const value = context.dataset.data[context.dataIndex].v;
                        const maxValue = Math.max(...data.map(d => d.v));
                        const alpha = value / maxValue;
                        return `rgba(67, 97, 238, ${alpha})`;
                    },
                    borderWidth: 1,
                    borderColor: 'rgba(255, 255, 255, 0.2)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            title: () => '',
                            label: (context) => {
                                const data = context.dataset.data[context.dataIndex];
                                return [
                                    `Predicción: ${data.x}`,
                                    `Real: ${data.y}`,
                                    `Valor: ${data.v}`
                                ];
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: `Matriz de Confusión - ${this.getSelectedModelName()}`
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Predicción'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Real'
                        }
                    }
                }
            }
        });
        
        // Guardar referencia al gráfico
        ctxConfusion.__chart__ = chart;
    }
    
    /**
     * Inicializa la gráfica de la curva ROC
     */
    initROCCurve() {
        const ctxROC = document.getElementById('rocCurve');
        
        // Limpiar cualquier gráfico anterior
        if (ctxROC.__chart__) {
            ctxROC.__chart__.destroy();
        }
        
        // Valores x comunes para todos los modelos (tasa de falsos positivos)
        const x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        
        // Datos para los modelos seleccionados
        const datasets = [];
        
        // Modelo principal (Random Forest por defecto)
        datasets.push({
            label: `Random Forest (AUC = 0.98)`,
            data: [0, 0.85, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 1.0],
            borderColor: 'rgba(67, 97, 238, 1)',
            backgroundColor: 'rgba(67, 97, 238, 0.1)',
            fill: true,
            tension: 0.3
        });
        
        // Agregar otros modelos si están seleccionados
        if (this.selectedModels.includes('svm')) {
            datasets.push({
                label: 'SVM (AUC = 0.96)',
                data: [0, 0.75, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0],
                borderColor: 'rgba(76, 201, 240, 1)',
                backgroundColor: 'rgba(76, 201, 240, 0.1)',
                fill: true,
                tension: 0.3
            });
        }
        
        if (this.selectedModels.includes('logistic_regression')) {
            datasets.push({
                label: 'Regresión Logística (AUC = 0.92)',
                data: [0, 0.60, 0.75, 0.82, 0.86, 0.89, 0.92, 0.94, 0.96, 0.98, 1.0],
                borderColor: 'rgba(247, 37, 133, 1)',
                backgroundColor: 'rgba(247, 37, 133, 0.1)',
                fill: true,
                tension: 0.3
            });
        }
        
        // Línea base (siempre incluida)
        datasets.push({
            label: 'Línea Base',
            data: x,
            borderColor: 'rgba(100, 100, 100, 0.5)',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
        });
        
        // Crear la curva ROC
        const chart = new Chart(ctxROC, {
            type: 'line',
            data: {
                labels: x,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Tasa de Falsos Positivos'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Tasa de Verdaderos Positivos'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Curva ROC'
                    }
                }
            }
        });
        
        // Guardar referencia al gráfico
        ctxROC.__chart__ = chart;
    }
    
    /**
     * Inicializa la gráfica de importancia de características
     */
    initFeatureImportance() {
        const ctxImportance = document.getElementById('featureImportance');
        
        // Limpiar cualquier gráfico anterior
        if (ctxImportance.__chart__) {
            ctxImportance.__chart__.destroy();
        }
        
        // Datos de importancia de características según el dataset
        let labels, data;
        
        switch (this.datasetSelect.value) {
            case 'iris':
                labels = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width'];
                data = [0.45, 0.41, 0.08, 0.06];
                break;
            case 'wine':
                labels = ['alcohol', 'flavanoids', 'color_intensity', 'proline', 'od280'];
                data = [0.22, 0.20, 0.15, 0.12, 0.10];
                break;
            case 'breast_cancer':
                labels = ['mean_radius', 'texture_error', 'perimeter_error', 'area_worst', 'smoothness_worst'];
                data = [0.25, 0.18, 0.15, 0.10, 0.08];
                break;
            case 'diabetes':
                labels = ['glucose', 'bmi', 'age', 'pregnancies', 'blood_pressure'];
                data = [0.38, 0.24, 0.18, 0.11, 0.09];
                break;
            default:
                labels = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'];
                data = [0.40, 0.30, 0.20, 0.10];
        }
        
        // Colores para las barras
        const backgroundColors = [
            'rgba(67, 97, 238, 0.7)',
            'rgba(76, 201, 240, 0.7)',
            'rgba(247, 37, 133, 0.7)',
            'rgba(148, 63, 219, 0.7)',
            'rgba(53, 162, 159, 0.7)',
            'rgba(233, 196, 106, 0.7)',
            'rgba(244, 162, 97, 0.7)',
            'rgba(231, 111, 81, 0.7)'
        ];
        
        const borderColors = backgroundColors.map(color => color.replace('0.7', '1'));
        
        // Crear el gráfico de importancia de características
        const chart = new Chart(ctxImportance, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Importancia',
                    data: data,
                    backgroundColor: backgroundColors.slice(0, labels.length),
                    borderColor: borderColors.slice(0, labels.length),
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        max: Math.max(...data) * 1.1,
                        title: {
                            display: true,
                            text: 'Importancia'
                        },
                        ticks: {
                            callback: (value) => `${(value * 100).toFixed(0)}%`
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: `Importancia de Características - ${this.getSelectedModelName()}`
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const value = context.raw;
                                return `Importancia: ${(value * 100).toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
        
        // Guardar referencia al gráfico
        ctxImportance.__chart__ = chart;
    }
    
    /**
     * Obtiene el nombre del modelo principal seleccionado
     * @returns {string} Nombre del modelo
     */
    getSelectedModelName() {
        if (this.selectedModels.length === 0) {
            switch (this.currentModelType) {
                case 'classification':
                    return 'Random Forest';
                case 'regression':
                    return 'Regresión Lineal';
                case 'clustering':
                    return 'K-Means';
                default:
                    return 'Modelo';
            }
        }
        
        const modelMap = {
            'random_forest': 'Random Forest',
            'svm': 'SVM',
            'logistic_regression': 'Regresión Logística',
            'knn': 'KNN',
            'decision_tree': 'Árbol de Decisión',
            'xgboost': 'XGBoost',
            'linear_regression': 'Regresión Lineal',
            'ridge': 'Ridge',
            'lasso': 'Lasso',
            'svr': 'SVR',
            'random_forest_regressor': 'Random Forest',
            'kmeans': 'K-Means',
            'dbscan': 'DBSCAN',
            'hierarchical': 'Jerárquico'
        };
        
        return modelMap[this.selectedModels[0]] || 'Modelo';
    }
    
    /**
     * Muestra u oculta el indicador de carga
     * @param {boolean} isLoading - Si está cargando o no
     * @param {string} message - Mensaje opcional a mostrar
     */
    setLoading(isLoading, message = 'Cargando...') {
        // Por ahora, esta es una implementación simple
        // En un caso real, se podría usar un componente de UI dedicado
        if (isLoading) {
            showMessage(message, 'info');
        }
    }

    /**
     * Crea la sección para clustering avanzado
     */
    createAdvancedClusteringSection() {
        const mainContainer = document.querySelector('.container-fluid');
        if (!mainContainer) return;
        
        // Crear la sección de clustering avanzado
        const clusteringSection = document.createElement('div');
        clusteringSection.id = 'advanced-clustering-section';
        clusteringSection.classList.add('card', 'mt-4', 'd-none');
        
        clusteringSection.innerHTML = `
            <div class="card-header bg-info text-white">
                <h4>Clustering Avanzado</h4>
            </div>
            <div class="card-body">
                <form id="clustering-form">
                    <div class="form-row">
                        <div class="form-group col-md-6">
                            <label for="clustering-dataset">Dataset</label>
                            <select id="clustering-dataset" class="form-control" required>
                                <option value="">Seleccionar dataset...</option>
                            </select>
                        </div>
                        <div class="form-group col-md-6">
                            <label for="clustering-algorithm">Algoritmo</label>
                            <select id="clustering-algorithm" class="form-control" required>
                                <option value="">Seleccionar algoritmo...</option>
                                <option value="kmeans">K-Means</option>
                                <option value="dbscan">DBSCAN</option>
                                <option value="agglomerative">Clustering Jerárquico</option>
                                <option value="spectral_clustering">Clustering Espectral</option>
                                <option value="gaussian_mixture">Gaussian Mixture</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group col-md-6">
                            <label for="n-clusters">Número de Clusters (para K-Means)</label>
                            <input type="number" id="n-clusters" class="form-control" min="2" max="20" value="3">
                        </div>
                        <div class="form-group col-md-6">
                            <label for="dim-reduction">Reducción de Dimensionalidad</label>
                            <select id="dim-reduction" class="form-control">
                                <option value="">Ninguna</option>
                                <option value="pca">PCA</option>
                                <option value="tsne">t-SNE</option>
                                <option value="umap">UMAP</option>
                            </select>
                        </div>
                    </div>
                    <div class="text-center">
                        <button type="submit" class="btn btn-primary">Realizar Clustering</button>
                    </div>
                </form>
                <div id="clustering-results" class="mt-4">
                    <h5>Resultados del Clustering</h5>
                    <div id="clustering-visualization" class="mt-3"></div>
                    <div id="clustering-stats" class="mt-3"></div>
                </div>
            </div>
        `;
        
        mainContainer.appendChild(clusteringSection);
        
        // Configurar el evento para el formulario de clustering
        const clusteringForm = document.getElementById('clustering-form');
        clusteringForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.performClustering();
        });
        
        // Completar el selector de datasets para clustering
        const clusteringDatasetSelect = document.getElementById('clustering-dataset');
        if (clusteringDatasetSelect) {
            // Rellenar con los mismos datasets que el selector principal
            Array.from(this.datasetSelect.options).forEach(option => {
                if (option.value) {
                    const newOption = document.createElement('option');
                    newOption.value = option.value;
                    newOption.text = option.text;
                    clusteringDatasetSelect.add(newOption);
                }
            });
        }
    }
    
    /**
     * Crea la sección para análisis de series temporales
     */
    createTimeSeriesSection() {
        const mainContainer = document.querySelector('.container-fluid');
        if (!mainContainer) return;
        
        // Crear la sección de series temporales
        const timeSeriesSection = document.createElement('div');
        timeSeriesSection.id = 'time-series-section';
        timeSeriesSection.classList.add('card', 'mt-4', 'd-none');
        
        timeSeriesSection.innerHTML = `
            <div class="card-header bg-warning text-white">
                <h4>Análisis de Series Temporales</h4>
            </div>
            <div class="card-body">
                <form id="time-series-form">
                    <div class="form-row">
                        <div class="form-group col-md-6">
                            <label for="ts-dataset">Dataset</label>
                            <select id="ts-dataset" class="form-control" required>
                                <option value="">Seleccionar dataset...</option>
                            </select>
                        </div>
                        <div class="form-group col-md-6">
                            <label for="ts-analysis-type">Tipo de Análisis</label>
                            <select id="ts-analysis-type" class="form-control" required>
                                <option value="">Seleccionar análisis...</option>
                                <option value="decomposition">Descomposición</option>
                                <option value="autocorrelation">Autocorrelación</option>
                                <option value="forecast_arima">Pronóstico ARIMA</option>
                                <option value="forecast_prophet">Pronóstico Prophet</option>
                                <option value="forecast_lstm">Pronóstico LSTM</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group col-md-6">
                            <label for="ts-date-column">Columna de Fecha</label>
                            <input type="text" id="ts-date-column" class="form-control" placeholder="Nombre de columna fecha">
                        </div>
                        <div class="form-group col-md-6">
                            <label for="ts-value-column">Columna de Valor</label>
                            <input type="text" id="ts-value-column" class="form-control" placeholder="Nombre de columna valor">
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="ts-forecast-periods">Periodos a Pronosticar</label>
                        <input type="number" id="ts-forecast-periods" class="form-control" min="1" value="10">
                    </div>
                    <div class="text-center">
                        <button type="submit" class="btn btn-primary">Analizar Serie Temporal</button>
                    </div>
                </form>
                <div id="time-series-results" class="mt-4">
                    <h5>Resultados del Análisis</h5>
                    <div id="ts-visualization" class="mt-3"></div>
                    <div id="ts-stats" class="mt-3"></div>
                </div>
            </div>
        `;
        
        mainContainer.appendChild(timeSeriesSection);
        
        // Configurar el evento para el formulario de series temporales
        const timeSeriesForm = document.getElementById('time-series-form');
        timeSeriesForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeTimeSeries();
        });
        
        // Completar el selector de datasets para series temporales
        const tsDatasetSelect = document.getElementById('ts-dataset');
        if (tsDatasetSelect) {
            // Rellenar con los mismos datasets que el selector principal
            Array.from(this.datasetSelect.options).forEach(option => {
                if (option.value) {
                    const newOption = document.createElement('option');
                    newOption.value = option.value;
                    newOption.text = option.text;
                    tsDatasetSelect.add(newOption);
                }
            });
        }
    }
    
    /**
     * Crea la sección para visualizaciones avanzadas
     */
    createAdvancedVisualizationsSection() {
        const mainContainer = document.querySelector('.container-fluid');
        if (!mainContainer) return;
        
        // Crear la sección de visualizaciones avanzadas
        const visualizationsSection = document.createElement('div');
        visualizationsSection.id = 'advanced-visualizations-section';
        visualizationsSection.classList.add('card', 'mt-4', 'd-none');
        
        visualizationsSection.innerHTML = `
            <div class="card-header bg-success text-white">
                <h4>Visualizaciones Avanzadas</h4>
            </div>
            <div class="card-body">
                <form id="visualization-form">
                    <div class="form-row">
                        <div class="form-group col-md-6">
                            <label for="visualization-model">Modelo</label>
                            <select id="visualization-model" class="form-control" required>
                                <option value="">Seleccionar modelo entrenado...</option>
                            </select>
                        </div>
                        <div class="form-group col-md-6">
                            <label for="visualization-type">Tipo de Visualización</label>
                            <select id="visualization-type" class="form-control" required>
                                <option value="">Seleccionar visualización...</option>
                                <option value="learning_curve">Curva de Aprendizaje</option>
                                <option value="feature_importance">Importancia de Variables</option>
                                <option value="roc_curve">Curva ROC</option>
                                <option value="pr_curve">Curva Precision-Recall</option>
                                <option value="confusion_matrix">Matriz de Confusión</option>
                                <option value="residuals">Gráfico de Residuos</option>
                            </select>
                        </div>
                    </div>
                    <div class="text-center">
                        <button type="submit" class="btn btn-primary">Generar Visualización</button>
                    </div>
                </form>
                <div id="visualization-results" class="mt-4">
                    <h5>Visualización</h5>
                    <div id="visualization-container" class="mt-3"></div>
                </div>
            </div>
        `;
        
        mainContainer.appendChild(visualizationsSection);
        
        // Configurar el evento para el formulario de visualizaciones
        const visualizationForm = document.getElementById('visualization-form');
        visualizationForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.generateVisualization();
        });
    }
    /**
     * Alterna entre modo básico y avanzado
     */
    toggleAdvancedMode() {
        const advancedModeToggle = document.getElementById('toggle-advanced-mode');
        const basicSections = document.querySelectorAll('.basic-mode-section');
        const advancedSections = [
            document.getElementById('advanced-clustering-section'),
            document.getElementById('time-series-section'),
            document.getElementById('advanced-visualizations-section')
        ];
        
        // Verificar si estamos activando o desactivando el modo avanzado
        const isActivating = advancedSections[0].classList.contains('d-none');
        
        if (isActivating) {
            // Activar modo avanzado
            advancedModeToggle.innerText = 'Usar Modelos Básicos';
            advancedModeToggle.classList.replace('btn-outline-primary', 'btn-outline-secondary');
            
            // Mostrar secciones avanzadas
            advancedSections.forEach(section => {
                if (section) section.classList.remove('d-none');
            });
            
            // Ocultar secciones básicas (opcional)
            basicSections.forEach(section => {
                section.classList.add('d-none');
            });
            
            // Cargar los modelos entrenados en el selector de visualización
            this.loadTrainedModelsForVisualization();
        } else {
            // Desactivar modo avanzado
            advancedModeToggle.innerText = 'Usar Modelos Avanzados';
            advancedModeToggle.classList.replace('btn-outline-secondary', 'btn-outline-primary');
            
            // Ocultar secciones avanzadas
            advancedSections.forEach(section => {
                if (section) section.classList.add('d-none');
            });
            
            // Mostrar secciones básicas
            basicSections.forEach(section => {
                section.classList.remove('d-none');
            });
        }
    }
    
    /**
     * Carga los modelos entrenados en el selector de visualización
     */
    loadTrainedModelsForVisualization() {
        const visualizationModelSelect = document.getElementById('visualization-model');
        if (!visualizationModelSelect) return;
        
        // Limpiar opciones existentes
        visualizationModelSelect.innerHTML = '<option value="">Seleccionar modelo entrenado...</option>';
        
        // Cargar modelos entrenados de todas las categorías
        for (const modelType in this.trainedModels) {
            const models = this.trainedModels[modelType];
            for (const modelId in models) {
                const option = document.createElement('option');
                option.value = modelId;
                option.text = `${models[modelId].name} (${modelType})`;
                visualizationModelSelect.appendChild(option);
            }
        }
    }
    
    /**
     * Realiza operación de clustering avanzado
     */
    async performClustering() {
        const dataset_id = document.getElementById('clustering-dataset').value;
        const algorithm = document.getElementById('clustering-algorithm').value;
        const n_clusters = parseInt(document.getElementById('n-clusters').value, 10);
        const dimensionality_reduction = document.getElementById('dim-reduction').value;
        
        if (!dataset_id || !algorithm) {
            this.showNotification('warning', 'Debe seleccionar un dataset y un algoritmo');
            return;
        }
        
        // Mostrar indicador de carga
        const clusteringResults = document.getElementById('clustering-results');
        clusteringResults.innerHTML = '<div class="text-center"><div class="spinner-border text-primary"></div><p>Realizando clustering...</p></div>';
        
        try {
            const response = await fetch('/api/ml/clustering', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    dataset_id,
                    algorithm,
                    n_clusters: algorithm === 'kmeans' || algorithm === 'gaussian_mixture' ? n_clusters : undefined,
                    dimensionality_reduction: dimensionality_reduction || undefined
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Mostrar resultados del clustering
                this.displayClusteringResults(data);
            } else {
                this.showNotification('danger', `Error al realizar clustering: ${data.error}`);
                clusteringResults.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            }
        } catch (error) {
            console.error('Error al realizar clustering:', error);
            this.showNotification('danger', 'Error al comunicarse con el servidor');
            clusteringResults.innerHTML = `<div class="alert alert-danger">Error de comunicación con el servidor</div>`;
        }
    }
    
    /**
     * Muestra los resultados de clustering en la UI
     */
    displayClusteringResults(data) {
        const clusteringVisualization = document.getElementById('clustering-visualization');
        const clusteringStats = document.getElementById('clustering-stats');
        
        // Limpiar áreas de resultados
        clusteringVisualization.innerHTML = '';
        clusteringStats.innerHTML = '';
        
        if (data.visualization) {
            // Crear elemento para mostrar la visualización (imagen en base64)
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data.visualization}`;
            img.classList.add('img-fluid', 'mx-auto', 'd-block');
            clusteringVisualization.appendChild(img);
        }
        
        if (data.metrics) {
            // Crear tabla para las métricas de evaluación
            const metricsTable = document.createElement('table');
            metricsTable.classList.add('table', 'table-bordered', 'table-sm');
            
            let tableHTML = '<thead><tr><th>Métrica</th><th>Valor</th></tr></thead><tbody>';
            
            for (const [metric, value] of Object.entries(data.metrics)) {
                tableHTML += `<tr><td>${this.formatMetricName(metric)}</td><td>${value.toFixed(4)}</td></tr>`;
            }
            
            tableHTML += '</tbody>';
            metricsTable.innerHTML = tableHTML;
            clusteringStats.appendChild(metricsTable);
        }
        
        if (data.cluster_sizes) {
            // Mostrar distribución de tamaños de cluster
            const sizesTitle = document.createElement('h6');
            sizesTitle.innerHTML = 'Distribución de Clusters';
            sizesTitle.classList.add('mt-3');
            clusteringStats.appendChild(sizesTitle);
            
            const sizesList = document.createElement('ul');
            sizesList.classList.add('list-group');
            
            for (const [cluster, size] of Object.entries(data.cluster_sizes)) {
                const item = document.createElement('li');
                item.classList.add('list-group-item', 'd-flex', 'justify-content-between', 'align-items-center');
                item.innerHTML = `Cluster ${cluster} <span class="badge badge-primary badge-pill">${size} elementos</span>`;
                sizesList.appendChild(item);
            }
            
            clusteringStats.appendChild(sizesList);
        }
    }
    
    /**
     * Analiza series temporales
     */
    async analyzeTimeSeries() {
        const dataset_id = document.getElementById('ts-dataset').value;
        const analysis_type = document.getElementById('ts-analysis-type').value;
        const date_column = document.getElementById('ts-date-column').value;
        const value_column = document.getElementById('ts-value-column').value;
        const forecast_periods = parseInt(document.getElementById('ts-forecast-periods').value, 10);
        
        if (!dataset_id || !analysis_type) {
            this.showNotification('warning', 'Debe seleccionar un dataset y un tipo de análisis');
            return;
        }
        
        // Mostrar indicador de carga
        const tsResults = document.getElementById('time-series-results');
        tsResults.innerHTML = '<div class="text-center"><div class="spinner-border text-primary"></div><p>Analizando serie temporal...</p></div>';
        
        try {
            const response = await fetch('/api/ml/timeseries', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    dataset_id,
                    analysis_type,
                    params: {
                        date_column,
                        value_column,
                        forecast_periods
                    }
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Mostrar resultados del análisis de series temporales
                this.displayTimeSeriesResults(data);
            } else {
                this.showNotification('danger', `Error en el análisis: ${data.error}`);
                tsResults.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            }
        } catch (error) {
            console.error('Error en el análisis de series temporales:', error);
            this.showNotification('danger', 'Error al comunicarse con el servidor');
            tsResults.innerHTML = `<div class="alert alert-danger">Error de comunicación con el servidor</div>`;
        }
    }
    
    /**
     * Muestra los resultados del análisis de series temporales
     */
    displayTimeSeriesResults(data) {
        const tsVisualization = document.getElementById('ts-visualization');
        const tsStats = document.getElementById('ts-stats');
        
        // Limpiar áreas de resultados
        tsVisualization.innerHTML = '';
        tsStats.innerHTML = '';
        
        if (data.visualization) {
            // Crear elemento para mostrar la visualización (imagen en base64)
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data.visualization}`;
            img.classList.add('img-fluid', 'mx-auto', 'd-block');
            tsVisualization.appendChild(img);
        }
        
        if (data.forecast_data) {
            // Crear tabla para los datos de pronóstico
            const forecastTable = document.createElement('table');
            forecastTable.classList.add('table', 'table-bordered', 'table-sm', 'mt-3');
            
            let tableHTML = '<thead><tr><th>Fecha</th><th>Pronóstico</th>';
            
            // Si hay intervalos de confianza
            if (data.forecast_data[0].lower_bound !== undefined) {
                tableHTML += '<th>Límite Inferior</th><th>Límite Superior</th>';
            }
            
            tableHTML += '</tr></thead><tbody>';
            
            data.forecast_data.forEach(row => {
                tableHTML += `<tr><td>${row.date}</td><td>${row.value.toFixed(2)}</td>`;
                
                if (row.lower_bound !== undefined) {
                    tableHTML += `<td>${row.lower_bound.toFixed(2)}</td><td>${row.upper_bound.toFixed(2)}</td>`;
                }
                
                tableHTML += '</tr>';
            });
            
            tableHTML += '</tbody>';
            forecastTable.innerHTML = tableHTML;
            tsStats.appendChild(forecastTable);
        }
        
        if (data.metrics) {
            // Crear sección para métricas
            const metricsTitle = document.createElement('h6');
            metricsTitle.innerHTML = 'Métricas de Evaluación';
            metricsTitle.classList.add('mt-3');
            tsStats.appendChild(metricsTitle);
            
            const metricsTable = document.createElement('table');
            metricsTable.classList.add('table', 'table-bordered', 'table-sm');
            
            let tableHTML = '<thead><tr><th>Métrica</th><th>Valor</th></tr></thead><tbody>';
            
            for (const [metric, value] of Object.entries(data.metrics)) {
                tableHTML += `<tr><td>${this.formatMetricName(metric)}</td><td>${typeof value === 'number' ? value.toFixed(4) : value}</td></tr>`;
            }
            
            tableHTML += '</tbody>';
            metricsTable.innerHTML = tableHTML;
            tsStats.appendChild(metricsTable);
        }
    }
    
    /**
     * Genera visualización avanzada para un modelo
     */
    async generateVisualization() {
        const model_id = document.getElementById('visualization-model').value;
        const visualization_type = document.getElementById('visualization-type').value;
        
        if (!model_id || !visualization_type) {
            this.showNotification('warning', 'Debe seleccionar un modelo y un tipo de visualización');
            return;
        }
        
        // Mostrar indicador de carga
        const visualizationContainer = document.getElementById('visualization-container');
        visualizationContainer.innerHTML = '<div class="text-center"><div class="spinner-border text-primary"></div><p>Generando visualización...</p></div>';
        
        try {
            const response = await fetch('/api/ml/visualize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_id,
                    visualization_type
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Mostrar visualización
                this.displayModelVisualization(data);
            } else {
                this.showNotification('danger', `Error al generar visualización: ${data.error}`);
                visualizationContainer.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            }
        } catch (error) {
            console.error('Error al generar visualización:', error);
            this.showNotification('danger', 'Error al comunicarse con el servidor');
            visualizationContainer.innerHTML = `<div class="alert alert-danger">Error de comunicación con el servidor</div>`;
        }
    }
    
    /**
     * Muestra visualización de modelo
     */
    displayModelVisualization(data) {
        const visualizationContainer = document.getElementById('visualization-container');
        visualizationContainer.innerHTML = '';
        
        if (data.visualization) {
            // Crear elemento para mostrar la visualización (imagen en base64)
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data.visualization}`;
            img.classList.add('img-fluid', 'mx-auto', 'd-block');
            visualizationContainer.appendChild(img);
        }
        
        if (data.metrics) {
            // Crear sección para métricas adicionales
            const metricsDiv = document.createElement('div');
            metricsDiv.classList.add('mt-4', 'card', 'card-body', 'bg-light');
            
            let metricsHTML = '<h6>Métricas Adicionales</h6>';
            metricsHTML += '<div class="row">';
            
            for (const [metric, value] of Object.entries(data.metrics)) {
                metricsHTML += `<div class="col-md-4 mb-2">
                    <div class="card">
                        <div class="card-body py-2">
                            <h6 class="card-title">${this.formatMetricName(metric)}</h6>
                            <p class="card-text font-weight-bold">${typeof value === 'number' ? value.toFixed(4) : value}</p>
                        </div>
                    </div>
                </div>`;
            }
            
            metricsHTML += '</div>';
            metricsDiv.innerHTML = metricsHTML;
            visualizationContainer.appendChild(metricsDiv);
        }
        
        if (data.explanation) {
            // Crear sección para explicación
            const explanationDiv = document.createElement('div');
            explanationDiv.classList.add('mt-3', 'alert', 'alert-info');
            explanationDiv.innerHTML = `<h6>Interpretación</h6><p>${data.explanation}</p>`;
            visualizationContainer.appendChild(explanationDiv);
        }
    }
    
    /**
     * Formatea el nombre de la métrica para mostrar
     */
    formatMetricName(metricName) {
        return metricName
            .replace(/_/g, ' ')
            .replace(/\b\w/g, letter => letter.toUpperCase());
    }
    
    /**
     * Muestra una notificación en la interfaz
     */
    showNotification(type, message) {
        const notificationArea = document.getElementById('notifications');
        if (!notificationArea) return;
        
        const alert = document.createElement('div');
        alert.classList.add('alert', `alert-${type}`, 'alert-dismissible', 'fade', 'show');
        alert.innerHTML = `
            ${message}
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                <span aria-hidden="true">&times;</span>
            </button>
        `;
        
        notificationArea.appendChild(alert);
        
        // Auto-eliminar la notificación después de 5 segundos
        setTimeout(() => {
            alert.classList.remove('show');
            setTimeout(() => {
                notificationArea.removeChild(alert);
            }, 150);
        }, 5000);
    }
}

// Variables globales para el estado del modelo
let modelTraining = false;
let currentIteration = 0;
let maxIterations = 10;
let trainingData = null;
let modelType = '';
let modelMetrics = {
    accuracy: 0,
    loss: 0,
    precision: 0,
    recall: 0,
    f1Score: 0
};

// Inicialización cuando se carga la página
document.addEventListener('DOMContentLoaded', function() {
    // Ocultar la sección de iteración al principio
    if (document.getElementById('iteration-section')) {
        document.getElementById('iteration-section').style.display = 'none';
    }
    
    // Configurar listeners para los botones principales
    if (document.getElementById('train-model')) {
        document.getElementById('train-model').addEventListener('click', startTraining);
    }
    
    if (document.getElementById('continue-iteration')) {
        document.getElementById('continue-iteration').addEventListener('click', continueIteration);
    }
    
    if (document.getElementById('stop-iteration')) {
        document.getElementById('stop-iteration').addEventListener('click', stopIteration);
    }
    
    // Configurar listeners para selectores y rangos
    if (document.getElementById('test-size')) {
        document.getElementById('test-size').addEventListener('input', function() {
            document.getElementById('test-size-value').textContent = this.value + '%';
        });
    }
    
    if (document.getElementById('max-iterations')) {
        document.getElementById('max-iterations').addEventListener('change', function() {
            maxIterations = parseInt(this.value);
        });
    }
    
    if (document.getElementById('model-type')) {
        document.getElementById('model-type').addEventListener('change', function() {
            modelType = this.value;
        });
    }
    
    // Inicializar tabs para resultados
    setupTabs();
});

// Función para iniciar el entrenamiento del modelo
function startTraining() {
    if (modelTraining) return;
    
    // Obtener parámetros de entrenamiento
    const testSize = document.getElementById('test-size').value / 100;
    const randomState = document.getElementById('random-state').value;
    maxIterations = document.getElementById('max-iterations').value;
    modelType = document.getElementById('model-type').value;
    
    // Mostrar indicador de carga
    showLoading('Iniciando entrenamiento...');
    
    // Aquí se haría la llamada al backend para iniciar el entrenamiento
    // Simulación de llamada a la API para propósitos de demostración
    setTimeout(() => {
        hideLoading();
        
        // Simular inicio del entrenamiento
        modelTraining = true;
        currentIteration = 1;
        
        // Mostrar sección de iteración
        document.getElementById('iteration-section').style.display = 'block';
        
        // Actualizar métricas iniciales
        updateMetrics({
            iteration: currentIteration,
            accuracy: 0.65,
            loss: 0.75
        });
        
        // Deshabilitar botón de entrenamiento
        document.getElementById('train-model').disabled = true;
    }, 1500);
}

// Función para continuar con la siguiente iteración
function continueIteration() {
    if (!modelTraining || currentIteration >= maxIterations) return;
    
    // Mostrar indicador de carga
    showLoading('Procesando iteración ' + (currentIteration + 1) + '...');
    
    // Simular procesamiento de la siguiente iteración
    setTimeout(() => {
        hideLoading();
        
        // Incrementar iteración
        currentIteration++;
        
        // Actualizar métricas (simulación)
        const newAccuracy = Math.min(0.65 + (currentIteration * 0.05), 0.98);
        const newLoss = Math.max(0.75 - (currentIteration * 0.08), 0.10);
        
        updateMetrics({
            iteration: currentIteration,
            accuracy: newAccuracy.toFixed(2),
            loss: newLoss.toFixed(2)
        });
        
        // Verificar si hemos alcanzado el máximo de iteraciones
        if (currentIteration >= maxIterations) {
            document.getElementById('continue-iteration').disabled = true;
            showNotification('Se ha alcanzado el máximo de iteraciones configurado', 'info');
        }
    }, 1500);
}

// Función para detener el entrenamiento y mostrar resultados
function stopIteration() {
    if (!modelTraining) return;
    
    // Mostrar indicador de carga
    showLoading('Finalizando entrenamiento y evaluando modelo...');
    
    // Simular finalización del entrenamiento y evaluación del modelo
    setTimeout(() => {
        hideLoading();
        
        // Marcar entrenamiento como finalizado
        modelTraining = false;
        
        // Actualizar métricas finales
        const finalMetrics = {
            accuracy: parseFloat(document.getElementById('current-accuracy').textContent),
            loss: parseFloat(document.getElementById('current-loss').textContent),
            precision: 0.89,
            recall: 0.85,
            f1Score: 0.87
        };
        
        // Guardar métricas
        modelMetrics = finalMetrics;
        
        // Ocultar sección de iteración
        document.getElementById('iteration-section').style.display = 'none';
        
        // Mostrar sección de resultados
        showResultsCard();
        
        // Habilitar botón de entrenamiento para un nuevo ciclo
        document.getElementById('train-model').disabled = false;
        document.getElementById('continue-iteration').disabled = false;
        
        // Notificar al usuario
        showNotification('Entrenamiento completado con éxito', 'success');
    }, 2000);
}

// Función para actualizar métricas en la interfaz
function updateMetrics(data) {
    if (document.getElementById('current-iteration')) {
        document.getElementById('current-iteration').textContent = data.iteration;
    }
    
    if (document.getElementById('current-accuracy')) {
        document.getElementById('current-accuracy').textContent = data.accuracy;
    }
    
    if (document.getElementById('current-loss')) {
        document.getElementById('current-loss').textContent = data.loss;
    }
}

// Función para mostrar la sección de resultados
function showResultsCard() {
    const resultsCard = document.getElementById('results-card');
    if (resultsCard) {
        resultsCard.style.display = 'block';
        
        // Llenar la tabla de métricas
        const metricsTableBody = document.getElementById('metrics-table-body');
        if (metricsTableBody) {
            metricsTableBody.innerHTML = `
                <tr>
                    <td>Precisión (Accuracy)</td>
                    <td>${modelMetrics.accuracy}</td>
                </tr>
                <tr>
                    <td>Loss</td>
                    <td>${modelMetrics.loss}</td>
                </tr>
                <tr>
                    <td>Precisión (Precision)</td>
                    <td>${modelMetrics.precision}</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>${modelMetrics.recall}</td>
                </tr>
                <tr>
                    <td>F1-Score</td>
                    <td>${modelMetrics.f1Score}</td>
                </tr>
            `;
        }
        
        // Generar visualizaciones
        generateConfusionMatrix();
        generateLearningCurve();
        setupPredictionTab();
    }
}

// Configuración de los tabs en la sección de resultados
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            button.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Función para generar matriz de confusión (simulada)
function generateConfusionMatrix() {
    // Aquí se implementaría la lógica para generar la matriz de confusión
    // usando una librería como Chart.js
    console.log('Generando matriz de confusión');
}

// Función para generar curva de aprendizaje (simulada)
function generateLearningCurve() {
    // Aquí se implementaría la lógica para generar la curva de aprendizaje
    // usando una librería como Chart.js
    console.log('Generando curva de aprendizaje');
}

// Configurar el tab de predicción
function setupPredictionTab() {
    // Aquí se generarían los campos necesarios según las características del dataset
    // y se configuraría la funcionalidad de predicción
    console.log('Configurando tab de predicción');
}

// Funciones auxiliares
function showLoading(message) {
    // Implementar lógica para mostrar indicador de carga
    console.log('Loading: ' + message);
    // Ejemplo: crear y mostrar un overlay con un spinner
}

function hideLoading() {
    // Implementar lógica para ocultar indicador de carga
    console.log('Loading complete');
    // Ejemplo: eliminar el overlay con el spinner
}

function showNotification(message, type) {
    // Implementar lógica para mostrar notificaciones
    console.log('Notification [' + type + ']: ' + message);
    // Ejemplo: crear y mostrar un toast o una notificación temporal
}

// Inicializar el gestor de modelos cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    window.modelManager = new MLModelManager();
});

// Polyfill para el tipo de gráfico matriz (matrix chart)
Chart.register({
    id: 'matrix',
    beforeInit: function(chart) {
        chart.data.datasets.forEach(function(dataset) {
            dataset.borderAlign = 'inner';
        });
    },
    defaults: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
            padding: {
                top: 10,
                right: 10,
                bottom: 10,
                left: 10
            }
        },
        plugins: {
            legend: {
                display: false,
            },
            tooltip: {
                callbacks: {
                    title: function() {
                        return '';
                    },
                    label: function(context) {
                        const d = context.dataset.data[context.dataIndex];
                        return [
                            `Predicción: ${d.x}`,
                            `Clase Real: ${d.y}`,
                            `Valor: ${d.v}`
                        ];
                    }
                }
            }
        }
    }
});

// Manejar la visualización del nombre del archivo seleccionado
document.addEventListener('DOMContentLoaded', function() {
    // Selector de archivos personalizado
    const fileInput = document.getElementById('dataset-file');
    const fileNameDisplay = document.querySelector('.file-name-display');
    
    if (fileInput && fileNameDisplay) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                fileNameDisplay.textContent = fileInput.files[0].name;
            } else {
                fileNameDisplay.textContent = 'Ningún archivo seleccionado';
            }
        });
    }
    
    // Efectos de hover y animaciones para botones
    const buttons = document.querySelectorAll('.btn-primary');
    buttons.forEach(button => {
        button.addEventListener('mouseover', function() {
            this.style.transition = 'all 0.3s ease';
        });
    });

    // Inicialización del formulario de carga de datasets
    const uploadForm = document.getElementById('upload-dataset-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Mostrar animación de carga
            const submitButton = this.querySelector('button[type="submit"]');
            const originalText = submitButton.innerHTML;
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Cargando...';
            submitButton.disabled = true;
            
            // Simulando carga (reemplazar con lógica real)
            setTimeout(() => {
                submitButton.innerHTML = '<i class="fas fa-check"></i> ¡Dataset cargado!';
                submitButton.classList.add('success');
                
                // Restaurar el botón después de 2 segundos
                setTimeout(() => {
                    submitButton.innerHTML = originalText;
                    submitButton.disabled = false;
                    submitButton.classList.remove('success');
                    uploadForm.reset();
                    fileNameDisplay.textContent = 'Ningún archivo seleccionado';
                }, 2000);
            }, 1500);
        });
    }
    
    // Actualización de datasets
    const refreshButton = document.getElementById('refresh-datasets');
    if (refreshButton) {
        refreshButton.addEventListener('click', function() {
            // Agregar clase de animación al botón
            this.classList.add('rotating');
            
            // Simulando actualización (reemplazar con lógica real)
            setTimeout(() => {
                this.classList.remove('rotating');
                
                // Mostrar notificación de éxito
                showNotification('Datasets actualizados correctamente');
            }, 1000);
        });
    }
    
    // Funciones auxiliares
    function showNotification(message, type = 'success') {
        // Crear elemento de notificación
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Añadir al DOM
        document.body.appendChild(notification);
        
        // Animar entrada
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Eliminar después de un tiempo
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }
});