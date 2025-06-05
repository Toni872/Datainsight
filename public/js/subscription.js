/**
 * Funciones para la gestión de suscripciones
 */

// Variables globales
let currentPlan = null;
let userSubscriptionData = null;

/**
 * Inicializa la página de suscripciones
 */
async function initSubscriptionPage() {
    try {
        // Comprobar si el usuario está au                <li class="feature-item">
                    <span class="feature-icon">✓</span> 
                    <span><span class="feature-highlight">${formatNumber(plan.features.apiCallsPerMonth)}</span> llamadas API por mes</span>
                </li>
                <li class="feature-item">
                    <span class="feature-icon">✓</span>
                    <span><span class="feature-highlight">${formatNumber(plan.features.modelTrainingPerMonth)}</span> entrenamientos de modelo por mes</span>
                </li>o
        const isAuthenticated = await checkAuthentication();
        if (!isAuthenticated) {
            window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            return;
        }

        // Cargar datos de la suscripción actual
        await loadCurrentSubscription();

        // Cargar planes de suscripción desde la API
        await loadSubscriptionPlans();

        // Configurar listeners de eventos
        setupEventListeners();
        
        // Procesar parámetros de URL (como redirecciones desde Stripe)
        handleUrlParameters();
    } catch (error) {
        console.error('Error en la inicialización:', error);
        showNotification('Error al cargar la página de suscripciones', 'error');
    }
}

/**
 * Comprueba si el usuario está autenticado
 * @returns {Promise<boolean>}
 */
async function checkAuthentication() {
    try {
        const response = await fetch('/api/auth/check', {
            credentials: 'include'
        });
        
        const data = await response.json();
        return data.authenticated === true;
    } catch (error) {
        console.error('Error al comprobar autenticación:', error);
        return false;
    }
}

/**
 * Carga la información de la suscripción actual del usuario
 */
async function loadCurrentSubscription() {
    try {
        const response = await fetch('/api/subscription/current', {
            credentials: 'include'
        });
        
        if (!response.ok) {
            throw new Error('No se pudo cargar la información de la suscripción');
        }
        
        const data = await response.json();
        
        if (data.success && data.subscription) {
            userSubscriptionData = data.subscription;
            updateSubscriptionUI(data.subscription);
            return data.subscription;
        } else {
            throw new Error(data.message || 'Error al cargar la suscripción');
        }
    } catch (error) {
        console.error('Error al cargar suscripción:', error);
        showNotification('No se pudo cargar la información de tu suscripción actual', 'error');
    }
}

/**
 * Actualiza la UI con la información de la suscripción
 * @param {Object} subscription Datos de la suscripción
 */
function updateSubscriptionUI(subscription) {
    // Si estamos en la página de perfil/suscripción, actualizar la información detallada
    const subscriptionDetailsElement = document.getElementById('subscription-details');
    if (subscriptionDetailsElement) {
        const expiresAt = new Date(subscription.expiresAt).toLocaleDateString();
        
        let statusClass = '';
        switch (subscription.status) {
            case 'active': statusClass = 'status-active'; break;
            case 'canceled': statusClass = 'status-canceled'; break;
            case 'past_due': statusClass = 'status-past-due'; break;
            default: statusClass = 'status-trial';
        }
        
        subscriptionDetailsElement.innerHTML = `
            <div class="subscription-info">
                <h3>Tu Suscripción Actual</h3>
                <div class="subscription-data">
                    <p><strong>Plan:</strong> ${subscription.planName}</p>
                    <p><strong>Estado:</strong> <span class="subscription-status ${statusClass}">${getStatusText(subscription.status)}</span></p>
                    <p><strong>Período de facturación:</strong> ${subscription.billingPeriod === 'monthly' ? 'Mensual' : 'Anual'}</p>
                    <p><strong>Fecha de renovación:</strong> ${expiresAt}</p>
                    ${subscription.cancelAtPeriodEnd ? '<p class="cancel-notice">Tu suscripción se cancelará al final del período actual</p>' : ''}
                </div>
            </div>
            <div class="usage-info">
                <h3>Uso Actual</h3>
                <div class="usage-meters">
                    <div class="usage-meter">
                        <label>Llamadas API</label>
                        <div class="meter-bar">
                            <div class="meter-fill" style="width: ${calculatePercentage(subscription.quotaInfo.apiCalls.used, subscription.quotaInfo.apiCalls.limit)}%"></div>
                        </div>
                        <div class="meter-text">${subscription.quotaInfo.apiCalls.used} / ${subscription.quotaInfo.apiCalls.limit}</div>
                    </div>
                    <div class="usage-meter">
                        <label>Entrenamientos de Modelos</label>
                        <div class="meter-bar">
                            <div class="meter-fill" style="width: ${calculatePercentage(subscription.quotaInfo.modelTraining.used, subscription.quotaInfo.modelTraining.limit)}%"></div>
                        </div>
                        <div class="meter-text">${subscription.quotaInfo.modelTraining.used} / ${subscription.quotaInfo.modelTraining.limit}</div>
                    </div>
                    <div class="usage-meter">
                        <label>Almacenamiento</label>
                        <div class="meter-bar">
                            <div class="meter-fill" style="width: ${calculatePercentage(subscription.quotaInfo.storage.used, subscription.quotaInfo.storage.limit)}%"></div>
                        </div>
                        <div class="meter-text">${formatStorage(subscription.quotaInfo.storage.used)} / ${formatStorage(subscription.quotaInfo.storage.limit)}</div>
                    </div>
                </div>
            </div>
            ${subscription.planId !== 'free' ? `
                <div class="subscription-actions">
                    <button id="cancel-subscription" class="btn-cancel">Cancelar Suscripción</button>
                </div>
            ` : ''}
        `;
        
        // Añadir listener al botón de cancelar suscripción
        const cancelButton = document.getElementById('cancel-subscription');
        if (cancelButton) {
            cancelButton.addEventListener('click', cancelSubscription);
        }
    }
    
    // También destacar el plan actual en la lista de planes si estamos en la página de suscripciones
    highlightCurrentPlan(subscription.planId);
    
    // Actualizar el badge de suscripción en el header
    updateSubscriptionBadge(subscription.planId);
}

/**
 * Destaca el plan actual del usuario en la lista de planes
 * @param {string} planId ID del plan actual
 */
function highlightCurrentPlan(planId) {
    // Resetear todos los planes
    document.querySelectorAll('.pricing-card').forEach(card => {
        card.classList.remove('current-plan');
        
        // Buscar el botón CTA y actualizar el texto si es el plan actual
        const ctaButton = card.querySelector('.pricing-cta');
        if (ctaButton && card.dataset.planId === planId) {
            ctaButton.textContent = 'Plan Actual';
            ctaButton.classList.add('current');
            card.classList.add('current-plan');
        }
    });
}

/**
 * Actualiza el badge de suscripción en el header
 * @param {string} planId ID del plan
 */
function updateSubscriptionBadge(planId) {
    const userMenuElement = document.querySelector('.user-menu');
    if (userMenuElement) {
        // Remover badges existentes
        const existingBadge = userMenuElement.querySelector('.plan-badge');
        if (existingBadge) {
            existingBadge.remove();
        }
        
        // Añadir nuevo badge según el plan
        let badgeClass = '';
        let badgeText = '';
        
        switch (planId) {
            case 'basic':
                badgeClass = 'badge-basic';
                badgeText = 'Básico';
                break;
            case 'pro':
                badgeClass = 'badge-pro';
                badgeText = 'Pro';
                break;
            case 'enterprise':
                badgeClass = 'badge-enterprise';
                badgeText = 'Enterprise';
                break;
            default:
                badgeClass = 'badge-free';
                badgeText = 'Free';
        }
        
        const badge = document.createElement('span');
        badge.className = `plan-badge ${badgeClass}`;
        badge.textContent = badgeText;
        userMenuElement.appendChild(badge);
    }
}

/**
 * Carga los planes de suscripción
 */
async function loadSubscriptionPlans() {
    try {
        const response = await fetch('/api/subscription/plans');
        const data = await response.json();
        
        if (data.success && data.plans) {
            // Guardar los planes para uso posterior
            window.plansData = data.plans;
            
            renderPricingCards(data.plans);
            return data.plans;
        } else {
            throw new Error('No se pudieron cargar los planes de suscripción');
        }
    } catch (error) {
        console.error('Error al cargar planes:', error);
        showError('Error al conectar con el servidor');
        throw error;
    }
}

/**
 * Renderiza las tarjetas de precios con los planes disponibles
 * @param {Array} plans Lista de planes
 */
function renderPricingCards(plans) {
    const container = document.getElementById('pricing-container');
    if (!container) return;
    
    container.innerHTML = '';
    
    const isYearly = document.getElementById('billingToggle')?.checked || false;
    
    plans.forEach(plan => {
        const price = isYearly ? plan.price.yearly : plan.price.monthly;
        const billingPeriod = isYearly ? 'yearly' : 'monthly';
        
        const card = document.createElement('div');
        card.className = `pricing-card ${plan.popular ? 'popular' : ''}`;
        card.dataset.planId = plan.id;
        
        if (plan.popular) {
            const badge = document.createElement('div');
            badge.className = 'popular-badge';
            badge.textContent = 'RECOMENDADO';
            card.appendChild(badge);
        }
        
        // Calcular el ahorro anual
        let savingsText = '';
        if (isYearly && plan.id !== 'free') {
            const monthlyCost = plan.price.monthly * 12;
            const yearlyCost = plan.price.yearly;
            const savings = monthlyCost - yearlyCost;
            const savingsPercent = Math.round((savings / monthlyCost) * 100);
            savingsText = `
                <div class="annual-savings">
                    <div class="savings-badge">¡AHORRO!</div>
                    <div class="savings-text">
                        <span class="savings-percent">${savingsPercent}%</span>
                        <span class="savings-amount">€${formatNumber(savings.toFixed(2))}</span>
                    </div>
                    <div class="savings-info">Comparado con facturación mensual</div>
                </div>
            `;
        }
        
        card.innerHTML = `
            <div class="pricing-header">
                <h3 class="plan-name">${plan.name}</h3>
                <div class="plan-price"><span class="price-currency">€</span>${price.toFixed(2)}<span class="price-period">/${isYearly ? 'año' : 'mes'}</span></div>
                ${savingsText}
                <p class="pricing-description">${plan.description}</p>
            </div>
            <ul class="feature-list">
                <li class="feature-item">
                    <span class="feature-icon">✓</span> 
                    <span><span class="feature-highlight">${formatNumber(plan.features.apiCallsPerMonth)}</span> llamadas API por mes</span>
                </li>
                <li class="feature-item">
                    <span class="feature-icon">✓</span>
                    <span><span class="feature-highlight">${formatNumber(plan.features.modelTrainingPerMonth)}</span> entrenamientos de modelo por mes</span>
                </li>
                <li class="feature-item">
                    <span class="feature-icon">✓</span>
                    <span><span class="feature-highlight">${formatStorage(plan.features.storageLimit)}</span> de almacenamiento</span>
                </li>
                <li class="feature-item">
                    <span class="feature-icon ${plan.features.advancedModels ? '' : 'no'}">${plan.features.advancedModels ? '✓' : '✕'}</span>
                    <span>Acceso a modelos avanzados</span>
                </li>
                <li class="feature-item">
                    <span class="feature-icon ${plan.features.prioritySupport ? '' : 'no'}">${plan.features.prioritySupport ? '✓' : '✕'}</span>
                    <span>Soporte técnico prioritario</span>
                </li>
                <li class="feature-item">
                    <span class="feature-icon ${plan.features.customModels ? '' : 'no'}">${plan.features.customModels ? '✓' : '✕'}</span>
                    <span>Entrenamiento de modelos personalizados</span>
                </li>
                <li class="feature-item">
                    <span class="feature-icon ${plan.features.dedicatedResources ? '' : 'no'}">${plan.features.dedicatedResources ? '✓' : '✕'}</span>
                    <span>Recursos computacionales dedicados</span>
                </li>
            </ul>
            <button class="pricing-cta" data-plan="${plan.id}" data-billing="${billingPeriod}">
                ${plan.id === 'free' ? 'Comenzar Ahora' : (plan.id === 'basic' ? 'Elegir Básico' : (plan.id === 'pro' ? 'Elegir Profesional' : 'Elegir Empresarial'))}
            </button>
        `;
        
        container.appendChild(card);
        
        // Añadir evento al botón de suscripción
        const ctaButton = card.querySelector('.pricing-cta');
        ctaButton.addEventListener('click', () => processSubscription(plan.id, billingPeriod));
    });
    
    // Si tenemos datos de suscripción del usuario, resaltar su plan actual
    if (userSubscriptionData) {
        highlightCurrentPlan(userSubscriptionData.planId);
    }
}

/**
 * Procesa una nueva suscripción o cambio de plan
 * @param {string} planId ID del plan seleccionado
 * @param {string} billingPeriod Período de facturación (monthly/yearly)
 */
async function processSubscription(planId, billingPeriod) {
    try {
        // Si no estamos autenticados, redirigir al login
        const isAuthenticated = await checkAuthentication();
        if (!isAuthenticated) {
            window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            return;
        }

        // Si ya tenemos este plan actualmente, no hacer nada
        if (userSubscriptionData && userSubscriptionData.planId === planId) {
            showNotification('Ya tienes este plan activo', 'info');
            return;
        }
        
        // Mostrar indicador de carga
        showLoadingOverlay('Procesando tu solicitud...');
        
        const response = await fetch('/api/subscription/upgrade', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify({
                planId,
                interval: billingPeriod
            })
        });
        
        const data = await response.json();
        
        // Ocultar indicador de carga
        hideLoadingOverlay();
        
        if (data.success) {
            if (data.redirectUrl) {
                // Para planes de pago, redirigir a Stripe
                window.location.href = data.redirectUrl;
            } else {
                // Para plan gratuito o downgrade, mostrar mensaje y recargar la información
                showNotification(data.message || 'Plan actualizado correctamente', 'success');
                await loadCurrentSubscription();
            }
        } else {
            throw new Error(data.message || 'Error al procesar la suscripción');
        }
    } catch (error) {
        hideLoadingOverlay();
        console.error('Error al procesar la suscripción:', error);
        showNotification(error.message || 'Error al procesar la suscripción', 'error');
    }
}

/**
 * Cancela la suscripción actual
 */
async function cancelSubscription() {
    try {
        if (!confirm('¿Estás seguro que deseas cancelar tu suscripción? Seguirás teniendo acceso hasta el final del período actual.')) {
            return;
        }
        
        // Mostrar indicador de carga
        showLoadingOverlay('Cancelando suscripción...');
        
        const response = await fetch('/api/subscription/cancel', {
            method: 'POST',
            credentials: 'include'
        });
        
        const data = await response.json();
        
        // Ocultar indicador de carga
        hideLoadingOverlay();
        
        if (data.success) {
            showNotification(data.message || 'Suscripción cancelada correctamente', 'success');
            await loadCurrentSubscription();
        } else {
            throw new Error(data.message || 'Error al cancelar la suscripción');
        }
    } catch (error) {
        hideLoadingOverlay();
        console.error('Error al cancelar suscripción:', error);
        showNotification(error.message || 'Error al cancelar la suscripción', 'error');
    }
}

/**
 * Configura los listeners de eventos
 */
function setupEventListeners() {    // Toggle de billing period
    const billingToggle = document.getElementById('billingToggle');
    if (billingToggle) {
        billingToggle.addEventListener('change', function() {
            const isYearly = this.checked;
            
            // Actualizar precios en tarjetas existentes en lugar de renderizar todo nuevamente
            if (window.plansData) {
                // Actualizar los precios y textos de ahorro directamente
                document.querySelectorAll('.pricing-card').forEach(card => {
                    const planId = card.dataset.planId;
                    const plan = window.plansData.find(p => p.id === planId);
                    
                    if (plan) {
                        // Actualizar el precio mostrado
                        const priceElement = card.querySelector('.plan-price');
                        if (priceElement) {
                            const price = isYearly ? plan.price.yearly : plan.price.monthly;
                            priceElement.innerHTML = `<span class="price-currency">€</span>${price.toFixed(2)}<span class="price-period">/${isYearly ? 'año' : 'mes'}</span>`;
                        }
                        
                        // Actualizar o eliminar el bloque de ahorro
                        const existingSavings = card.querySelector('.annual-savings');
                        
                        if (isYearly && plan.id !== 'free') {
                            // Calcular el ahorro
                            const monthlyCost = plan.price.monthly * 12;
                            const yearlyCost = plan.price.yearly;
                            const savings = monthlyCost - yearlyCost;
                            const savingsPercent = Math.round((savings / monthlyCost) * 100);
                            
                            // Si ya existe un bloque de ahorro, actualizarlo
                            if (existingSavings) {
                                const savingsPercentElement = existingSavings.querySelector('.savings-percent');
                                const savingsAmountElement = existingSavings.querySelector('.savings-amount');
                                
                                if (savingsPercentElement) {
                                    savingsPercentElement.textContent = `${savingsPercent}%`;
                                }
                                if (savingsAmountElement) {
                                    savingsAmountElement.textContent = `€${formatNumber(savings.toFixed(2))}`;
                                }
                            } else {
                                // Si no existe, crear el bloque de ahorro
                                const savingsHTML = `
                                    <div class="annual-savings">
                                        <div class="savings-badge">¡AHORRO!</div>
                                        <div class="savings-text">
                                            <span class="savings-percent">${savingsPercent}%</span>
                                            <span class="savings-amount">€${formatNumber(savings.toFixed(2))}</span>
                                        </div>
                                        <div class="savings-info">Comparado con facturación mensual</div>
                                    </div>
                                `;
                                
                                // Insertar después del precio
                                if (priceElement) {
                                    priceElement.insertAdjacentHTML('afterend', savingsHTML);
                                }
                            }
                        } else if (existingSavings) {
                            // Si cambiamos a mensual, eliminar el bloque de ahorro
                            existingSavings.remove();
                        }
                        
                        // Actualizar el botón CTA para reflejar el período de facturación
                        const ctaButton = card.querySelector('.pricing-cta');
                        if (ctaButton) {
                            ctaButton.dataset.billing = isYearly ? 'yearly' : 'monthly';
                        }
                    }
                });
            } else {
                // Si no tenemos los datos de planes, renderizar todo de nuevo (fallback)
                renderPricingCards(window.plansData);
            }
        });
    }
    
    // Listener para los botones de suscripción
    document.addEventListener('click', function(event) {
        if (event.target.classList.contains('pricing-cta') || event.target.closest('.pricing-cta')) {
            event.preventDefault();
            const button = event.target.classList.contains('pricing-cta') ? event.target : event.target.closest('.pricing-cta');
            const planId = button.dataset.plan;
            const billing = button.dataset.billing || 'monthly';
            
            // Si es el plan actual, no hacer nada
            if (button.classList.contains('current')) {
                showNotification('Ya estás suscrito a este plan', 'info');
                return;
            }
            
            // Redirigir a la página de pago con los parámetros del plan
            window.location.href = `/pago.html?plan=${planId}&billing=${billing}`;
        }
    });
      // Toggle de FAQ con animación mejorada
    document.querySelectorAll('.faq-question').forEach(question => {
        question.addEventListener('click', () => {
            const item = question.parentElement;
            const answer = item.querySelector('.faq-answer');
            
            // Si ya está activo, desactivarlo con animación
            if (item.classList.contains('active')) {
                // Primero animamos la altura
                answer.style.height = answer.scrollHeight + 'px';
                // Forzamos un reflow para que la transición funcione
                answer.offsetHeight;
                // Luego establecemos la altura a 0
                answer.style.height = '0px';
                // Esperamos a que termine la transición para quitar la clase active
                setTimeout(() => {
                    item.classList.remove('active');
                    answer.style.height = '';
                }, 300); // Duración de la transición
            } else {
                // Opcionalmente, cerrar otros elementos abiertos con animación
                document.querySelectorAll('.faq-item.active').forEach(activeItem => {
                    if (activeItem !== item) {
                        const activeAnswer = activeItem.querySelector('.faq-answer');
                        activeAnswer.style.height = activeAnswer.scrollHeight + 'px';
                        activeAnswer.offsetHeight;
                        activeAnswer.style.height = '0px';
                        setTimeout(() => {
                            activeItem.classList.remove('active');
                            activeAnswer.style.height = '';
                        }, 300);
                    }
                });
                
                // Activar el elemento actual con animación
                item.classList.add('active');
                // Establecer altura a 0 primero
                answer.style.height = '0px';
                // Forzar reflow
                answer.offsetHeight;
                // Establecer altura a scrollHeight para animar la apertura
                answer.style.height = answer.scrollHeight + 'px';
                // Limpiar la altura después de la transición
                setTimeout(() => {
                    answer.style.height = '';
                }, 300);
            }
        });
    });
        });
    });
}

/**
 * Procesa los parámetros de URL
 */
function handleUrlParameters() {
    const urlParams = new URLSearchParams(window.location.search);
    
    // Procesar redirección desde Stripe
    const success = urlParams.get('success');
    const canceled = urlParams.get('canceled');
    
    if (success === 'true') {
        showNotification('¡Tu suscripción se ha procesado correctamente!', 'success');
        // Limpiar la URL
        window.history.replaceState({}, document.title, window.location.pathname);
    } else if (canceled === 'true') {
        showNotification('Has cancelado el proceso de pago', 'info');
        // Limpiar la URL
        window.history.replaceState({}, document.title, window.location.pathname);
    }
    
    // Procesar selección de plan desde la página de planes
    const planId = urlParams.get('plan');
    const billing = urlParams.get('billing');
    
    if (planId && billing) {
        // Intentar seleccionar el plan automáticamente
        processSubscription(planId, billing === 'anual' ? 'yearly' : 'monthly');
        // Limpiar la URL
        window.history.replaceState({}, document.title, window.location.pathname);
    }
}

/**
 * Muestra un mensaje de error en el contenedor de planes
 * @param {string} message Mensaje de error
 */
function showError(message) {
    const container = document.getElementById('pricing-container');
    if (!container) return;
    
    container.innerHTML = `
        <div class="error-message">
            <p>${message}</p>
            <button onclick="loadSubscriptionPlans()">Reintentar</button>
        </div>
    `;
}

/**
 * Muestra una notificación en pantalla
 * @param {string} message Mensaje a mostrar
 * @param {string} type Tipo de notificación: success, error, info
 */
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
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
    }, 5000);
}

/**
 * Calcula el porcentaje para las barras de uso
 * @param {number} used Cantidad usada
 * @param {number} limit Límite máximo
 * @returns {number} Porcentaje de uso
 */
function calculatePercentage(used, limit) {
    if (!limit) return 0;
    return Math.min(Math.round((used / limit) * 100), 100);
}

/**
 * Formatea un valor numérico con separadores de miles
 * @param {number} num Número a formatear
 * @returns {string} Número formateado
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
}

/**
 * Formatea un valor de almacenamiento en MB
 * @param {number} sizeInMB Tamaño en megabytes
 * @returns {string} Tamaño formateado
 */
function formatStorage(sizeInMB) {
    if (sizeInMB < 1024) {
        return `${sizeInMB} MB`;
    } else {
        return `${(sizeInMB / 1024).toFixed(1)} GB`;
    }
}

/**
 * Muestra un overlay de carga
 * @param {string} message Mensaje a mostrar
 */
function showLoadingOverlay(message = 'Cargando...') {
    // Primero eliminar cualquier overlay existente
    hideLoadingOverlay();
    
    const overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.innerHTML = `
        <div class="loading-spinner"></div>
        <p>${message}</p>
    `;
    
    document.body.appendChild(overlay);
    
    // Forzar reflow
    overlay.offsetHeight;
    
    // Mostrar con animación
    overlay.classList.add('visible');
}

/**
 * Oculta el overlay de carga
 */
function hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.remove('visible');
        
        // Esperar a que termine la animación antes de eliminar
        setTimeout(() => {
            overlay.remove();
        }, 300);
    }
}

/**
 * Devuelve el texto descriptivo del estado de suscripción
 * @param {string} status Estado de la suscripción
 * @returns {string} Texto descriptivo
 */
function getStatusText(status) {
    switch (status) {
        case 'active': return 'Activa';
        case 'canceled': return 'Cancelada';
        case 'past_due': return 'Pago pendiente';
        case 'trial': return 'Prueba';
        case 'unpaid': return 'Impago';
        default: return 'Desconocido';
    }
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', initSubscriptionPage);
