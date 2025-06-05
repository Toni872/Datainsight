/**
 * Script para la página de suscripción
 * Maneja la visualización de planes, el estado de la suscripción actual y las operaciones relacionadas
 */

// Variables globales
let currentUser = null;
let currentSubscription = null;
let subscriptionPlans = [];
let isYearlyBilling = false;

// Elementos DOM principales
const authRequiredMessage = document.getElementById('authRequiredMessage');
const subscriptionContent = document.getElementById('subscriptionContent');
const currentPlanSection = document.getElementById('currentPlanSection');
const plansContainer = document.getElementById('plansContainer');
const invoicesSection = document.getElementById('invoicesSection');
const invoicesTableBody = document.getElementById('invoicesTableBody');

// Indicadores de uso
const apiCallsBar = document.getElementById('apiCallsBar');
const apiCallsUsage = document.getElementById('apiCallsUsage');
const modelTrainingBar = document.getElementById('modelTrainingBar');
const modelTrainingUsage = document.getElementById('modelTrainingUsage');
const storageBar = document.getElementById('storageBar');
const storageUsage = document.getElementById('storageUsage');

// Botones y controles
const viewInvoicesBtn = document.getElementById('viewInvoicesBtn');
const hideInvoicesBtn = document.getElementById('hideInvoicesBtn');
const cancelSubscriptionBtn = document.getElementById('cancelSubscriptionBtn');
const billingToggle = document.getElementById('billingToggle');

// Popup de confirmación
const confirmationPopup = document.getElementById('confirmationPopup');
const popupClose = document.getElementById('popupClose');
const popupTitle = document.getElementById('popupTitle');
const popupMessage = document.getElementById('popupMessage');
const popupCancelBtn = document.getElementById('popupCancelBtn');
const popupConfirmBtn = document.getElementById('popupConfirmBtn');

// Spinner de carga
const loadingSpinner = document.getElementById('loadingSpinner');

/**
 * Inicialización cuando se carga la página
 */
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Verificar autenticación
        await checkAuthentication();

        // Si el usuario está autenticado, cargar datos
        if (currentUser) {
            showLoading();

            // Cargar datos en paralelo
            await Promise.all([
                loadCurrentSubscription(),
                loadSubscriptionPlans()
            ]);

            // Configurar eventos
            setupEventListeners();

            // Verificar parámetros de URL (para manejo de retornos desde Stripe)
            handleUrlParams();

            hideLoading();
        }
    } catch (error) {
        console.error('Error al inicializar la página:', error);
        handleError('Error al cargar los datos de suscripción');
        hideLoading();
    }
});

/**
 * Verifica si el usuario está autenticado
 */
async function checkAuthentication() {
    try {
        const token = localStorage.getItem('token');
        
        if (!token) {
            showAuthRequiredMessage();
            return;
        }

        // Obtener información del usuario
        const response = await fetch('/api/users/me', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            throw new Error('Error al obtener información del usuario');
        }

        const data = await response.json();
        
        if (data.success && data.user) {
            currentUser = data.user;
            
            // Actualizar UI para usuario autenticado
            document.getElementById('username').textContent = currentUser.name || currentUser.email;
            document.getElementById('userDropdown').style.display = 'block';
            document.getElementById('loginBtn').style.display = 'none';
            showSubscriptionContent();
        } else {
            showAuthRequiredMessage();
        }
    } catch (error) {
        console.error('Error al verificar autenticación:', error);
        showAuthRequiredMessage();
    }
}

/**
 * Carga la información de la suscripción actual del usuario
 */
async function loadCurrentSubscription() {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('/api/subscription/current', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            throw new Error('Error al obtener la suscripción actual');
        }

        const data = await response.json();
        
        if (data.success && data.subscription) {
            currentSubscription = data.subscription;
            updateCurrentPlanUI();
        }
    } catch (error) {
        console.error('Error al cargar suscripción:', error);
        handleError('No se pudo cargar la información de tu suscripción actual');
    }
}

/**
 * Carga los planes de suscripción disponibles
 */
async function loadSubscriptionPlans() {
    try {
        const response = await fetch('/api/subscription/plans');
        
        if (!response.ok) {
            throw new Error('Error al obtener los planes de suscripción');
        }

        const data = await response.json();
        
        if (data.success && data.plans) {
            subscriptionPlans = data.plans;
            renderSubscriptionPlans();
        }
    } catch (error) {
        console.error('Error al cargar planes:', error);
        handleError('No se pudieron cargar los planes de suscripción disponibles');
    }
}

/**
 * Actualiza la interfaz de usuario con los datos de la suscripción actual
 */
function updateCurrentPlanUI() {
    if (!currentSubscription) return;

    // Actualizar nombre del plan y estado
    document.getElementById('currentPlanName').textContent = currentSubscription.planName;
    document.getElementById('subscriptionStatus').textContent = `Estado: ${getStatusText(currentSubscription.status)}`;
    
    // Mostrar fecha de expiración si aplica
    if (currentSubscription.expiresAt) {
        const expirationDate = new Date(currentSubscription.expiresAt);
        document.getElementById('subscriptionExpiration').textContent = 
            `Tu suscripción ${currentSubscription.cancelAtPeriodEnd ? 'se cancelará' : 'se renovará'} el ${expirationDate.toLocaleDateString()}`;
    }

    // Actualizar barras de progreso de uso
    if (currentSubscription.quotaInfo) {
        updateUsageBar(
            apiCallsBar, 
            apiCallsUsage, 
            currentSubscription.quotaInfo.apiCalls.used, 
            currentSubscription.quotaInfo.apiCalls.limit
        );
        
        updateUsageBar(
            modelTrainingBar, 
            modelTrainingUsage, 
            currentSubscription.quotaInfo.modelTraining.used, 
            currentSubscription.quotaInfo.modelTraining.limit
        );
        
        updateUsageBar(
            storageBar, 
            storageUsage, 
            currentSubscription.quotaInfo.storage.used, 
            currentSubscription.quotaInfo.storage.limit,
            true // Usar formato de bytes
        );
    }
}

/**
 * Actualiza una barra de progreso de uso
 */
function updateUsageBar(barElement, textElement, used, limit, isBytes = false) {
    // Calcular porcentaje de uso
    const percentage = limit > 0 ? Math.min(100, (used / limit) * 100) : 0;
    
    // Actualizar ancho y color de la barra
    barElement.style.width = `${percentage}%`;
    
    if (percentage >= 90) {
        barElement.className = "usage-progress bg-danger";
    } else if (percentage >= 70) {
        barElement.className = "usage-progress bg-warning";
    } else {
        barElement.className = "usage-progress bg-primary";
    }
    
    // Actualizar texto
    if (isBytes) {
        textElement.textContent = `${formatBytes(used)} / ${formatBytes(limit)}`;
    } else {
        textElement.textContent = `${used} / ${limit}`;
    }
}

/**
 * Formatea un número de bytes a una unidad legible
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
}

/**
 * Renderiza los planes de suscripción disponibles
 */
function renderSubscriptionPlans() {
    // Limpiar contenedor
    plansContainer.innerHTML = '';
    
    // Renderizar cada plan
    subscriptionPlans.forEach(plan => {
        // Seleccionar el precio según el período de facturación
        const priceInfo = plan.prices.find(p => 
            p.interval === (isYearlyBilling ? 'yearly' : 'monthly')
        );
        
        if (!priceInfo) return;
        
        // Verificar si este es el plan actual del usuario
        const isCurrentPlan = currentSubscription && 
            currentSubscription.plan === plan.id;
        
        // Crear elemento de plan
        const planElement = document.createElement('div');
        planElement.className = 'col';
        
        // Clase especial para el plan recomendado/destacado
        const isFeatured = plan.id === 'pro';
        
        planElement.innerHTML = `
            <div class="subscription-card">
                <div class="card-header ${isFeatured ? 'featured-plan' : ''}">
                    ${plan.name}
                </div>
                <div class="price-container">
                    <span class="price">${priceInfo.currency}${priceInfo.amount}</span>
                    <span class="period">/${isYearlyBilling ? 'año' : 'mes'}</span>
                </div>
                <ul class="feature-list">
                    ${plan.features.map(feature => 
                        `<li ${!feature.included ? 'class="unavailable"' : ''}>${feature.text}</li>`
                    ).join('')}
                </ul>
                <div class="card-footer">
                    ${isCurrentPlan ?
                        `<button class="btn btn-success" disabled>Plan Actual</button>` :
                        `<button class="btn btn-subscribe ${isFeatured ? 'btn-featured' : ''}" 
                         data-plan-id="${plan.id}">
                            ${plan.id === 'free' ? 'Seleccionar' : 'Suscribirse'}
                         </button>`
                    }
                </div>
            </div>
        `;
        
        plansContainer.appendChild(planElement);
    });
    
    // Añadir eventos a los botones de suscripción
    document.querySelectorAll('.btn-subscribe').forEach(button => {
        button.addEventListener('click', () => handlePlanSelection(button.dataset.planId));
    });
}

/**
 * Maneja la selección de un plan de suscripción
 */
async function handlePlanSelection(planId) {
    try {
        const selectedPlan = subscriptionPlans.find(p => p.id === planId);
        
        if (!selectedPlan) {
            throw new Error('Plan no encontrado');
        }
        
        const interval = isYearlyBilling ? 'yearly' : 'monthly';
        
        // Mostrar confirmación
        showConfirmationPopup(
            `Cambiar a plan ${selectedPlan.name}`,
            `¿Estás seguro que deseas cambiar al plan ${selectedPlan.name} con facturación ${isYearlyBilling ? 'anual' : 'mensual'}?`,
            async () => {
                showLoading();
                
                const token = localStorage.getItem('token');
                const response = await fetch('/api/subscription/upgrade', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    },
                    body: JSON.stringify({
                        planId: planId,
                        interval: interval
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Error al actualizar la suscripción');
                }
                
                const data = await response.json();
                
                // Si hay URL de redirección (para Stripe), redirigir
                if (data.redirectUrl) {
                    window.location.href = data.redirectUrl;
                } else {
                    // Sino, mostrar mensaje de éxito y recargar
                    showAlert('success', data.message || 'Plan actualizado correctamente');
                    // Recargar datos
                    await loadCurrentSubscription();
                }
                
                hideLoading();
            }
        );
    } catch (error) {
        console.error('Error al seleccionar plan:', error);
        handleError('No se pudo procesar la selección del plan');
        hideLoading();
    }
}

/**
 * Cancela la suscripción del usuario
 */
async function cancelCurrentSubscription() {
    try {
        showLoading();
        
        const token = localStorage.getItem('token');
        const response = await fetch('/api/subscription/cancel', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Error al cancelar la suscripción');
        }
        
        const data = await response.json();
        
        showAlert('success', data.message);
        await loadCurrentSubscription(); // Recargar datos
        
        hideLoading();
    } catch (error) {
        console.error('Error al cancelar suscripción:', error);
        handleError('No se pudo cancelar la suscripción');
        hideLoading();
    }
}

/**
 * Carga y muestra el historial de facturas
 */
async function loadInvoicesHistory() {
    try {
        showLoading();
        
        const token = localStorage.getItem('token');
        const response = await fetch('/api/subscription/invoices', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Error al obtener el historial de facturas');
        }
        
        const data = await response.json();
        
        if (data.success && data.invoices) {
            // Limpiar tabla
            invoicesTableBody.innerHTML = '';
            
            if (data.invoices.length === 0) {
                invoicesTableBody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center">No hay facturas disponibles</td>
                    </tr>
                `;
            } else {
                // Renderizar cada factura
                data.invoices.forEach(invoice => {
                    const row = document.createElement('tr');
                    
                    const invoiceDate = new Date(invoice.date);
                    let periodText = '';
                    
                    if (invoice.periodStart && invoice.periodEnd) {
                        const start = new Date(invoice.periodStart);
                        const end = new Date(invoice.periodEnd);
                        periodText = `${start.toLocaleDateString()} - ${end.toLocaleDateString()}`;
                    }
                    
                    row.innerHTML = `
                        <td>${invoiceDate.toLocaleDateString()}</td>
                        <td>${invoice.id}</td>
                        <td>${periodText}</td>
                        <td>${invoice.currency || '€'} ${invoice.amount.toFixed(2)}</td>
                        <td>
                            <span class="badge bg-${getStatusBadgeColor(invoice.status)}">
                                ${getInvoiceStatusText(invoice.status)}
                            </span>
                        </td>
                        <td>
                            ${invoice.pdfUrl ? 
                                `<a href="${invoice.pdfUrl}" target="_blank" class="btn btn-sm btn-outline-primary">
                                    <i class="fas fa-file-pdf"></i> PDF
                                </a>` : 
                                ''
                            }
                        </td>
                    `;
                    
                    invoicesTableBody.appendChild(row);
                });
            }
            
            // Mostrar la sección de facturas
            invoicesSection.style.display = 'block';
            
            // Ocultar cards de planes
            document.querySelector('.billing-toggle').style.display = 'none';
            plansContainer.style.display = 'none';
        }
        
        hideLoading();
    } catch (error) {
        console.error('Error al cargar facturas:', error);
        handleError('No se pudo cargar el historial de facturas');
        hideLoading();
    }
}

/**
 * Oculta la sección de facturas y vuelve a mostrar los planes
 */
function hideInvoicesHistorySection() {
    invoicesSection.style.display = 'none';
    document.querySelector('.billing-toggle').style.display = 'flex';
    plansContainer.style.display = 'flex';
}

/**
 * Configura los eventos para los elementos interactivos
 */
function setupEventListeners() {
    // Cambio de facturación (mensual/anual)
    billingToggle.addEventListener('change', () => {
        isYearlyBilling = billingToggle.checked;
        renderSubscriptionPlans();
    });
    
    // Ver facturas
    viewInvoicesBtn.addEventListener('click', loadInvoicesHistory);
    
    // Ocultar facturas
    hideInvoicesBtn.addEventListener('click', hideInvoicesHistorySection);
    
    // Cancelar suscripción
    cancelSubscriptionBtn.addEventListener('click', () => {
        showConfirmationPopup(
            'Cancelar Suscripción',
            'Tu suscripción se cancelará al final del período de facturación actual. ¿Estás seguro?',
            cancelCurrentSubscription
        );
    });
    
    // Eventos del popup de confirmación
    popupClose.addEventListener('click', hideConfirmationPopup);
    popupCancelBtn.addEventListener('click', hideConfirmationPopup);
    
    // Evento de logout
    document.getElementById('logoutBtn').addEventListener('click', (e) => {
        e.preventDefault();
        localStorage.removeItem('token');
        window.location.href = '../index.html';
    });
}

/**
 * Maneja parámetros de URL (por ejemplo, después de retornar de Stripe)
 */
function handleUrlParams() {
    const urlParams = new URLSearchParams(window.location.search);
    
    if (urlParams.has('success') && urlParams.get('success') === 'true') {
        showAlert('success', 'Tu suscripción ha sido actualizada correctamente');
    } else if (urlParams.has('canceled') && urlParams.get('canceled') === 'true') {
        showAlert('info', 'Has cancelado el proceso de pago');
    }
    
    // Limpiar URL
    if (urlParams.has('success') || urlParams.has('canceled')) {
        window.history.replaceState({}, document.title, window.location.pathname);
    }
}

/**
 * Muestra el popup de confirmación
 */
function showConfirmationPopup(title, message, confirmCallback) {
    popupTitle.textContent = title;
    popupMessage.textContent = message;
    
    // Configurar callback de confirmación
    popupConfirmBtn.onclick = () => {
        hideConfirmationPopup();
        if (typeof confirmCallback === 'function') {
            confirmCallback();
        }
    };
    
    // Mostrar popup
    confirmationPopup.style.display = 'flex';
}

/**
 * Oculta el popup de confirmación
 */
function hideConfirmationPopup() {
    confirmationPopup.style.display = 'none';
}

/**
 * Muestra un mensaje de alerta
 */
function showAlert(type, message) {
    // Crear elemento de alerta
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type} alert-dismissible fade show`;
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Insertar al principio del contenido
    subscriptionContent.insertBefore(alertElement, subscriptionContent.firstChild);
    
    // Auto-ocultar después de 5 segundos
    setTimeout(() => {
        alertElement.classList.remove('show');
        setTimeout(() => alertElement.remove(), 300);
    }, 5000);
}

/**
 * Muestra mensaje de error genérico
 */
function handleError(message) {
    showAlert('danger', message);
}

/**
 * Muestra el spinner de carga
 */
function showLoading() {
    loadingSpinner.style.display = 'flex';
}

/**
 * Oculta el spinner de carga
 */
function hideLoading() {
    loadingSpinner.style.display = 'none';
}

/**
 * Muestra el mensaje de autenticación requerida
 */
function showAuthRequiredMessage() {
    authRequiredMessage.style.display = 'block';
    subscriptionContent.style.display = 'none';
}

/**
 * Muestra el contenido de suscripción
 */
function showSubscriptionContent() {
    authRequiredMessage.style.display = 'none';
    subscriptionContent.style.display = 'block';
}

/**
 * Obtiene el texto descriptivo para un estado de suscripción
 */
function getStatusText(status) {
    switch (status) {
        case 'active':
            return 'Activa';
        case 'canceled':
            return 'Cancelada';
        case 'past_due':
            return 'Pago pendiente';
        case 'trial':
            return 'En período de prueba';
        case 'unpaid':
            return 'Impagada';
        default:
            return 'Desconocido';
    }
}

/**
 * Obtiene el texto descriptivo para un estado de factura
 */
function getInvoiceStatusText(status) {
    switch (status) {
        case 'paid':
            return 'Pagada';
        case 'pending':
            return 'Pendiente';
        case 'void':
            return 'Anulada';
        default:
            return status;
    }
}

/**
 * Obtiene el color de la etiqueta para un estado de factura
 */
function getStatusBadgeColor(status) {
    switch (status) {
        case 'paid':
            return 'success';
        case 'pending':
            return 'warning';
        case 'void':
            return 'secondary';
        default:
            return 'info';
    }
}