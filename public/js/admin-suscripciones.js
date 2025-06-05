/**
 * Panel de Administración de Suscripciones - Mi Proyecto
 * Script para gestionar suscripciones de usuarios desde el panel de administración
 */

// Variables globales
let currentPage = 1;
let totalPages = 1;
let subscriptionsData = [];
let currentFilters = {
    plan: '',
    status: '',
    search: ''
};
let selectedUserId = null;
let subscriptionsChart = null;

// DOM Elements
const adminContent = document.getElementById('adminContent');
const subscriptionsTableBody = document.getElementById('subscriptionsTableBody');
const paginationElement = document.getElementById('pagination');
const filterForm = document.getElementById('filterForm');
const planFilter = document.getElementById('planFilter');
const statusFilter = document.getElementById('statusFilter');
const searchFilter = document.getElementById('searchFilter');
const exportBtn = document.getElementById('exportBtn');

// Modal Elements
const userDetailsModal = new bootstrap.Modal(document.getElementById('userDetailsModal'));
const changePlanModal = new bootstrap.Modal(document.getElementById('changePlanModal'));
const extendSubscriptionModal = new bootstrap.Modal(document.getElementById('extendSubscriptionModal'));
const loadingSpinner = new bootstrap.Modal(document.getElementById('loadingSpinner'));

// Modal Action Buttons
const btnChangePlan = document.getElementById('btnChangePlan');
const btnExtendSubscription = document.getElementById('btnExtendSubscription');
const btnResetQuotas = document.getElementById('btnResetQuotas');
const btnCancelSubscription = document.getElementById('btnCancelSubscription');
const btnSaveChanges = document.getElementById('btnSaveChanges');
const btnConfirmPlanChange = document.getElementById('btnConfirmPlanChange');
const btnConfirmExtension = document.getElementById('btnConfirmExtension');

/**
 * Inicialización cuando se carga la página
 */
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Verificar autenticación de administrador
        if (!await checkAdminAuth()) {
            window.location.href = '../perfil.html';
            return;
        }
        
        // Configurar eventos
        setupEventListeners();
        
        // Cargar datos iniciales
        await Promise.all([
            loadSubscriptions(),
            loadSubscriptionStats()
        ]);
        
        // Inicializar gráficos
        initializeCharts();
    } catch (error) {
        console.error('Error al inicializar página:', error);
        showAlert('danger', 'Error al cargar los datos. Por favor, recarga la página.');
    }
});

/**
 * Verifica si el usuario tiene permisos de administrador
 */
async function checkAdminAuth() {
    try {
        const token = localStorage.getItem('token');
        
        if (!token) {
            return false;
        }
        
        const response = await fetch('/api/users/me', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            return false;
        }
        
        const data = await response.json();
        
        // Verificar si el usuario es administrador
        if (data.success && data.user && data.user.role === 'admin') {
            // Mostrar nombre de admin
            document.getElementById('adminName').textContent = data.user.name || data.user.email;
            return true;
        }
        
        return false;
    } catch (error) {
        console.error('Error al verificar permisos de administrador:', error);
        return false;
    }
}

/**
 * Configura los eventos para los elementos interactivos
 */
function setupEventListeners() {
    // Filtros
    filterForm.addEventListener('submit', (e) => {
        e.preventDefault();
        currentPage = 1; // Resetear página
        currentFilters = {
            plan: planFilter.value,
            status: statusFilter.value,
            search: searchFilter.value
        };
        loadSubscriptions();
    });
    
    // Exportar datos
    exportBtn.addEventListener('click', exportSubscriptionsData);
    
    // Modal de detalles de usuario - Botones
    btnChangePlan.addEventListener('click', () => {
        changePlanModal.show();
    });
    
    btnExtendSubscription.addEventListener('click', () => {
        extendSubscriptionModal.show();
    });
    
    btnResetQuotas.addEventListener('click', resetUserQuotas);
    btnCancelSubscription.addEventListener('click', cancelUserSubscription);
    
    // Modal de cambio de plan
    btnConfirmPlanChange.addEventListener('click', handlePlanChange);
    
    // Modal de extensión de suscripción
    btnConfirmExtension.addEventListener('click', handleSubscriptionExtension);
    
    // Logout
    document.getElementById('logoutBtn').addEventListener('click', (e) => {
        e.preventDefault();
        localStorage.removeItem('token');
        window.location.href = '../index.html';
    });
}

/**
 * Carga los datos de suscripciones para mostrar en la tabla
 */
async function loadSubscriptions() {
    try {
        showLoading('Cargando datos de suscripciones...');
        
        const token = localStorage.getItem('token');
        let url = `/api/admin/subscriptions?page=${currentPage}`;
        
        // Añadir filtros si existen
        if (currentFilters.plan) url += `&plan=${currentFilters.plan}`;
        if (currentFilters.status) url += `&status=${currentFilters.status}`;
        if (currentFilters.search) url += `&search=${encodeURIComponent(currentFilters.search)}`;
        
        const response = await fetch(url, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Error al cargar los datos de suscripciones');
        }
        
        const data = await response.json();
        
        if (data.success) {
            subscriptionsData = data.subscriptions;
            totalPages = data.pagination.totalPages;
            currentPage = data.pagination.currentPage;
            
            renderSubscriptionsTable();
            renderPagination();
            updateSummaryStats(data.stats);
        }
        
        hideLoading();
    } catch (error) {
        console.error('Error al cargar suscripciones:', error);
        showAlert('danger', 'No se pudieron cargar los datos de suscripciones');
        hideLoading();
    }
}

/**
 * Carga las estadísticas de suscripciones para los gráficos
 */
async function loadSubscriptionStats() {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch('/api/admin/subscription-stats', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Error al cargar estadísticas');
        }
        
        const data = await response.json();
        
        if (data.success) {
            // Las estadísticas se usarán para actualizar los gráficos
            return data.stats;
        }
        
        return null;
    } catch (error) {
        console.error('Error al cargar estadísticas:', error);
        return null;
    }
}

/**
 * Actualiza los contadores de resumen en la parte superior
 */
function updateSummaryStats(stats) {
    if (!stats) return;
    
    document.getElementById('totalUsers').textContent = stats.totalUsers;
    document.getElementById('activeSubscriptions').textContent = stats.activeSubscriptions;
    document.getElementById('expiringSubscriptions').textContent = stats.expiringSubscriptions;
    document.getElementById('monthlyRevenue').textContent = `€${stats.monthlyRevenue.toFixed(2)}`;
}

/**
 * Renderiza la tabla de suscripciones
 */
function renderSubscriptionsTable() {
    // Limpiar tabla
    subscriptionsTableBody.innerHTML = '';
    
    if (subscriptionsData.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td colspan="7" class="text-center py-4">
                <i class="fas fa-info-circle me-2 text-info"></i>
                No se encontraron suscripciones que coincidan con los filtros
            </td>
        `;
        subscriptionsTableBody.appendChild(tr);
        return;
    }
    
    // Crear fila para cada suscripción
    subscriptionsData.forEach(sub => {
        const tr = document.createElement('tr');
        
        // Crear iniciales para el avatar
        const userName = sub.user.name || sub.user.email;
        const initials = userName.split(' ')
            .map(name => name.charAt(0).toUpperCase())
            .join('')
            .substring(0, 2);
        
        // Procesar fechas
        const startDate = new Date(sub.startDate).toLocaleDateString();
        const renewalDate = sub.renewalDate ? new Date(sub.renewalDate).toLocaleDateString() : 'N/A';
        
        // Calcular porcentaje de uso de API
        const apiUsagePercent = sub.quotaInfo ? 
            Math.min(100, Math.round((sub.quotaInfo.apiCalls.used / sub.quotaInfo.apiCalls.limit) * 100)) : 0;
        
        tr.innerHTML = `
            <td>
                <div class="d-flex align-items-center">
                    <div class="user-avatar me-2" style="background-color: ${getAvatarColor(sub.user._id)}">
                        ${initials}
                    </div>
                    <div>
                        <div>${sub.user.name || 'Sin nombre'}</div>
                        <small class="text-muted">${sub.user.email}</small>
                    </div>
                </div>
            </td>
            <td>
                <span class="plan-badge plan-${sub.plan.toLowerCase()}">
                    ${getPlanDisplayName(sub.plan)}
                </span>
            </td>
            <td>
                <div>
                    <span class="status-badge status-${getStatusClass(sub.status)}"></span>
                    ${getStatusDisplayName(sub.status)}
                </div>
                ${sub.cancelAtPeriodEnd ? '<small class="text-muted">Cancelación pendiente</small>' : ''}
            </td>
            <td>${startDate}</td>
            <td>${renewalDate}</td>
            <td>
                <div class="progress" style="height: 5px; width: 100px;">
                    <div class="progress-bar ${getProgressBarClass(apiUsagePercent)}" 
                         style="width: ${apiUsagePercent}%"></div>
                </div>
                <small>${apiUsagePercent}%</small>
            </td>
            <td>
                <div class="dropdown">
                    <button class="btn btn-sm btn-light dropdown-toggle" type="button" data-bs-toggle="dropdown">
                        Acciones
                    </button>
                    <ul class="dropdown-menu">
                        <li><a class="dropdown-item view-details" href="#" data-id="${sub.user._id}">
                            <i class="fas fa-info-circle me-1"></i> Ver detalles
                        </a></li>
                        <li><a class="dropdown-item edit-subscription" href="#" data-id="${sub.user._id}">
                            <i class="fas fa-edit me-1"></i> Editar
                        </a></li>
                        <li><hr class="dropdown-divider"></li>
                        <li><a class="dropdown-item reset-quotas text-warning" href="#" data-id="${sub.user._id}">
                            <i class="fas fa-redo me-1"></i> Resetear cuotas
                        </a></li>
                        <li><a class="dropdown-item cancel-subscription text-danger" href="#" data-id="${sub.user._id}">
                            <i class="fas fa-ban me-1"></i> Cancelar suscripción
                        </a></li>
                    </ul>
                </div>
            </td>
        `;
        
        subscriptionsTableBody.appendChild(tr);
    });
    
    // Añadir eventos a los botones de acción
    document.querySelectorAll('.view-details, .edit-subscription').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const userId = btn.dataset.id;
            openUserDetailsModal(userId);
        });
    });
    
    document.querySelectorAll('.reset-quotas').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const userId = btn.dataset.id;
            confirmAction(
                'Resetear Cuotas', 
                '¿Estás seguro que deseas resetear todas las cuotas mensuales de este usuario?',
                () => resetUserQuotas(userId)
            );
        });
    });
    
    document.querySelectorAll('.cancel-subscription').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const userId = btn.dataset.id;
            confirmAction(
                'Cancelar Suscripción', 
                '¿Estás seguro que deseas cancelar la suscripción de este usuario?',
                () => cancelUserSubscription(userId)
            );
        });
    });
}

/**
 * Renderiza la paginación
 */
function renderPagination() {
    paginationElement.innerHTML = '';
    
    // Si solo hay 1 página, no mostrar paginación
    if (totalPages <= 1) return;
    
    // Botón de "Anterior"
    const prevLi = document.createElement('li');
    prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `
        <a class="page-link" href="#" aria-label="Anterior">
            <span aria-hidden="true">&laquo;</span>
        </a>
    `;
    if (currentPage > 1) {
        prevLi.addEventListener('click', (e) => {
            e.preventDefault();
            currentPage--;
            loadSubscriptions();
        });
    }
    paginationElement.appendChild(prevLi);
    
    // Botones de páginas
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);
    
    for (let i = startPage; i <= endPage; i++) {
        const pageLi = document.createElement('li');
        pageLi.className = `page-item ${i === currentPage ? 'active' : ''}`;
        pageLi.innerHTML = `<a class="page-link" href="#">${i}</a>`;
        
        if (i !== currentPage) {
            pageLi.addEventListener('click', (e) => {
                e.preventDefault();
                currentPage = i;
                loadSubscriptions();
            });
        }
        
        paginationElement.appendChild(pageLi);
    }
    
    // Botón de "Siguiente"
    const nextLi = document.createElement('li');
    nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `
        <a class="page-link" href="#" aria-label="Siguiente">
            <span aria-hidden="true">&raquo;</span>
        </a>
    `;
    if (currentPage < totalPages) {
        nextLi.addEventListener('click', (e) => {
            e.preventDefault();
            currentPage++;
            loadSubscriptions();
        });
    }
    paginationElement.appendChild(nextLi);
}

/**
 * Inicializa los gráficos de la página
 */
function initializeCharts() {
    const ctx = document.getElementById('subscriptionChart').getContext('2d');
    
    // Datos de ejemplo (esto se debería reemplazar con datos reales)
    const chartData = {
        labels: ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio'],
        datasets: [
            {
                label: 'Free',
                data: [120, 125, 132, 145, 150, 162],
                borderColor: '#6c757d',
                backgroundColor: 'rgba(108, 117, 125, 0.2)',
                tension: 0.3
            },
            {
                label: 'Basic',
                data: [45, 52, 60, 65, 72, 78],
                borderColor: '#0d6efd',
                backgroundColor: 'rgba(13, 110, 253, 0.2)',
                tension: 0.3
            },
            {
                label: 'Pro',
                data: [20, 25, 28, 32, 36, 42],
                borderColor: '#198754',
                backgroundColor: 'rgba(25, 135, 84, 0.2)',
                tension: 0.3
            },
            {
                label: 'Enterprise',
                data: [3, 4, 4, 5, 7, 8],
                borderColor: '#dc3545',
                backgroundColor: 'rgba(220, 53, 69, 0.2)',
                tension: 0.3
            }
        ]
    };
    
    subscriptionsChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Evolución de Suscripciones por Plan'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Número de suscripciones'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Mes'
                    }
                }
            }
        }
    });
}

/**
 * Abre el modal de detalles de usuario
 */
async function openUserDetailsModal(userId) {
    try {
        selectedUserId = userId;
        showLoading('Cargando detalles del usuario...');
        
        // Obtener datos del usuario y su suscripción
        const token = localStorage.getItem('token');
        const response = await fetch(`/api/admin/users/${userId}/subscription`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Error al cargar los detalles del usuario');
        }
        
        const data = await response.json();
        
        if (data.success) {
            // Llenar el modal con los datos
            populateUserDetailsModal(data.userData);
            userDetailsModal.show();
        }
        
        hideLoading();
    } catch (error) {
        console.error('Error al abrir modal de detalles:', error);
        showAlert('danger', 'No se pudieron cargar los detalles del usuario');
        hideLoading();
    }
}

/**
 * Rellena el modal de detalles con la información del usuario
 */
function populateUserDetailsModal(userData) {
    // Información de usuario
    document.getElementById('userDetailId').textContent = userData.user._id;
    document.getElementById('userDetailName').textContent = userData.user.name || 'Sin nombre';
    document.getElementById('userDetailEmail').textContent = userData.user.email;
    
    // Fecha de registro
    const registeredDate = new Date(userData.user.createdAt).toLocaleDateString();
    document.getElementById('userDetailRegistered').textContent = registeredDate;
    
    // Información de suscripción
    document.getElementById('userDetailPlan').textContent = getPlanDisplayName(userData.subscription.plan);
    document.getElementById('userDetailStatus').textContent = getStatusDisplayName(userData.subscription.status);
    document.getElementById('userDetailBillingPeriod').textContent = 
        userData.subscription.billingPeriod === 'monthly' ? 'Mensual' : 'Anual';
    
    // Próxima renovación
    const nextBillingDate = userData.subscription.renewalDate ? 
        new Date(userData.subscription.renewalDate).toLocaleDateString() : 'N/A';
    document.getElementById('userDetailNextBilling').textContent = nextBillingDate;
    
    // Uso de cuotas
    if (userData.quotaInfo) {
        updateProgressBar(
            'userDetailApiProgress', 
            'userDetailApiUsage', 
            userData.quotaInfo.apiCalls.used, 
            userData.quotaInfo.apiCalls.limit
        );
        
        updateProgressBar(
            'userDetailTrainingProgress', 
            'userDetailTrainingUsage', 
            userData.quotaInfo.modelTraining.used, 
            userData.quotaInfo.modelTraining.limit
        );
        
        updateProgressBar(
            'userDetailStorageProgress', 
            'userDetailStorageUsage', 
            userData.quotaInfo.storage.used, 
            userData.quotaInfo.storage.limit, 
            true // formatear bytes
        );
    }
    
    // Historial de pagos
    const paymentsTable = document.getElementById('userDetailPayments');
    paymentsTable.innerHTML = '';
    
    if (!userData.payments || userData.payments.length === 0) {
        paymentsTable.innerHTML = `
            <tr>
                <td colspan="4" class="text-center">No hay pagos registrados</td>
            </tr>
        `;
    } else {
        userData.payments.forEach(payment => {
            const paymentDate = new Date(payment.date).toLocaleDateString();
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${paymentDate}</td>
                <td>${payment.currency || '€'} ${payment.amount.toFixed(2)}</td>
                <td>
                    <span class="badge bg-${getStatusBadgeColor(payment.status)}">
                        ${getPaymentStatusText(payment.status)}
                    </span>
                </td>
                <td>${payment.reference}</td>
            `;
            paymentsTable.appendChild(row);
        });
    }
    
    // Historial de cambios
    const historyContainer = document.getElementById('userDetailHistory');
    historyContainer.innerHTML = '';
    
    if (!userData.history || userData.history.length === 0) {
        historyContainer.innerHTML = `
            <div class="text-center text-muted">No hay cambios registrados</div>
        `;
    } else {
        userData.history.forEach(item => {
            const historyDate = new Date(item.date).toLocaleDateString();
            
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.innerHTML = `
                <small class="text-muted">${historyDate}</small>
                <div>${item.description}</div>
            `;
            historyContainer.appendChild(historyItem);
        });
    }
}

/**
 * Actualiza las barras de progreso en el modal de detalles
 */
function updateProgressBar(progressBarId, usageTextId, used, limit, isBytes = false) {
    const progressBar = document.getElementById(progressBarId);
    const usageText = document.getElementById(usageTextId);
    
    // Calcular porcentaje
    const percentage = limit > 0 ? Math.min(100, (used / limit) * 100) : 0;
    
    // Actualizar barra
    progressBar.style.width = `${percentage}%`;
    
    // Color según el porcentaje
    if (percentage >= 90) {
        progressBar.className = 'progress-bar bg-danger';
    } else if (percentage >= 70) {
        progressBar.className = 'progress-bar bg-warning';
    } else {
        progressBar.className = 'progress-bar bg-primary';
    }
    
    // Actualizar texto
    if (isBytes) {
        usageText.textContent = `${formatBytes(used)} / ${formatBytes(limit)}`;
    } else {
        usageText.textContent = `${used} / ${limit}`;
    }
}

/**
 * Maneja el cambio de plan desde el modal
 */
async function handlePlanChange() {
    try {
        if (!selectedUserId) return;
        
        const newPlan = document.getElementById('selectPlan').value;
        const billingPeriod = document.getElementById('selectBillingPeriod').value;
        const applyImmediately = document.getElementById('applyImmediately').checked;
        const note = document.getElementById('planChangeNote').value;
        
        showLoading('Cambiando plan de suscripción...');
        
        const token = localStorage.getItem('token');
        const response = await fetch(`/api/admin/users/${selectedUserId}/change-plan`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                planId: newPlan,
                billingPeriod,
                applyImmediately,
                note
            })
        });
        
        if (!response.ok) {
            throw new Error('Error al cambiar el plan');
        }
        
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', 'Plan actualizado correctamente');
            changePlanModal.hide();
            await loadSubscriptions(); // Recargar datos
            await openUserDetailsModal(selectedUserId); // Recargar modal
        } else {
            showAlert('danger', data.message || 'Error al actualizar el plan');
        }
        
        hideLoading();
    } catch (error) {
        console.error('Error al cambiar plan:', error);
        showAlert('danger', 'No se pudo cambiar el plan de suscripción');
        hideLoading();
    }
}

/**
 * Maneja la extensión de suscripción
 */
async function handleSubscriptionExtension() {
    try {
        if (!selectedUserId) return;
        
        const duration = document.getElementById('extensionDuration').value;
        const unit = document.getElementById('extensionUnit').value;
        const reason = document.getElementById('extensionReason').value;
        const note = document.getElementById('extensionNote').value;
        
        showLoading('Extendiendo suscripción...');
        
        const token = localStorage.getItem('token');
        const response = await fetch(`/api/admin/users/${selectedUserId}/extend-subscription`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                duration: parseInt(duration),
                unit,
                reason,
                note
            })
        });
        
        if (!response.ok) {
            throw new Error('Error al extender la suscripción');
        }
        
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', 'Suscripción extendida correctamente');
            extendSubscriptionModal.hide();
            await loadSubscriptions(); // Recargar datos
            await openUserDetailsModal(selectedUserId); // Recargar modal
        } else {
            showAlert('danger', data.message || 'Error al extender la suscripción');
        }
        
        hideLoading();
    } catch (error) {
        console.error('Error al extender suscripción:', error);
        showAlert('danger', 'No se pudo extender la suscripción');
        hideLoading();
    }
}

/**
 * Resetea las cuotas de un usuario
 */
async function resetUserQuotas(userId) {
    try {
        const targetUserId = userId || selectedUserId;
        if (!targetUserId) return;
        
        showLoading('Reseteando cuotas...');
        
        const token = localStorage.getItem('token');
        const response = await fetch(`/api/admin/users/${targetUserId}/reset-quotas`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Error al resetear cuotas');
        }
        
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', 'Cuotas reseteadas correctamente');
            await loadSubscriptions(); // Recargar datos
            
            if (selectedUserId === targetUserId) {
                await openUserDetailsModal(selectedUserId); // Recargar modal si está abierto
            }
        } else {
            showAlert('danger', data.message || 'Error al resetear cuotas');
        }
        
        hideLoading();
    } catch (error) {
        console.error('Error al resetear cuotas:', error);
        showAlert('danger', 'No se pudieron resetear las cuotas');
        hideLoading();
    }
}

/**
 * Cancela la suscripción de un usuario
 */
async function cancelUserSubscription(userId) {
    try {
        const targetUserId = userId || selectedUserId;
        if (!targetUserId) return;
        
        showLoading('Cancelando suscripción...');
        
        const token = localStorage.getItem('token');
        const response = await fetch(`/api/admin/users/${targetUserId}/cancel-subscription`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Error al cancelar suscripción');
        }
        
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', 'Suscripción cancelada correctamente');
            await loadSubscriptions(); // Recargar datos
            
            if (selectedUserId === targetUserId) {
                await openUserDetailsModal(selectedUserId); // Recargar modal si está abierto
            }
        } else {
            showAlert('danger', data.message || 'Error al cancelar suscripción');
        }
        
        hideLoading();
    } catch (error) {
        console.error('Error al cancelar suscripción:', error);
        showAlert('danger', 'No se pudo cancelar la suscripción');
        hideLoading();
    }
}

/**
 * Exporta los datos de suscripciones a CSV
 */
function exportSubscriptionsData() {
    try {
        if (!subscriptionsData || subscriptionsData.length === 0) {
            showAlert('warning', 'No hay datos para exportar');
            return;
        }
        
        // Crear cabeceras
        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += "Usuario,Email,Plan,Estado,Fecha Inicio,Fecha Renovación,API Calls Usados,API Calls Límite\n";
        
        // Añadir filas
        subscriptionsData.forEach(sub => {
            const row = [
                (sub.user.name || 'Sin nombre').replace(/,/g, ' '), // Evitar conflictos con comas
                sub.user.email,
                getPlanDisplayName(sub.plan),
                getStatusDisplayName(sub.status),
                new Date(sub.startDate).toLocaleDateString(),
                sub.renewalDate ? new Date(sub.renewalDate).toLocaleDateString() : 'N/A',
                sub.quotaInfo ? sub.quotaInfo.apiCalls.used : 0,
                sub.quotaInfo ? sub.quotaInfo.apiCalls.limit : 0
            ];
            
            csvContent += row.join(',') + "\n";
        });
        
        // Crear link de descarga
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `suscripciones_${new Date().toISOString().slice(0, 10)}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
    } catch (error) {
        console.error('Error al exportar datos:', error);
        showAlert('danger', 'No se pudieron exportar los datos');
    }
}

/**
 * Muestra una alerta en la página
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
    adminContent.insertBefore(alertElement, adminContent.firstChild);
    
    // Auto-ocultar después de 5 segundos
    setTimeout(() => {
        alertElement.classList.remove('show');
        setTimeout(() => alertElement.remove(), 300);
    }, 5000);
}

/**
 * Muestra el spinner de carga con mensaje personalizado
 */
function showLoading(message = 'Cargando...') {
    document.getElementById('loadingMessage').textContent = message;
    loadingSpinner.show();
}

/**
 * Oculta el spinner de carga
 */
function hideLoading() {
    loadingSpinner.hide();
}

/**
 * Muestra una confirmación para acciones importantes
 */
function confirmAction(title, message, callback) {
    if (confirm(`${title}\n\n${message}`)) {
        callback();
    }
}

/**
 * Formatea bytes a una unidad legible
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
}

/**
 * Genera un color consistente basado en un ID
 */
function getAvatarColor(id) {
    // Colores predefinidos para avatares
    const colors = [
        '#0d6efd', '#6610f2', '#6f42c1', '#d63384', '#dc3545', 
        '#fd7e14', '#ffc107', '#198754', '#20c997', '#0dcaf0'
    ];
    
    // Crear un número basado en la cadena ID
    let hash = 0;
    for (let i = 0; i < id.length; i++) {
        hash = id.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    // Asignar un color basado en el hash
    const index = Math.abs(hash) % colors.length;
    return colors[index];
}

/**
 * Devuelve el nombre de un plan para mostrar
 */
function getPlanDisplayName(planId) {
    const planNames = {
        'free': 'Free',
        'basic': 'Basic',
        'pro': 'Pro',
        'enterprise': 'Enterprise'
    };
    
    return planNames[planId] || planId;
}

/**
 * Devuelve el nombre de un estado para mostrar
 */
function getStatusDisplayName(status) {
    const statusNames = {
        'active': 'Activo',
        'canceled': 'Cancelado',
        'past_due': 'Pago Pendiente',
        'trial': 'Prueba',
        'unpaid': 'Impagado'
    };
    
    return statusNames[status] || status;
}

/**
 * Devuelve la clase CSS para el indicador de estado
 */
function getStatusClass(status) {
    switch (status) {
        case 'active':
            return 'active';
        case 'trial':
            return 'trial';
        case 'past_due':
            return 'pastdue';
        default:
            return 'inactive';
    }
}

/**
 * Devuelve la clase CSS para la barra de progreso
 */
function getProgressBarClass(percentage) {
    if (percentage >= 90) return 'bg-danger';
    if (percentage >= 70) return 'bg-warning';
    return 'bg-primary';
}

/**
 * Devuelve el color de la etiqueta para un estado de pago
 */
function getStatusBadgeColor(status) {
    switch (status) {
        case 'paid':
        case 'success':
            return 'success';
        case 'pending':
            return 'warning';
        case 'failed':
            return 'danger';
        default:
            return 'secondary';
    }
}

/**
 * Devuelve el texto para un estado de pago
 */
function getPaymentStatusText(status) {
    switch (status) {
        case 'paid':
        case 'success':
            return 'Pagado';
        case 'pending':
            return 'Pendiente';
        case 'failed':
            return 'Fallido';
        default:
            return status;
    }
}
