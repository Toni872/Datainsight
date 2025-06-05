/**
 * Actualización de precios en tiempo real para la página de suscripciones
 * Este script mejora la funcionalidad del interruptor de facturación 
 * para actualizar los precios inmediatamente sin recargar todas las tarjetas
 */

document.addEventListener('DOMContentLoaded', function() {
    // Buscar el interruptor de facturación
    const billingToggle = document.getElementById('billingToggle');
    if (!billingToggle) return;

    // Reemplazar el evento original por nuestra versión mejorada
    billingToggle.addEventListener('change', function() {
        const isYearly = this.checked;
        
        // Solo proceder si tenemos datos de planes disponibles
        if (!window.plansData) return;
        
        // Actualizar cada tarjeta de precios
        document.querySelectorAll('.pricing-card').forEach(card => {
            const planId = card.dataset.planId;
            const plan = window.plansData.find(p => p.id === planId);
            
            if (!plan) return;
            
            // 1. Actualizar el precio mostrado
            const priceElement = card.querySelector('.plan-price');
            if (priceElement) {
                const price = isYearly ? plan.price.yearly : plan.price.monthly;
                priceElement.innerHTML = `<span class="price-currency">€</span>${price.toFixed(2)}<span class="price-period">/${isYearly ? 'año' : 'mes'}</span>`;
            }
            
            // 2. Manejar el bloque de ahorro anual
            const existingSavings = card.querySelector('.annual-savings');
            
            if (isYearly && plan.id !== 'free') {
                // Calcular el ahorro
                const monthlyCost = plan.price.monthly * 12;
                const yearlyCost = plan.price.yearly;
                const savings = monthlyCost - yearlyCost;
                const savingsPercent = Math.round((savings / monthlyCost) * 100);
                
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
                
                if (existingSavings) {
                    // Actualizar el contenido
                    existingSavings.innerHTML = savingsHTML;
                } else {
                    // Crear y añadir después del precio
                    priceElement.insertAdjacentHTML('afterend', savingsHTML);
                }
            } else if (existingSavings) {
                // Eliminar el bloque de ahorro al cambiar a facturación mensual
                existingSavings.remove();
            }
            
            // 3. Actualizar el botón CTA
            const ctaButton = card.querySelector('.pricing-cta');
            if (ctaButton) {
                ctaButton.dataset.billing = isYearly ? 'yearly' : 'monthly';
            }
        });
    });
    
    // Función auxiliar para formatear números (copia de la función original)
    function formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
    }
});
