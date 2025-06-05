import { Router } from 'express';
import { SubscriptionController } from '../controllers/subscription.controller';
import { checkActiveSubscription } from '../middlewares/subscription.middleware';
import { authenticateJWT } from '../middlewares/auth.middleware';

const router = Router();
const subscriptionController = new SubscriptionController();

/**
 * @route   GET /api/subscription/current
 * @desc    Obtiene la información de la suscripción actual del usuario
 * @access  Privado
 */
router.get('/current', 
    authenticateJWT, 
    subscriptionController.getCurrentSubscription
);

/**
 * @route   GET /api/subscription/plans
 * @desc    Lista todos los planes de suscripción disponibles
 * @access  Público
 */
router.get('/plans', 
    subscriptionController.getSubscriptionPlans
);

/**
 * @route   POST /api/subscription/upgrade
 * @desc    Actualiza el plan de suscripción del usuario
 * @access  Privado
 */
router.post('/upgrade', 
    authenticateJWT, 
    subscriptionController.upgradeSubscription
);

/**
 * @route   POST /api/subscription/cancel
 * @desc    Cancela la suscripción del usuario al final del período actual
 * @access  Privado
 */
router.post('/cancel', 
    authenticateJWT, 
    checkActiveSubscription,
    subscriptionController.cancelSubscription
);

/**
 * @route   GET /api/subscription/usage
 * @desc    Obtiene las estadísticas de uso del usuario
 * @access  Privado
 */
router.get('/usage', 
    authenticateJWT, 
    subscriptionController.getUsageStatistics
);

/**
 * @route   GET /api/subscription/invoices
 * @desc    Obtiene el historial de facturas del usuario
 * @access  Privado
 */
router.get('/invoices', 
    authenticateJWT, 
    subscriptionController.getInvoicesHistory
);

/**
 * @route   POST /api/subscription/process
 * @desc    Procesa una nueva suscripción después de un pago exitoso
 * @access  Privado
 */
router.post('/process', 
    authenticateJWT, 
    subscriptionController.processNewSubscription
);

/**
 * @route   POST /api/subscription/webhook
 * @desc    Maneja webhooks de Stripe para eventos de suscripción
 * @access  Público (pero verificado con firma de Stripe)
 */
router.post('/webhook',
    // No usar JSON parser aquí, necesitamos el raw body para verificar la firma
    subscriptionController.handleStripeWebhook
);

export default router;