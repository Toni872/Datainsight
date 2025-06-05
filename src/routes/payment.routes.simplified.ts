import { Router } from 'express';
import { PaymentController } from '../controllers/payment.controller.simplified';
import { authenticateJWT } from '../middlewares/auth.middleware';

const router = Router();
const paymentController = new PaymentController();

/**
 * @route   GET /api/payment/plans
 * @desc    Obtiene los planes de suscripción disponibles
 * @access  Público
 */
router.get('/plans', 
    paymentController.getSubscriptionPlans
);

/**
 * @route   POST /api/payment/create-subscription
 * @desc    Crea una suscripción (versión simplificada)
 * @access  Privado
 */
router.post('/create-subscription',
    authenticateJWT,
    paymentController.createSubscription
);

/**
 * @route   POST /api/payment/cancel-subscription
 * @desc    Cancela una suscripción (versión simplificada)
 * @access  Privado
 */
router.post('/cancel-subscription',
    authenticateJWT,
    paymentController.cancelSubscription
);

/**
 * @route   GET /api/payment/billing-history
 * @desc    Obtiene el historial de facturación (versión simplificada)
 * @access  Privado
 */
router.get('/billing-history',
    authenticateJWT,
    paymentController.getBillingHistory
);

/**
 * @route   POST /api/payment/webhook
 * @desc    Webhook para eventos de Stripe
 * @access  Público (pero verificado con firma de Stripe)
 */
router.post('/webhook',
    paymentController.handleWebhook
);

export default router;
