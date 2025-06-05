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
router.get('/plans', paymentController.getSubscriptionPlans);

/**
 * @route   POST /api/payment/webhook
 * @desc    Webhook para eventos de Stripe
 * @access  Público (pero verificado con firma de Stripe)
 */
router.post('/webhook', paymentController.handleWebhook);

/**
 * @route   POST /api/payment/subscription
 * @desc    Crea una nueva suscripción
 * @access  Privado
 */
router.post('/subscription',
    authenticateJWT,
    paymentController.createSubscription
);

/**
 * @route   DELETE /api/payment/subscription
 * @desc    Cancela una suscripción existente
 * @access  Privado
 */
router.delete('/subscription',
    authenticateJWT,
    paymentController.cancelSubscription
);

/**
 * @route   GET /api/payment/billing-history
 * @desc    Obtiene el historial de facturación del usuario
 * @access  Privado
 */
router.get('/billing-history',
    authenticateJWT,
    paymentController.getBillingHistory
);

export default router;