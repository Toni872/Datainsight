import { Router } from 'express';
import { authenticateJWT } from '../middlewares/auth.middleware';
import { checkAdminRole } from '../middlewares/admin.middleware';
import { AdminSubscriptionController } from '../controllers/admin/subscription.controller';

const router = Router();
const adminSubscriptionController = new AdminSubscriptionController();

/**
 * @route   GET /api/admin/subscriptions
 * @desc    Obtiene todas las suscripciones con filtros y paginación
 * @access  Admin
 */
router.get('/subscriptions', 
    authenticateJWT, 
    checkAdminRole,
    adminSubscriptionController.getAllSubscriptions
);

/**
 * @route   GET /api/admin/subscription-stats
 * @desc    Obtiene estadísticas de suscripciones para el panel de administración
 * @access  Admin
 */
router.get('/subscription-stats', 
    authenticateJWT, 
    checkAdminRole,
    adminSubscriptionController.getSubscriptionStats
);

/**
 * @route   GET /api/admin/users/:userId/subscription
 * @desc    Obtiene los detalles de suscripción de un usuario específico
 * @access  Admin
 */
router.get('/users/:userId/subscription', 
    authenticateJWT, 
    checkAdminRole,
    adminSubscriptionController.getUserSubscriptionDetails
);

/**
 * @route   POST /api/admin/users/:userId/change-plan
 * @desc    Cambia el plan de suscripción de un usuario
 * @access  Admin
 */
router.post('/users/:userId/change-plan', 
    authenticateJWT, 
    checkAdminRole,
    adminSubscriptionController.changeUserPlan
);

/**
 * @route   POST /api/admin/users/:userId/extend-subscription
 * @desc    Extiende la suscripción de un usuario
 * @access  Admin
 */
router.post('/users/:userId/extend-subscription', 
    authenticateJWT, 
    checkAdminRole,
    adminSubscriptionController.extendUserSubscription
);

/**
 * @route   POST /api/admin/users/:userId/reset-quotas
 * @desc    Resetea las cuotas mensuales de un usuario
 * @access  Admin
 */
router.post('/users/:userId/reset-quotas', 
    authenticateJWT, 
    checkAdminRole,
    adminSubscriptionController.resetUserQuotas
);

/**
 * @route   POST /api/admin/users/:userId/cancel-subscription
 * @desc    Cancela la suscripción de un usuario
 * @access  Admin
 */
router.post('/users/:userId/cancel-subscription', 
    authenticateJWT, 
    checkAdminRole,
    adminSubscriptionController.cancelUserSubscription
);

export default router;
