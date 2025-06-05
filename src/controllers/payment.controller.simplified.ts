import { Request, Response } from 'express';
import { User } from '../models/user.model';
import { getPlanById, SubscriptionTransaction } from '../models/subscription.model.simplified';
import logger from '../utils/logger';
import Stripe from 'stripe';

// Configurar cliente de Stripe
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || '', {
  apiVersion: '2023-10-16' as any
});

// Webhook secret para verificar eventos de Stripe
const endpointSecret = process.env.STRIPE_WEBHOOK_SECRET || '';

/**
 * Controlador para gestión de pagos - Versión simplificada para primera iteración
 */
export class PaymentController {

  /**
   * Obtiene los planes de suscripción disponibles
   */
  public getSubscriptionPlans = async (req: Request, res: Response): Promise<void> => {
    try {
      // Obtener planes de suscripción
      const plans = await getPlanById('all');
      
      res.json({
        success: true,
        plans
      });
    } catch (error: any) {
      logger.error(`Error al obtener planes de suscripción: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al obtener planes de suscripción',
        error: error.message
      });
    }
  };

  /**
   * Endpoint para procesar los webhooks de Stripe
   */
  public handleWebhook = async (req: Request, res: Response): Promise<void> => {
    try {
      const sig = req.headers['stripe-signature'] as string;
      
      // Versión simplificada - solo logueamos el evento
      logger.info('Webhook de Stripe recibido');
      
      // Para esta iteración, simplemente devolvemos éxito
      res.json({ received: true });
    } catch (error: any) {
      logger.error(`Error en webhook de Stripe: ${error.message}`);
      res.status(400).json({
        success: false,
        message: 'Error en webhook de Stripe',
        error: error.message
      });
    }
  };

  /**
   * Crea una suscripción (versión simplificada)
   */
  public createSubscription = async (req: Request, res: Response): Promise<void> => {
    try {
      const { planId } = req.body;
      const user = req.user;
      
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
        return;
      }
      
      // Actualizar el plan del usuario (versión simplificada)
      await User.findByIdAndUpdate(user._id, {
        subscriptionPlan: planId,
        subscriptionStatus: 'active',
        subscriptionStartDate: new Date(),
        subscriptionExpiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // 30 días
      });
      
      res.status(201).json({
        success: true,
        message: `Suscripción al plan ${planId} creada correctamente`
      });
      
    } catch (error: any) {
      logger.error(`Error al crear suscripción: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al crear suscripción',
        error: error.message
      });
    }
  };

  /**
   * Cancela una suscripción (versión simplificada)
   */
  public cancelSubscription = async (req: Request, res: Response): Promise<void> => {
    try {
      const user = req.user;
      
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
        return;
      }
      
      // Actualizar el estado de la suscripción (versión simplificada)
      await User.findByIdAndUpdate(user._id, {
        cancelAtPeriodEnd: true
      });
      
      res.json({
        success: true,
        message: 'La suscripción se cancelará al final del período actual'
      });
      
    } catch (error: any) {
      logger.error(`Error al cancelar suscripción: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al cancelar suscripción',
        error: error.message
      });
    }
  };

  /**
   * Obtiene el historial de facturación (simplificado)
   */
  public getBillingHistory = async (req: Request, res: Response): Promise<void> => {
    try {
      const user = req.user;
      
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
        return;
      }
      
      // Datos de facturación mockup para la primera iteración
      const billingHistory = [
        {
          id: 'inv_123456',
          amount: 29.99,
          status: 'paid',
          created: new Date(),
          periodStart: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
          periodEnd: new Date(),
          planId: 'basic'
        }
      ];
      
      res.json({
        success: true,
        billingHistory
      });
      
    } catch (error: any) {
      logger.error(`Error al obtener historial de facturación: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al obtener historial de facturación',
        error: error.message
      });
    }
  };
}

export default PaymentController;
