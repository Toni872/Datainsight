import { Request, Response } from 'express';
import { User } from '../models/user.model';
import { SUBSCRIPTION_PLANS, getPlanById, SubscriptionTransaction } from '../models/subscription.model';
import { getQuotaInfo, resetMonthlyQuotas } from '../middlewares/subscription.middleware';
import logger from '../utils/logger';
import Stripe from 'stripe';
import mongoose from 'mongoose';

// Configurar cliente de Stripe
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || '', {
  apiVersion: '2023-10-16' as any
});

/**
 * Controlador para gestión de suscripciones
 */
export class SubscriptionController {

  /**
   * Obtiene información de la suscripción actual del usuario
   */
  public getCurrentSubscription = async (req: Request, res: Response): Promise<void> => {
    try {
      const user = req.user as any; // <-- Cambia a IUser si tienes el tipo importado
      
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
        return;
      }
        // Obtener información de uso y límites
      const quotaInfo = await getQuotaInfo(user._id);
      
      // Verificar si las cuotas mensuales deben resetearse
      await resetMonthlyQuotas(user._id);
      
      // Obtener transacciones de suscripción
      const transactions = await SubscriptionTransaction.find({ userId: user._id }).sort({ createdAt: -1 }).limit(5);
      
      // Obtener plan actual
      const planDetails = getPlanById(user.subscriptionPlan);
      
      res.json({
        success: true,
        subscription: {
          plan: user.subscriptionPlan,
          planName: planDetails?.name || 'Desconocido',
          status: user.subscriptionStatus,
          billingPeriod: user.billingPeriod,
          expiresAt: user.subscriptionExpiresAt,
          cancelAtPeriodEnd: user.cancelAtPeriodEnd,
          quotaInfo,
          stripeInfo: {
            customerId: user.stripeCustomerId,
            subscriptionId: user.stripeSubscriptionId
          }
        }
      });
    } catch (error: any) {
      logger.error(`Error al obtener suscripción: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al obtener información de la suscripción'
      });
    }
  };

  /**
   * Lista todos los planes de suscripción disponibles
   */
  public getSubscriptionPlans = async (req: Request, res: Response): Promise<void> => {
    try {
      // Devolver la lista completa de planes con sus detalles
      res.json({
        success: true,
        plans: SUBSCRIPTION_PLANS
      });
    } catch (error: any) {
      logger.error(`Error al obtener planes: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al obtener los planes de suscripción'
      });
    }
  };

  /**
   * Actualiza el plan de suscripción del usuario
   */
  public upgradeSubscription = async (req: Request, res: Response): Promise<void> => {
    try {
      const user = req.user;
      const { planId, interval } = req.body;
      
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
        return;
      }
      
      // Validar el plan seleccionado
      const newPlan = getPlanById(planId);
      if (!newPlan) {
        res.status(400).json({
          success: false,
          message: 'Plan de suscripción no válido'
        });
        return;
      }
      
      // Validar el intervalo
      if (interval !== 'monthly' && interval !== 'yearly') {
        res.status(400).json({
          success: false,
          message: 'Intervalo de facturación no válido'
        });
        return;
      }
      
      // Si es un downgrade y el usuario ya tiene una suscripción activa,
      // marcar para que se aplique al final del período actual
      const isDowngrade = this.isPlanDowngrade(user.subscriptionPlan, planId);
      let updateData: any = {
        subscriptionPlan: planId,
        billingPeriod: interval
      };
      
      if (isDowngrade && user.subscriptionStatus === 'active') {
        updateData.cancelAtPeriodEnd = true;
        // No cambiar el plan inmediatamente, solo marcar para cambio al final del período
      } else {
        // Para un upgrade o nueva suscripción, aplicar cambios inmediatamente
        if (planId === 'free') {
          // Si cambia a plan gratuito, cancelar la suscripción de pago si existe
          if (user.stripeSubscriptionId) {
            await stripe.subscriptions.cancel(user.stripeSubscriptionId);
          }
          
          updateData.subscriptionStatus = 'active';
          updateData.stripeSubscriptionId = null;
        }
      }
      
      // Actualizar el usuario
      await User.findByIdAndUpdate(user._id, updateData);
        // Si es un upgrade o nueva suscripción de pago, redirigir a Stripe para el pago
      if (planId !== 'free' && !isDowngrade) {
        const priceId = interval === 'monthly' ? 
          newPlan.stripeMonthlyPriceId : 
          newPlan.stripeYearlyPriceId;
        
        if (!priceId) {
          res.status(400).json({
            success: false,
            message: 'No se encontró el precio para este plan e intervalo'
          });
          return;
        }
        
        // Crear sesión de checkout en Stripe
        const session = await stripe.checkout.sessions.create({
          customer: user.stripeCustomerId || undefined,
          payment_method_types: ['card'],
          mode: 'subscription',
          line_items: [
            {
              price: priceId,
              quantity: 1
            }
          ],
          success_url: `${process.env.FRONTEND_URL}/perfil/suscripcion.html?success=true`,
          cancel_url: `${process.env.FRONTEND_URL}/perfil/suscripcion.html?canceled=true`,          metadata: {
            userId: user._id?.toString() || '',
            planId,
            interval
          }
        });
        
        res.json({
          success: true,
          redirectUrl: session.url,
          sessionId: session.id
        });
      } else {
        // Para downgrades o cambios a plan gratuito que no requieren pago
        res.json({
          success: true,
          message: isDowngrade ? 
            'Tu plan cambiará al final del período de facturación actual' : 
            'Tu plan ha sido actualizado correctamente'
        });
      }
    } catch (error: any) {
      logger.error(`Error en upgradeSubscription: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al actualizar la suscripción'
      });
    }
  };

  /**
   * Cancela la suscripción del usuario
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
      
      // Si no tiene suscripción de pago, no hay nada que cancelar
      if (user.subscriptionPlan === 'free' || !user.stripeSubscriptionId) {
        res.status(400).json({
          success: false,
          message: 'No tienes una suscripción activa de pago para cancelar'
        });
        return;
      }
      
      // Cancelar la suscripción al final del período actual en Stripe
      await stripe.subscriptions.update(user.stripeSubscriptionId, {
        cancel_at_period_end: true
      });
      
      // Actualizar el estado en la base de datos
      await User.findByIdAndUpdate(user._id, {
        cancelAtPeriodEnd: true
      });
      
      res.json({
        success: true,
        message: 'Tu suscripción será cancelada al final del período actual'
      });
    } catch (error: any) {
      logger.error(`Error en cancelSubscription: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al cancelar la suscripción'
      });
    }
  };

  /**
   * Obtiene las estadísticas de uso del usuario
   */
  public getUsageStatistics = async (req: Request, res: Response): Promise<void> => {
    try {
      const user = req.user;
      
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
        return;
      }
        // Obtener información de cuota y uso
      const quotaInfo = await getQuotaInfo(new mongoose.Types.ObjectId(user._id?.toString() || ''));
      
      // Obtener historial de uso (simplificado, se podría expandir)
      const usageHistory = {
        apiCalls: [
          /* Aquí podrías incluir datos históricos de llamadas API por día/semana */
        ],
        modelTraining: [
          /* Aquí podrías incluir datos históricos de entrenamientos por día/semana */
        ]
      };
      
      res.json({
        success: true,
        currentUsage: quotaInfo,
        history: usageHistory
      });
    } catch (error: any) {
      logger.error(`Error en getUsageStatistics: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al obtener las estadísticas de uso'
      });
    }
  };

  /**
   * Obtiene el historial de facturas del usuario
   */
  public getInvoicesHistory = async (req: Request, res: Response): Promise<void> => {
    try {
      const user = req.user;
      
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
        return;
      }
        // Obtener transacciones de suscripción
      const invoices = await SubscriptionTransaction.find({ userId: user._id }).sort({ createdAt: -1 });
      
      // Si el usuario tiene un ID de cliente en Stripe, obtener también las facturas de Stripe
      let stripeInvoices: any[] = [];
      if (user.stripeCustomerId) {
        const stripeInvoicesData = await stripe.invoices.list({
          customer: user.stripeCustomerId,
          limit: 10
        });
        
        stripeInvoices = stripeInvoicesData.data.map(invoice => ({
          id: invoice.id,
          amount: invoice.amount_paid / 100, // Convertir de centavos a euros
          currency: invoice.currency.toUpperCase(),
          status: invoice.status,
          date: new Date(invoice.created * 1000),
          pdfUrl: invoice.invoice_pdf,
          periodStart: new Date(invoice.period_start * 1000),
          periodEnd: new Date(invoice.period_end * 1000)
        }));
      }
      
      res.json({
        success: true,
        invoices: [...invoices, ...stripeInvoices]
      });
    } catch (error: any) {
      logger.error(`Error en getInvoicesHistory: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al obtener el historial de facturas'
      });
    }
  };
  /**
   * Procesa una nueva suscripción después de un pago exitoso en Stripe
   */
  public processNewSubscription = async (req: Request, res: Response): Promise<void> => {
    try {
      const { sessionId } = req.body;
      
      if (!sessionId) {
        res.status(400).json({
          success: false,
          message: 'ID de sesión no proporcionado'
        });
        return;
      }
      
      // Obtener información de la sesión de checkout desde Stripe
      const session = await stripe.checkout.sessions.retrieve(sessionId, {
        expand: ['subscription', 'customer']
      });
      
      // Verificar que la sesión fue completada
      if (session.status !== 'complete') {
        res.status(400).json({
          success: false,
          message: 'La sesión de pago no ha sido completada'
        });
        return;
      }
      
      // Extraer metadata de la sesión
      const userId = session.metadata?.userId;
      const planId = session.metadata?.planId;
      const interval = session.metadata?.interval as 'monthly' | 'yearly';
      
      if (!userId || !planId || !interval) {
        res.status(400).json({
          success: false,
          message: 'Información incompleta en la sesión'
        });
        return;
      }
      
      // Obtener detalles de la suscripción
      const subscriptionId = session.subscription as string;
      const subscription = await stripe.subscriptions.retrieve(subscriptionId);
      
      // Calcular fechas
      const currentPeriodStart = new Date((subscription as any).current_period_start * 1000);
      const currentPeriodEnd = new Date((subscription as any).current_period_end * 1000);
      
      // Obtener plan
      const plan = getPlanById(planId);
      if (!plan) {
        res.status(400).json({
          success: false,
          message: 'Plan de suscripción no válido'
        });
        return;
      }
      
      // Actualizar información del usuario
      const userUpdate = {
        subscriptionPlan: planId,
        subscriptionStatus: 'active',
        subscriptionExpiresAt: currentPeriodEnd,
        billingPeriod: interval,
        stripeCustomerId: session.customer as string,
        stripeSubscriptionId: subscriptionId,
        cancelAtPeriodEnd: false,
        lastPaymentDate: new Date()
      };
      
      await User.findByIdAndUpdate(userId, userUpdate);
      
      // Registrar la transacción
      const price = interval === 'monthly' ? plan.price.monthly : plan.price.yearly;
      
      const transactionData = {
        userId: new mongoose.Types.ObjectId(userId),
        planId,
        amount: price,
        currency: 'EUR', // Ajustar según corresponda
        billingPeriod: interval,
        status: 'completed' as const,
        paymentMethod: 'stripe',
        paymentId: session.payment_intent as string,
        periodStart: currentPeriodStart,
        periodEnd: currentPeriodEnd,
        metadata: {
          stripeSessionId: sessionId,
          stripeSubscriptionId: subscriptionId
        }
      };
      
      await SubscriptionTransaction.create(transactionData);
      
      // Devolver respuesta exitosa
      res.json({
        success: true,
        message: 'Suscripción procesada correctamente',
        subscription: {
          plan: planId,
          status: 'active',
          expiresAt: currentPeriodEnd,
          billingPeriod: interval,
          currentPeriodStart,
          currentPeriodEnd
        }
      });
      
    } catch (error: any) {
      logger.error(`Error en processNewSubscription: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al procesar la nueva suscripción'
      });
    }
  };  /**
   * Procesa webhooks de Stripe para eventos relacionados con suscripciones
   */
  public handleStripeWebhook = async (req: Request, res: Response): Promise<void> => {
    const sig = req.headers['stripe-signature'];
    
    if (!sig || typeof sig !== 'string') {
      res.status(400).json({ 
        success: false,
        message: 'Falta la firma de Stripe' 
      });
      return;
    }
    
    try {
      // Verificar la firma del evento usando el secreto del webhook
      const event = stripe.webhooks.constructEvent(
        (req as any).rawBody, 
        sig, 
        process.env.STRIPE_WEBHOOK_SECRET || ''
      );
      
      // Manejar diferentes tipos de eventos
      switch (event.type) {
        case 'invoice.paid':
          await this.handleInvoicePaid(event.data.object);
          break;
          
        case 'invoice.payment_failed':
          await this.handlePaymentFailed(event.data.object);
          break;
          
        case 'customer.subscription.deleted':
          await this.handleSubscriptionCanceled(event.data.object);
          break;
          
        case 'customer.subscription.updated':
          await this.handleSubscriptionUpdated(event.data.object);
          break;
      }
      
      // Responder al webhook
      res.json({ received: true });
    } catch (error: any) {
      logger.error(`Error en el webhook de Stripe: ${error.message}`);
      res.status(400).json({
        success: false,
        message: 'Error al procesar el webhook'
      });
    }
  };
  
  /**
   * Maneja un evento de factura pagada
   */
  private async handleInvoicePaid(invoice: any): Promise<void> {
    try {
      const subscriptionId = invoice.subscription;
      const customerId = invoice.customer;
      
      // Buscar al usuario por su Stripe Customer ID
      const user = await User.findOne({ stripeCustomerId: customerId });
      
      if (!user) {
        logger.error(`No se encontró un usuario para el customerId: ${customerId}`);
        return;
      }
      
      // Registrar la transacción exitosa
      const transaction = {
        userId: user._id,
        planId: user.subscriptionPlan,
        amount: invoice.amount_paid / 100, // Convertir de centavos
        currency: invoice.currency.toUpperCase(),
        billingPeriod: user.billingPeriod,
        status: 'completed' as const,
        paymentMethod: 'stripe',
        paymentId: invoice.payment_intent,
        periodStart: new Date(invoice.period_start * 1000),
        periodEnd: new Date(invoice.period_end * 1000),
        metadata: {
          invoiceId: invoice.id,
          stripeSubscriptionId: subscriptionId
        }
      };
      
      await SubscriptionTransaction.create(transaction);
      
      // Actualizar estado de suscripción del usuario
      await User.findByIdAndUpdate(user._id, {
        subscriptionStatus: 'active',
        subscriptionExpiresAt: new Date(invoice.period_end * 1000),
        lastPaymentDate: new Date(),
        paymentFailCount: 0 // Resetear contador de fallos
      });
      
    } catch (error: any) {
      logger.error(`Error al manejar invoice.paid: ${error.message}`);
    }
  }
  
  /**
   * Maneja un evento de pago fallido
   */
  private async handlePaymentFailed(invoice: any): Promise<void> {
    try {
      const customerId = invoice.customer;
      
      // Buscar al usuario por su Stripe Customer ID
      const user = await User.findOne({ stripeCustomerId: customerId });
      
      if (!user) {
        logger.error(`No se encontró un usuario para el customerId: ${customerId}`);
        return;
      }
      
      // Incrementar contador de fallos de pago
      const failCount = (user.paymentFailCount || 0) + 1;
      
      // Si hay demasiados fallos, cambiar el estado a past_due
      let subscriptionStatus = user.subscriptionStatus;
      if (failCount >= 3) {
        subscriptionStatus = 'past_due';
      }
      
      // Actualizar usuario
      await User.findByIdAndUpdate(user._id, {
        subscriptionStatus,
        paymentFailCount: failCount
      });
      
      // Registrar la transacción fallida
      const transaction = {
        userId: user._id,
        planId: user.subscriptionPlan,
        amount: invoice.amount_due / 100,
        currency: invoice.currency.toUpperCase(),
        billingPeriod: user.billingPeriod,
        status: 'failed' as const,
        paymentMethod: 'stripe',
        paymentId: invoice.payment_intent,
        periodStart: new Date(invoice.period_start * 1000),
        periodEnd: new Date(invoice.period_end * 1000),
        metadata: {
          invoiceId: invoice.id,
          failReason: invoice.last_payment_error?.message || 'Pago rechazado'
        }
      };
      
      await SubscriptionTransaction.create(transaction);
      
    } catch (error: any) {
      logger.error(`Error al manejar invoice.payment_failed: ${error.message}`);
    }
  }
  
  /**
   * Maneja un evento de suscripción cancelada
   */
  private async handleSubscriptionCanceled(subscription: any): Promise<void> {
    try {
      const subscriptionId = subscription.id;
      const customerId = subscription.customer;
      
      // Buscar al usuario por su ID de suscripción
      const user = await User.findOne({ stripeSubscriptionId: subscriptionId });
      
      if (!user) {
        logger.error(`No se encontró un usuario para la suscripción: ${subscriptionId}`);
        return;
      }
      
      // Actualizar el usuario cuando se cancela la suscripción
      await User.findByIdAndUpdate(user._id, {
        subscriptionStatus: 'canceled',
        cancelAtPeriodEnd: true
      });
      
    } catch (error: any) {
      logger.error(`Error al manejar subscription.deleted: ${error.message}`);
    }
  }
  
  /**
   * Maneja un evento de suscripción actualizada
   */
  private async handleSubscriptionUpdated(subscription: any): Promise<void> {
    try {
      const subscriptionId = subscription.id;
      
      // Buscar al usuario por su ID de suscripción
      const user = await User.findOne({ stripeSubscriptionId: subscriptionId });
      
      if (!user) {
        logger.error(`No se encontró un usuario para la suscripción: ${subscriptionId}`);
        return;
      }
      
      // Actualizar información del usuario basada en los cambios de la suscripción
      await User.findByIdAndUpdate(user._id, {
        cancelAtPeriodEnd: subscription.cancel_at_period_end,
        subscriptionExpiresAt: new Date(subscription.current_period_end * 1000)
      });
      
    } catch (error: any) {
      logger.error(`Error al manejar subscription.updated: ${error.message}`);
    }
  }
  // Método auxiliar para determinar si un cambio de plan es un downgrade
  private isPlanDowngrade(currentPlan: string, newPlan: string): boolean {
    const planOrder = ['free', 'basic', 'professional', 'enterprise'];
    
    const currentIndex = planOrder.indexOf(currentPlan);
    const newIndex = planOrder.indexOf(newPlan);
    
    return newIndex < currentIndex;
  }
}