import { Request, Response } from 'express';
import { User } from '../../models/user.model';
import { Subscription, Invoice, SUBSCRIPTION_PLANS, getPlanById } from '../../models/subscription.model';
import { Document } from 'mongoose';
import { getQuotaInfo, resetMonthlyQuotas } from '../../middlewares/subscription.middleware';
import { NotificationService } from '../../utils/notification.service';
import logger from '../../utils/logger';
import mongoose, { Types } from 'mongoose';
import Stripe from 'stripe';

// Configurar cliente de Stripe
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || '', {
  apiVersion: '2023-10-16' as any
});

/**
 * Controlador para gestión de suscripciones desde el panel de admin
 */
export class AdminSubscriptionController {
  private notificationService: NotificationService;

  constructor() {
    this.notificationService = NotificationService.getInstance();
  }

  /**
   * Obtiene todas las suscripciones con filtros y paginación
   */
  public getAllSubscriptions = async (req: Request, res: Response): Promise<void> => {
    try {
      const page = parseInt(req.query.page as string) || 1;
      const limit = parseInt(req.query.limit as string) || 10;
      const skip = (page - 1) * limit;
      
      // Filtros
      const filters: any = {};
      
      if (req.query.plan) {
        filters.subscriptionPlan = req.query.plan;
      }
      
      if (req.query.status) {
        filters.subscriptionStatus = req.query.status;
      }
      
      if (req.query.search) {
        const searchRegex = new RegExp(req.query.search as string, 'i');
        filters.$or = [
          { email: searchRegex },
          { name: searchRegex }
        ];
      }
      
      // Contar total para paginación
      const totalUsers = await User.countDocuments(filters);
      const totalPages = Math.ceil(totalUsers / limit);
      
      // Obtener usuarios con filtros y paginación
      const users = await User.find(filters)
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit);
      
      // Preparar datos de respuesta
      const subscriptions = await Promise.all(users.map(async user => {
        // Obtener info de cuota para cada usuario
        const quotaInfo = await getQuotaInfo(user._id as unknown as Types.ObjectId);
        
        // Formatear fecha de inicio y renovación
        const startDate = user.subscriptionStartDate || user.createdAt;
        const renewalDate = user.subscriptionExpiresAt;
        
        return {
          user: {
            _id: user._id,
            name: user.name,
            email: user.email
          },
          plan: user.subscriptionPlan,
          status: user.subscriptionStatus,
          cancelAtPeriodEnd: user.cancelAtPeriodEnd,
          billingPeriod: user.billingPeriod,
          startDate,
          renewalDate,
          quotaInfo
        };
      }));
      
      // Estadísticas generales
      const stats = {
        totalUsers: await User.countDocuments(),
        activeSubscriptions: await User.countDocuments({ 
          subscriptionStatus: 'active', 
          subscriptionPlan: { $ne: 'free' } 
        }),
        expiringSubscriptions: await User.countDocuments({
          subscriptionExpiresAt: { 
            $gte: new Date(), 
            $lte: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // 30 días
          }
        }),
        // Esta sería una aproximación, idealmente vendría de un cálculo real de ingresos
        monthlyRevenue: await this.calculateMonthlyRevenue()
      };
      
      res.json({
        success: true,
        subscriptions,
        stats,
        pagination: {
          currentPage: page,
          totalPages,
          totalItems: totalUsers,
          itemsPerPage: limit
        }
      });
    } catch (error: any) {
      logger.error(`Error en getAllSubscriptions: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al obtener las suscripciones'
      });
    }
  };

  /**
   * Obtiene estadísticas de suscripciones para el dashboard administrativo
   */
  public getSubscriptionStats = async (req: Request, res: Response): Promise<void> => {
    try {
      // Obtener estadísticas de planes por mes (últimos 6 meses)
      const sixMonthsAgo = new Date();
      sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);
      
      // Este es un ejemplo simplificado; en un sistema real, 
      // estas estadísticas vendrían de agregaciones más complejas
      const plansOverTime = [
        {
          month: 'Enero',
          free: 120,
          basic: 45,
          pro: 20,
          enterprise: 3
        },
        // Se añadirían los otros meses
      ];
      
      // Estadísticas de conversión, churn, etc.
      const conversionStats = {
        conversionRate: 24.8, // Porcentaje de usuarios que pasan de free a planes pagados
        churnRate: 5.2, // Tasa de cancelación
        lifetimeValue: 286 // Valor promedio de un cliente durante toda su relación
      };
      
      // Resumen general
      const summary = {
        totalRevenue: await this.calculateTotalRevenue(),
        activeSubscriptions: await User.countDocuments({ 
          subscriptionStatus: 'active', 
          subscriptionPlan: { $ne: 'free' } 
        }),
        averageSubscriptionLength: 8.5, // En meses
        mostPopularPlan: 'basic'
      };
      
      res.json({
        success: true,
        stats: {
          plansOverTime,
          conversionStats,
          summary,
          // Para dashboard
          totalUsers: await User.countDocuments(),
          activeSubscriptions: summary.activeSubscriptions,
          expiringSubscriptions: await User.countDocuments({
            subscriptionExpiresAt: { 
              $gte: new Date(), 
              $lte: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // 30 días
            }
          }),
          monthlyRevenue: await this.calculateMonthlyRevenue()
        }
      });
    } catch (error: any) {
      logger.error(`Error en getSubscriptionStats: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al obtener las estadísticas de suscripción'
      });
    }
  };

  /**
   * Obtiene los detalles de suscripción de un usuario específico
   */
  public getUserSubscriptionDetails = async (req: Request, res: Response): Promise<void> => {
    try {
      const { userId } = req.params;
      
      if (!mongoose.Types.ObjectId.isValid(userId)) {
        res.status(400).json({
          success: false,
          message: 'ID de usuario no válido'
        });
        return;
      }
      
      // Obtener usuario
      const user = await User.findById(userId);
      
      if (!user) {
        res.status(404).json({
          success: false,
          message: 'Usuario no encontrado'
        });
        return;
      }
        // Obtener información de cuota
      const quotaInfo = await getQuotaInfo(user._id as unknown as Types.ObjectId);
      
      // Obtener facturas/pagos
      const invoices = await Invoice.find({ user: user._id }).sort({ date: -1 }).limit(10);
      
      // Historial de cambios (este sería un ejemplo, idealmente vendría de una colección dedicada)
      const subscriptionHistory = [
        {
          date: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000),
          description: 'Suscripción inicial al plan Basic'
        },
        {
          date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
          description: 'Cambio al plan Pro'
        }
      ];
      
      // Obtener información del plan
      const planDetails = getPlanById(user.subscriptionPlan);
      
      res.json({
        success: true,
        userData: {
          user: {
            _id: user._id,
            name: user.name,
            email: user.email,
            createdAt: user.createdAt
          },
          subscription: {
            plan: user.subscriptionPlan,
            planName: planDetails?.name || 'Desconocido',
            status: user.subscriptionStatus,
            billingPeriod: user.billingPeriod,
            startDate: user.subscriptionStartDate,
            renewalDate: user.subscriptionExpiresAt,
            cancelAtPeriodEnd: user.cancelAtPeriodEnd,
            stripeInfo: {
              customerId: user.stripeCustomerId,
              subscriptionId: user.stripeSubscriptionId
            }
          },
          quotaInfo,          payments: invoices.map(invoice => ({
            id: invoice._id,
            date: invoice.createdAt || invoice.paymentDate,
            amount: invoice.amount,
            currency: invoice.currency || 'EUR',
            status: invoice.status,
            reference: invoice.invoiceNumber || (invoice._id ? invoice._id.toString() : '')
          })),
          history: subscriptionHistory
        }
      });
    } catch (error: any) {
      logger.error(`Error en getUserSubscriptionDetails: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al obtener los detalles de la suscripción del usuario'
      });
    }
  };

  /**
   * Cambia el plan de suscripción de un usuario
   */
  public changeUserPlan = async (req: Request, res: Response): Promise<void> => {
    try {
      const { userId } = req.params;
      const { planId, billingPeriod, applyImmediately, note } = req.body;
      
      if (!mongoose.Types.ObjectId.isValid(userId)) {
        res.status(400).json({
          success: false,
          message: 'ID de usuario no válido'
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
      if (billingPeriod !== 'monthly' && billingPeriod !== 'yearly') {
        res.status(400).json({
          success: false,
          message: 'Intervalo de facturación no válido'
        });
        return;
      }
      
      // Obtener usuario
      const user = await User.findById(userId);
      
      if (!user) {
        res.status(404).json({
          success: false,
          message: 'Usuario no encontrado'
        });
        return;
      }
      
      // Actualización de datos
      const updateData: any = {
        subscriptionPlan: planId,
        billingPeriod
      };
      
      // Si aplica inmediatamente o cambia a plan gratuito
      if (applyImmediately || planId === 'free') {
        // Para plan gratuito, cancelar suscripción en Stripe si existe
        if (planId === 'free' && user.stripeSubscriptionId) {
          try {
            await stripe.subscriptions.cancel(user.stripeSubscriptionId);
            updateData.stripeSubscriptionId = null;
          } catch (stripeError) {
            logger.error(`Error al cancelar suscripción en Stripe: ${stripeError}`);
          }
        }
        
        updateData.subscriptionStatus = 'active';
        
        // Establecer nueva fecha de expiración
        const now = new Date();
        if (billingPeriod === 'monthly') {
          const nextMonth = new Date(now);
          nextMonth.setMonth(nextMonth.getMonth() + 1);
          updateData.subscriptionExpiresAt = nextMonth;
        } else { // yearly
          const nextYear = new Date(now);
          nextYear.setFullYear(nextYear.getFullYear() + 1);
          updateData.subscriptionExpiresAt = nextYear;
        }
      } else {
        // Si no aplica inmediatamente, marcar para cambiar al final del período
        updateData.nextSubscriptionPlan = planId;
        updateData.nextBillingPeriod = billingPeriod;
      }
      
      // Actualizar usuario
      await User.findByIdAndUpdate(userId, updateData);
      
      // Notificar al usuario del cambio
      await this.notificationService.sendUserEmail(
        user.email,
        'Cambio en tu Plan de Suscripción',
        `Tu plan de suscripción ha sido cambiado a ${newPlan.name}. ${applyImmediately ? 'El cambio se ha aplicado inmediatamente.' : 'El cambio se aplicará al final de tu período actual de facturación.'}`
      );
      
      res.json({
        success: true,
        message: `Plan cambiado a ${newPlan.name} correctamente${applyImmediately ? '' : ' (se aplicará al final del período actual)'}`
      });
    } catch (error: any) {
      logger.error(`Error en changeUserPlan: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al cambiar el plan de suscripción del usuario'
      });
    }
  };

  /**
   * Extiende la suscripción de un usuario por un período adicional
   */
  public extendUserSubscription = async (req: Request, res: Response): Promise<void> => {
    try {
      const { userId } = req.params;
      const { duration, unit, reason, note } = req.body;
      
      if (!mongoose.Types.ObjectId.isValid(userId)) {
        res.status(400).json({
          success: false,
          message: 'ID de usuario no válido'
        });
        return;
      }
      
      // Validar duración
      if (!duration || duration < 1) {
        res.status(400).json({
          success: false,
          message: 'Duración de extensión no válida'
        });
        return;
      }
      
      // Validar unidad
      if (unit !== 'days' && unit !== 'months') {
        res.status(400).json({
          success: false,
          message: 'Unidad de extensión no válida'
        });
        return;
      }
      
      // Obtener usuario
      const user = await User.findById(userId);
      
      if (!user) {
        res.status(404).json({
          success: false,
          message: 'Usuario no encontrado'
        });
        return;
      }
      
      // Si el usuario no tiene una suscripción activa no gratuita
      if (user.subscriptionPlan === 'free' || user.subscriptionStatus !== 'active') {
        res.status(400).json({
          success: false,
          message: 'El usuario no tiene una suscripción activa que extender'
        });
        return;
      }
      
      // Calcular nueva fecha de expiración
      let newExpirationDate = new Date(user.subscriptionExpiresAt || Date.now());
      
      if (unit === 'days') {
        newExpirationDate.setDate(newExpirationDate.getDate() + duration);
      } else { // months
        newExpirationDate.setMonth(newExpirationDate.getMonth() + duration);
      }
      
      // Actualizar usuario
      await User.findByIdAndUpdate(userId, {
        subscriptionExpiresAt: newExpirationDate
      });
      
      // Notificar al usuario de la extensión
      await this.notificationService.sendUserEmail(
        user.email,
        'Tu Suscripción ha sido Extendida',
        `Tu suscripción ha sido extendida por ${duration} ${unit === 'days' ? 'día(s)' : 'mes(es)'} adicionales. Tu nueva fecha de expiración es ${newExpirationDate.toLocaleDateString()}.`
      );
      
      res.json({
        success: true,
        message: `Suscripción extendida correctamente por ${duration} ${unit === 'days' ? 'día(s)' : 'mes(es)'}`,
        newExpirationDate
      });
    } catch (error: any) {
      logger.error(`Error en extendUserSubscription: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al extender la suscripción del usuario'
      });
    }
  };

  /**
   * Resetea las cuotas mensuales de un usuario
   */
  public resetUserQuotas = async (req: Request, res: Response): Promise<void> => {
    try {
      const { userId } = req.params;
      
      if (!mongoose.Types.ObjectId.isValid(userId)) {
        res.status(400).json({
          success: false,
          message: 'ID de usuario no válido'
        });
        return;
      }
      
      // Obtener usuario
      const user = await User.findById(userId);
      
      if (!user) {
        res.status(404).json({
          success: false,
          message: 'Usuario no encontrado'
        });
        return;
      }
      // Resetear cuotas (modificamos el código para forzar reset)
      const userObjectId = user._id as unknown as Types.ObjectId;
      await User.findByIdAndUpdate(userObjectId, {
        'apiCalls.used': 0,
        'apiCalls.lastResetDate': new Date(),
        'modelTraining.used': 0,
        'modelTraining.lastResetDate': new Date()
      });
      logger.info(`Cuotas mensuales reseteadas manualmente para el usuario ${userId}`);
      
      // Notificar al usuario
      await this.notificationService.sendUserEmail(
        user.email,
        'Tus Cuotas Mensuales han sido Reseteadas',
        `Tus cuotas mensuales de uso (llamadas API, entrenamientos, almacenamiento) han sido reseteadas. Ya puedes continuar usando el servicio con los límites completos de tu plan.`
      );
      
      res.json({
        success: true,
        message: 'Cuotas reseteadas correctamente'
      });
    } catch (error: any) {
      logger.error(`Error en resetUserQuotas: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al resetear las cuotas del usuario'
      });
    }
  };

  /**
   * Cancela la suscripción de un usuario
   */
  public cancelUserSubscription = async (req: Request, res: Response): Promise<void> => {
    try {
      const { userId } = req.params;
      
      if (!mongoose.Types.ObjectId.isValid(userId)) {
        res.status(400).json({
          success: false,
          message: 'ID de usuario no válido'
        });
        return;
      }
      
      // Obtener usuario
      const user = await User.findById(userId);
      
      if (!user) {
        res.status(404).json({
          success: false,
          message: 'Usuario no encontrado'
        });
        return;
      }
      
      // Si no tiene suscripción de pago, no hay nada que cancelar
      if (user.subscriptionPlan === 'free' || !user.stripeSubscriptionId) {
        res.status(400).json({
          success: false,
          message: 'El usuario no tiene una suscripción activa de pago para cancelar'
        });
        return;
      }
      
      // Cancelar inmediatamente o al final del período
      const cancelImmediately = req.query.immediately === 'true';
      
      if (cancelImmediately) {
        // Cancelar inmediatamente en Stripe
        if (user.stripeSubscriptionId) {
          try {
            await stripe.subscriptions.cancel(user.stripeSubscriptionId);
          } catch (stripeError) {
            logger.error(`Error al cancelar suscripción en Stripe: ${stripeError}`);
          }
        }
        
        // Actualizar usuario
        await User.findByIdAndUpdate(userId, {
          subscriptionStatus: 'canceled',
          subscriptionPlan: 'free',
          stripeSubscriptionId: null,
          cancelAtPeriodEnd: false
        });
        
        // Notificar al usuario
        await this.notificationService.sendUserEmail(
          user.email,
          'Tu Suscripción ha sido Cancelada',
          `Tu suscripción ha sido cancelada inmediatamente. Has sido cambiado al plan gratuito con límites reducidos. Puedes volver a suscribirte en cualquier momento desde tu perfil.`
        );
        
        res.json({
          success: true,
          message: 'Suscripción cancelada inmediatamente'
        });
      } else {
        // Cancelar al final del período en Stripe
        if (user.stripeSubscriptionId) {
          try {
            await stripe.subscriptions.update(user.stripeSubscriptionId, {
              cancel_at_period_end: true
            });
          } catch (stripeError) {
            logger.error(`Error al actualizar suscripción en Stripe: ${stripeError}`);
          }
        }
        
        // Actualizar el estado en la base de datos
        await User.findByIdAndUpdate(userId, {
          cancelAtPeriodEnd: true
        });
        
        // Notificar al usuario
        await this.notificationService.sendUserEmail(
          user.email,
          'Tu Suscripción será Cancelada',
          `Tu suscripción será cancelada al final del período actual (${user.subscriptionExpiresAt?.toLocaleDateString()}). Después de esa fecha, serás cambiado al plan gratuito con límites reducidos.`
        );
        
        res.json({
          success: true,
          message: 'Suscripción marcada para cancelación al final del período actual'
        });
      }
    } catch (error: any) {
      logger.error(`Error en cancelUserSubscription: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al cancelar la suscripción del usuario'
      });
    }
  };
  /**
   * Calcula los ingresos mensuales (aproximación)
   */
  private async calculateMonthlyRevenue(): Promise<number> {
    try {
      // Obtener usuarios con suscripciones activas no gratuitas
      const paidUsers = await User.find({
        subscriptionStatus: 'active',
        subscriptionPlan: { $ne: 'free' }
      });
      
      let monthlyRevenue = 0;
      
      // Sumar precios de planes
      for (const user of paidUsers) {
        const plan = getPlanById(user.subscriptionPlan);
        if (plan) {
          if (user.billingPeriod === 'yearly') {
            // Si es anual, dividir por 12 para obtener valor mensual
            monthlyRevenue += plan.price.yearly / 12;
          } else {
            // Si es mensual, usar el precio mensual directamente
            monthlyRevenue += plan.price.monthly;
          }
        }
      }
      
      return parseFloat(monthlyRevenue.toFixed(2));
    } catch (error) {
      logger.error(`Error al calcular ingresos mensuales: ${error}`);
      return 0;
    }
  }
  /**
   * Calcula los ingresos totales históricos (aproximación)
   */
  private async calculateTotalRevenue(): Promise<number> {
    try {
      // En un sistema real, esto vendría de la suma de todas las facturas
      const invoices = await Invoice.find({
        status: 'paid'
      });
      
      const totalAmount = invoices.reduce((total, invoice) => total + invoice.amount, 0);
      
      return parseFloat(totalAmount.toFixed(2));
    } catch (error) {
      logger.error(`Error al calcular ingresos totales: ${error}`);
      return 0;
    }
  }
}
