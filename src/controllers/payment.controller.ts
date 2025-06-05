import { Request, Response } from 'express';
import { User } from '../models/user.model';
import { getPlanById, SubscriptionTransaction } from '../models/subscription.model';
import logger from '../utils/logger';
import Stripe from 'stripe';

// Configurar cliente de Stripe
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || '', {
  apiVersion: '2023-10-16' as any
});

// Webhook secret para verificar eventos de Stripe
const endpointSecret = process.env.STRIPE_WEBHOOK_SECRET || '';

/**
 * Controlador para gestión de pagos
 */
export class PaymentController {

  /**
   * Crea una sesión de checkout de Stripe
   */
  public createCheckoutSession = async (req: Request, res: Response): Promise<void> => {
    try {
      const user = req.user;
      const { planId, priceId, interval } = req.body;
      
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
        return;
      }
      
      // Validar el plan seleccionado
      const plan = getPlanById(planId);
      if (!plan) {
        res.status(400).json({
          success: false,
          message: 'Plan de suscripción no válido'
        });
        return;
      }
        // Validar el ID de precio de Stripe
      const stripePriceId = interval === 'yearly' ? plan.stripeYearlyPriceId : plan.stripeMonthlyPriceId;
      if (!stripePriceId) {
        res.status(400).json({
          success: false,
          message: 'Precio de suscripción no válido'
        });
        return;
      }
      
      // Crear cliente en Stripe si no existe
      let customerId = user.stripeCustomerId;
      if (!customerId) {
        const customer = await stripe.customers.create({
          email: user.email,
          name: `${user.firstName} ${user.lastName}`,
          metadata: {
            userId: user._id.toString()
          }
        });
        
        customerId = customer.id;
        
        // Actualizar el ID de cliente en la base de datos
        await User.findByIdAndUpdate(user._id, {
          stripeCustomerId: customerId
        });
      }
      
      // Crear sesión de checkout
      const session = await stripe.checkout.sessions.create({
        customer: customerId,
        payment_method_types: ['card'],
        line_items: [
          {
            price: planPrice.stripePriceId,
            quantity: 1
          }
        ],
        mode: 'subscription',
        success_url: `${process.env.FRONTEND_URL}/perfil/suscripcion.html?success=true&session_id={CHECKOUT_SESSION_ID}`,
        cancel_url: `${process.env.FRONTEND_URL}/perfil/suscripcion.html?canceled=true`,
        metadata: {
          userId: user._id.toString(),
          planId: planId,
          interval: interval
        }
      });
      
      res.json({
        success: true,
        sessionId: session.id,
        url: session.url
      });
    } catch (error: any) {
      logger.error(`Error en createCheckoutSession: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al crear la sesión de checkout'
      });
    }
  };

  /**
   * Maneja los eventos de webhook de Stripe
   */
  public handleStripeWebhook = async (req: Request, res: Response): Promise<void> => {
    const signature = req.headers['stripe-signature'];
    
    if (!signature) {
      res.status(400).json({ success: false, message: 'No Stripe signature found' });
      return;
    }
    
    let event;
    
    try {
      event = stripe.webhooks.constructEvent(
        req.body,
        signature,
        endpointSecret
      );
    } catch (err: any) {
      logger.error(`Webhook signature verification failed: ${err.message}`);
      res.status(400).json({ success: false, message: err.message });
      return;
    }
    
    try {
      // Manejar diferentes eventos de Stripe
      switch (event.type) {
        case 'checkout.session.completed':
          await this.handleCheckoutSessionCompleted(event.data.object);
          break;
          
        case 'customer.subscription.created':
        case 'customer.subscription.updated':
          await this.handleSubscriptionUpdated(event.data.object);
          break;
          
        case 'customer.subscription.deleted':
          await this.handleSubscriptionDeleted(event.data.object);
          break;
          
        case 'invoice.paid':
          await this.handleInvoicePaid(event.data.object);
          break;
          
        case 'invoice.payment_failed':
          await this.handleInvoicePaymentFailed(event.data.object);
          break;
      }
      
      res.json({ success: true, received: true });
    } catch (error: any) {
      logger.error(`Error en handleStripeWebhook: ${error.message}`);
      res.status(500).json({ success: false, message: error.message });
    }
  };

  /**
   * Maneja el éxito de un pago
   */
  public handlePaymentSuccess = async (req: Request, res: Response): Promise<void> => {
    try {
      const { session_id } = req.query;
      
      if (!session_id) {
        res.status(400).json({
          success: false,
          message: 'ID de sesión no proporcionado'
        });
        return;
      }
      
      // Verificar la sesión en Stripe
      const session = await stripe.checkout.sessions.retrieve(session_id as string);
      
      if (!session) {
        res.status(404).json({
          success: false,
          message: 'Sesión no encontrada'
        });
        return;
      }
      
      // Redirigir a la página de éxito
      res.redirect('/perfil/suscripcion.html?success=true');
    } catch (error: any) {
      logger.error(`Error en handlePaymentSuccess: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al procesar el éxito del pago'
      });
    }
  };

  /**
   * Maneja la cancelación de un pago
   */
  public handlePaymentCancel = (req: Request, res: Response): void => {
    // Redirigir a la página de cancelación
    res.redirect('/perfil/suscripcion.html?canceled=true');
  };

  /**
   * Obtiene los métodos de pago del usuario
   */
  public getPaymentMethods = async (req: Request, res: Response): Promise<void> => {
    try {
      const user = req.user;
      
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
        return;
      }
      
      if (!user.stripeCustomerId) {
        res.json({
          success: true,
          paymentMethods: []
        });
        return;
      }
      
      // Obtener métodos de pago del cliente en Stripe
      const paymentMethods = await stripe.paymentMethods.list({
        customer: user.stripeCustomerId,
        type: 'card'
      });
      
      // Transformar los datos para el cliente
      const formattedPaymentMethods = paymentMethods.data.map(method => ({
        id: method.id,
        type: method.type,
        card: {
          brand: method.card?.brand,
          last4: method.card?.last4,
          expMonth: method.card?.exp_month,
          expYear: method.card?.exp_year
        },
        isDefault: method.id === user.defaultPaymentMethodId
      }));
      
      res.json({
        success: true,
        paymentMethods: formattedPaymentMethods
      });
    } catch (error: any) {
      logger.error(`Error en getPaymentMethods: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al obtener los métodos de pago'
      });
    }
  };

  /**
   * Añade un método de pago
   */
  public addPaymentMethod = async (req: Request, res: Response): Promise<void> => {
    try {
      const user = req.user;
      const { paymentMethodId } = req.body;
      
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
        return;
      }
      
      if (!paymentMethodId) {
        res.status(400).json({
          success: false,
          message: 'ID del método de pago no proporcionado'
        });
        return;
      }
      
      // Si el usuario no tiene ID de cliente en Stripe, crear uno
      if (!user.stripeCustomerId) {
        const customer = await stripe.customers.create({
          email: user.email,
          name: `${user.firstName} ${user.lastName}`,
          metadata: {
            userId: user._id.toString()
          }
        });
        
        await User.findByIdAndUpdate(user._id, {
          stripeCustomerId: customer.id
        });
        
        user.stripeCustomerId = customer.id;
      }
      
      // Asociar el método de pago al cliente
      await stripe.paymentMethods.attach(paymentMethodId, {
        customer: user.stripeCustomerId
      });
      
      // Si es el primer método de pago, establecerlo como predeterminado
      const paymentMethods = await stripe.paymentMethods.list({
        customer: user.stripeCustomerId,
        type: 'card'
      });
      
      if (paymentMethods.data.length === 1) {
        await User.findByIdAndUpdate(user._id, {
          defaultPaymentMethodId: paymentMethodId
        });
        
        // Actualizar el método de pago predeterminado en Stripe
        await stripe.customers.update(user.stripeCustomerId, {
          invoice_settings: {
            default_payment_method: paymentMethodId
          }
        });
      }
      
      res.json({
        success: true,
        message: 'Método de pago añadido correctamente'
      });
    } catch (error: any) {
      logger.error(`Error en addPaymentMethod: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al añadir el método de pago'
      });
    }
  };

  /**
   * Elimina un método de pago
   */
  public deletePaymentMethod = async (req: Request, res: Response): Promise<void> => {
    try {
      const user = req.user;
      const { id } = req.params;
      
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
        return;
      }
      
      if (!user.stripeCustomerId) {
        res.status(400).json({
          success: false,
          message: 'No tienes métodos de pago registrados'
        });
        return;
      }
      
      // Verificar que el método de pago pertenezca al usuario
      const paymentMethods = await stripe.paymentMethods.list({
        customer: user.stripeCustomerId,
        type: 'card'
      });
      
      const paymentMethod = paymentMethods.data.find(method => method.id === id);
      
      if (!paymentMethod) {
        res.status(404).json({
          success: false,
          message: 'Método de pago no encontrado'
        });
        return;
      }
      
      // Si es el método de pago predeterminado y hay otros, establecer otro como predeterminado
      if (id === user.defaultPaymentMethodId && paymentMethods.data.length > 1) {
        const newDefaultMethod = paymentMethods.data.find(method => method.id !== id);
        
        if (newDefaultMethod) {
          await User.findByIdAndUpdate(user._id, {
            defaultPaymentMethodId: newDefaultMethod.id
          });
          
          // Actualizar el método de pago predeterminado en Stripe
          await stripe.customers.update(user.stripeCustomerId, {
            invoice_settings: {
              default_payment_method: newDefaultMethod.id
            }
          });
        }
      }
      
      // Desvincular el método de pago del cliente
      await stripe.paymentMethods.detach(id);
      
      res.json({
        success: true,
        message: 'Método de pago eliminado correctamente'
      });
    } catch (error: any) {
      logger.error(`Error en deletePaymentMethod: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al eliminar el método de pago'
      });
    }
  };

  /**
   * Maneja el evento de checkout completado
   */
  private handleCheckoutSessionCompleted = async (session: any): Promise<void> => {
    try {
      const userId = session.metadata?.userId;
      const planId = session.metadata?.planId;
      const interval = session.metadata?.interval;
      
      if (!userId || !planId) {
        logger.error('Checkout session completed sin userId o planId en los metadatos');
        return;
      }
      
      // Actualizar el usuario con la nueva suscripción
      await User.findByIdAndUpdate(userId, {
        subscriptionPlan: planId,
        billingPeriod: interval || 'monthly',
        subscriptionStatus: 'active',
        cancelAtPeriodEnd: false
      });
      
      logger.info(`Suscripción actualizada para el usuario ${userId} al plan ${planId}`);
    } catch (error: any) {
      logger.error(`Error en handleCheckoutSessionCompleted: ${error.message}`);
    }
  };

  /**
   * Maneja el evento de actualización de suscripción
   */
  private handleSubscriptionUpdated = async (subscription: any): Promise<void> => {
    try {
      const customerId = subscription.customer;
      
      // Buscar el usuario por ID de cliente de Stripe
      const user = await User.findOne({ stripeCustomerId: customerId });
      
      if (!user) {
        logger.error(`Usuario no encontrado para el cliente de Stripe ${customerId}`);
        return;
      }
      
      // Actualizar la información de suscripción
      await User.findByIdAndUpdate(user._id, {
        stripeSubscriptionId: subscription.id,
        subscriptionStatus: subscription.status,
        subscriptionExpiresAt: new Date(subscription.current_period_end * 1000),
        cancelAtPeriodEnd: subscription.cancel_at_period_end
      });
      
      logger.info(`Información de suscripción actualizada para el usuario ${user._id}`);
    } catch (error: any) {
      logger.error(`Error en handleSubscriptionUpdated: ${error.message}`);
    }
  };

  /**
   * Maneja el evento de eliminación de suscripción
   */
  private handleSubscriptionDeleted = async (subscription: any): Promise<void> => {
    try {
      const customerId = subscription.customer;
      
      // Buscar el usuario por ID de cliente de Stripe
      const user = await User.findOne({ stripeCustomerId: customerId });
      
      if (!user) {
        logger.error(`Usuario no encontrado para el cliente de Stripe ${customerId}`);
        return;
      }
      
      // Actualizar la información de suscripción
      await User.findByIdAndUpdate(user._id, {
        subscriptionPlan: 'free',
        subscriptionStatus: 'canceled',
        stripeSubscriptionId: null,
        cancelAtPeriodEnd: false
      });
      
      logger.info(`Suscripción cancelada para el usuario ${user._id}`);
    } catch (error: any) {
      logger.error(`Error en handleSubscriptionDeleted: ${error.message}`);
    }
  };

  /**
   * Maneja el evento de factura pagada
   */
  private handleInvoicePaid = async (invoice: any): Promise<void> => {
    try {
      const customerId = invoice.customer;
      
      // Buscar el usuario por ID de cliente de Stripe
      const user = await User.findOne({ stripeCustomerId: customerId });
      
      if (!user) {
        logger.error(`Usuario no encontrado para el cliente de Stripe ${customerId}`);
        return;
      }
      
      // Crear registro de factura en la base de datos
      await Invoice.create({
        invoiceId: invoice.id,
        amount: invoice.amount_paid / 100, // Convertir de centavos a la moneda base
        currency: invoice.currency.toUpperCase(),
        status: invoice.status,
        date: new Date(invoice.created * 1000),
        period: {
          start: new Date(invoice.period_start * 1000),
          end: new Date(invoice.period_end * 1000)
        },
        items: invoice.lines.data.map((line: any) => ({
          description: line.description || 'Suscripción',
          amount: line.amount / 100
        })),
        pdf: invoice.invoice_pdf,
        user: user._id,
        paymentMethod: invoice.payment_method_details?.type
      });
      
      logger.info(`Factura registrada para el usuario ${user._id}`);
    } catch (error: any) {
      logger.error(`Error en handleInvoicePaid: ${error.message}`);
    }
  };

  /**
   * Maneja el evento de fallo en el pago de factura
   */
  private handleInvoicePaymentFailed = async (invoice: any): Promise<void> => {
    try {
      const customerId = invoice.customer;
      
      // Buscar el usuario por ID de cliente de Stripe
      const user = await User.findOne({ stripeCustomerId: customerId });
      
      if (!user) {
        logger.error(`Usuario no encontrado para el cliente de Stripe ${customerId}`);
        return;
      }
      
      // Actualizar el estado de la suscripción
      await User.findByIdAndUpdate(user._id, {
        subscriptionStatus: 'past_due'
      });
      
      logger.info(`Estado de suscripción actualizado a past_due para el usuario ${user._id}`);
      
      // Aquí podrías enviar una notificación al usuario sobre el fallo en el pago
    } catch (error: any) {
      logger.error(`Error en handleInvoicePaymentFailed: ${error.message}`);
    }
  };
}