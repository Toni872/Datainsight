import { User, IUser } from '../models/user.model';
import { getPlanById } from '../models/subscription.model';
import logger from '../utils/logger';
import nodemailer from 'nodemailer';

/**
 * Servicio para enviar notificaciones a los usuarios
 */
export class NotificationService {
  private static instance: NotificationService;
  private transporter?: nodemailer.Transporter;
  private emailEnabled: boolean;

  /**
   * Constructor privado (singleton)
   */
  private constructor() {
    this.emailEnabled = process.env.EMAIL_ENABLED === 'true';

    // Configurar el transporter de email si está habilitado
    if (this.emailEnabled) {
      this.transporter = nodemailer.createTransport({
        host: process.env.EMAIL_HOST,
        port: parseInt(process.env.EMAIL_PORT || '587'),
        secure: process.env.EMAIL_SECURE === 'true',
        auth: {
          user: process.env.EMAIL_USER,
          pass: process.env.EMAIL_PASSWORD
        }
      });
    }
  }

  /**
   * Obtiene la instancia única del servicio
   */
  public static getInstance(): NotificationService {
    if (!NotificationService.instance) {
      NotificationService.instance = new NotificationService();
    }
    return NotificationService.instance;
  }

  /**
   * Envía un correo directamente a un usuario por su email
   */
  public async sendUserEmail(email: string, subject: string, message: string): Promise<boolean> {
    if (!this.emailEnabled) {
      logger.info(`Email deshabilitado. Mensaje para ${email}: ${subject}`);
      return false;
    }
    
    try {
      if (this.transporter) {
        await this.transporter.sendMail({
          from: process.env.EMAIL_FROM || 'Mi Proyecto <noreply@miproyecto.com>',
          to: email,
          subject: subject,
          html: `
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 5px;">
              <h2 style="color: #333;">${subject}</h2>
              <div style="margin: 20px 0; line-height: 1.5;">${message}</div>
              <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #777;">
                &copy; ${new Date().getFullYear()} Mi Proyecto. Todos los derechos reservados.
              </div>
            </div>
          `
        });
      }

      logger.info(`Email enviado a ${email}: ${subject}`);
      return true;
    } catch (error: any) {
      logger.error(`Error al enviar email a ${email}: ${error.message}`);
      return false;
    }
  }

  /**
   * Envía una notificación por email
   */
  private async sendEmailNotification(user: IUser, subject: string, message: string): Promise<void> {
    if (!this.emailEnabled) {
      logger.info(`Email deshabilitado. Mensaje para ${user.email}: ${subject}`);
      return;
    }    try {
      if (this.transporter) {
        await this.transporter.sendMail({
          from: process.env.EMAIL_FROM,
          to: user.email,
          subject: subject,
          html: message
        });
      }

      logger.info(`Email enviado a ${user.email}: ${subject}`);
    } catch (error: any) {
      logger.error(`Error al enviar email a ${user.email}: ${error.message}`);
    }
  }

  /**
   * Notifica al usuario sobre un bajo nivel de cuota
   * @param user Usuario a notificar
   * @param quotaType Tipo de cuota (apiCalls, modelTraining, storage)
   * @param percentage Porcentaje utilizado
   */
  public async notifyLowQuota(user: IUser, quotaType: string, percentage: number): Promise<void> {
    const plan = getPlanById(user.subscriptionPlan);

    if (!plan) {
      logger.error(`Plan no encontrado para el usuario ${user._id}`);
      return;
    }

    let quotaName = '';
    let quotaLimit = 0;
    let quotaUsed = 0;    switch (quotaType) {
      case 'apiCalls':
        quotaName = 'Llamadas API';
        quotaLimit = plan.features.apiCallsPerMonth;
        quotaUsed = user.apiCalls.used;
        break;
      case 'modelTraining':
        quotaName = 'Entrenamientos de modelo';
        quotaLimit = plan.features.modelTrainingPerMonth;
        quotaUsed = user.modelTraining.used;
        break;
      case 'storage':
        quotaName = 'Almacenamiento';
        quotaLimit = plan.features.storageLimit;
        quotaUsed = user.storage.used;
        break;
    }

    const subject = `Alerta de cuota: Has alcanzado el ${percentage}% de tu cuota de ${quotaName}`;
      const message = `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2 style="color: #333;">¡Atención! Tu cuota está llegando al límite</h2>
        <p>Hola ${user.name || 'Usuario'},</p>
        <p>Has utilizado el <strong>${percentage}%</strong> de tu cuota de <strong>${quotaName}</strong> en tu plan ${plan.name}.</p>
        <ul>
          <li>Uso actual: ${quotaUsed}</li>
          <li>Límite del plan: ${quotaLimit}</li>
        </ul>
        
        <div style="margin: 20px 0; background-color: #f8f8f8; padding: 15px; border-radius: 5px;">
          <p>¿Necesitas más recursos? Considera actualizar a un plan superior:</p>
          <a href="${process.env.FRONTEND_URL}/perfil/suscripcion.html" 
             style="background-color: #4CAF50; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; display: inline-block;">
            Ver planes disponibles
          </a>
        </div>
        
        <p>Gracias por usar nuestros servicios.</p>
        <p>El equipo de Mi Proyecto</p>
      </div>
    `;

    await this.sendEmailNotification(user, subject, message);
  }

  /**
   * Notifica al usuario sobre una suscripción próxima a expirar
   */
  public async notifySubscriptionExpiring(user: IUser, daysRemaining: number): Promise<void> {
    const plan = getPlanById(user.subscriptionPlan);

    if (!plan || plan.id === 'free') {
      return; // No notificar para plan gratuito
    }

    const subject = `Tu suscripción al plan ${plan.name} expirará pronto`;
    
    const message = `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">        <h2 style="color: #333;">Tu suscripción está próxima a expirar</h2>
        <p>Hola ${user.name || 'Usuario'},</p>
        
        <p>Tu suscripción al plan <strong>${plan.name}</strong> expirará en <strong>${daysRemaining} días</strong>.</p>
        
        <div style="margin: 20px 0; background-color: #f8f8f8; padding: 15px; border-radius: 5px;">
          <p>Para mantener el acceso a todas las funciones, por favor renueva tu suscripción:</p>
          <a href="${process.env.FRONTEND_URL}/perfil/suscripcion.html" 
             style="background-color: #4CAF50; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; display: inline-block;">
            Renovar suscripción
          </a>
        </div>
        
        <p>Si tienes alguna pregunta sobre tu suscripción, no dudes en contactarnos.</p>
        <p>El equipo de Mi Proyecto</p>
      </div>
    `;

    await this.sendEmailNotification(user, subject, message);
  }

  /**
   * Notifica al administrador sobre eventos importantes
   */
  public async notifyAdmin(subject: string, message: string): Promise<void> {
    const adminEmail = process.env.ADMIN_EMAIL;
    if (!adminEmail || !this.emailEnabled) {
      logger.info(`Notificación de admin (email deshabilitado): ${subject}`);
      return;
    }    try {
      if (this.transporter) {
        await this.transporter.sendMail({
          from: process.env.EMAIL_FROM,
          to: adminEmail,
          subject: `[ADMIN] ${subject}`,
          html: message
        });
      }

      logger.info(`Email de admin enviado: ${subject}`);
    } catch (error: any) {
      logger.error(`Error al enviar email de admin: ${error.message}`);
    }
  }

  /**
   * Comprueba los límites de cuota y envía notificaciones si es necesario
   */
  public async checkQuotaLimits(userId: string): Promise<void> {
    try {
      const user = await User.findById(userId);
      if (!user) {
        return;
      }

      const plan = getPlanById(user.subscriptionPlan);
      if (!plan) {
        return;
      }      // Comprobar límites de API calls
      const apiCallsPercentage = Math.round((user.apiCalls.used / plan.features.apiCallsPerMonth) * 100);
      if (apiCallsPercentage >= 80 && apiCallsPercentage < 90) {
        await this.notifyLowQuota(user, 'apiCalls', 80);
      } else if (apiCallsPercentage >= 90 && apiCallsPercentage < 100) {
        await this.notifyLowQuota(user, 'apiCalls', 90);
      }

      // Comprobar límites de entrenamientos de modelo
      const trainingPercentage = Math.round((user.modelTraining.used / plan.features.modelTrainingPerMonth) * 100);
      if (trainingPercentage >= 80 && trainingPercentage < 90) {
        await this.notifyLowQuota(user, 'modelTraining', 80);
      } else if (trainingPercentage >= 90 && trainingPercentage < 100) {
        await this.notifyLowQuota(user, 'modelTraining', 90);
      }

      // Comprobar límites de almacenamiento
      const storagePercentage = Math.round((user.storage.used / plan.features.storageLimit) * 100);
      if (storagePercentage >= 80 && storagePercentage < 90) {
        await this.notifyLowQuota(user, 'storage', 80);
      } else if (storagePercentage >= 90 && storagePercentage < 100) {
        await this.notifyLowQuota(user, 'storage', 90);
      }

    } catch (error: any) {
      logger.error(`Error al comprobar límites de cuota: ${error.message}`);
    }
  }
}

// Exportar la instancia única
export const notificationService = NotificationService.getInstance();
