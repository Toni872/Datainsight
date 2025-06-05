import cron from 'node-cron';
import { User } from '../models/user.model';
import { resetMonthlyQuotas } from '../middlewares/subscription.middleware';
import logger from './logger';
import mongoose from 'mongoose';

/**
 * Servicio para gestionar tareas programadas
 */
export class CronService {
  /**
   * Inicializa todas las tareas programadas
   */
  static initializeAllCronJobs(): void {
    this.initializeQuotaResetCron();
    this.initializeSubscriptionExpirationCheck();
    
    logger.info('Todas las tareas programadas han sido inicializadas');
  }

  /**
   * Inicializa el trabajo cron para resetear las cuotas mensuales automáticamente
   */
  static initializeQuotaResetCron(): void {
    // Ejecutar a las 00:00 del primer día de cada mes
    cron.schedule('0 0 1 * *', async () => {
      logger.info('Ejecutando reseteo mensual de cuotas programado');
      try {
        // Obtener todos los usuarios con suscripciones activas
        const users = await User.find({ 
          subscriptionStatus: { $in: ['active', 'trial'] }
        });
        
        let successCount = 0;
        let errorCount = 0;          for (const user of users) {
          try {
            if (user && user._id) {
              // Asegurarse de que es un ObjectId válido
              const userId = mongoose.Types.ObjectId.isValid(user._id.toString()) 
                ? new mongoose.Types.ObjectId(user._id.toString())
                : null;
                
              if (userId) {
                await resetMonthlyQuotas(userId);
                successCount++;
              } else {
                throw new Error('ID de usuario inválido');
              }
            }
          } catch (error: any) {
            errorCount++;
            logger.error(`Error al resetear cuotas del usuario ${user?._id || 'desconocido'}: ${error.message}`);
          }
        }
        
        logger.info(`Reseteo mensual completado: ${successCount} exitosos, ${errorCount} fallidos`);
      } catch (error: any) {
        logger.error(`Error al ejecutar reseteo mensual de cuotas: ${error.message}`);
      }
    }, {
      timezone: 'Europe/Madrid' // Ajusta a la zona horaria apropiada
    });
    
    logger.info('Tarea de reseteo mensual de cuotas inicializada');
  }

  /**
   * Inicializa el trabajo cron para verificar suscripciones a punto de expirar
   */
  static initializeSubscriptionExpirationCheck(): void {
    // Ejecutar a las 09:00 todos los días
    cron.schedule('0 9 * * *', async () => {
      logger.info('Verificando suscripciones próximas a expirar');
      try {
        const currentDate = new Date();
        const threeDaysFromNow = new Date();
        threeDaysFromNow.setDate(currentDate.getDate() + 3);
        
        // Buscar usuarios cuya suscripción expira en los próximos 3 días
        const usersWithExpiringSubscriptions = await User.find({
          subscriptionStatus: 'active',
          subscriptionPlan: { $ne: 'free' },
          subscriptionExpiresAt: {
            $gte: currentDate,
            $lte: threeDaysFromNow
          },
          cancelAtPeriodEnd: true // Solo los que no se renovarán automáticamente
        });
        
        logger.info(`Se encontraron ${usersWithExpiringSubscriptions.length} suscripciones próximas a expirar`);
        
        // Aquí se podrían enviar notificaciones a los usuarios
        // Este código se implementaría según los requisitos
        
      } catch (error: any) {
        logger.error(`Error al verificar suscripciones próximas a expirar: ${error.message}`);
      }
    }, {
      timezone: 'Europe/Madrid'
    });
    
    logger.info('Tarea de verificación de expiración de suscripciones inicializada');
  }
}

export default CronService;
