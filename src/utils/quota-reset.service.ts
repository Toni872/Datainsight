import { User } from '../models/user.model';
import { resetMonthlyQuotas } from '../middlewares/subscription.middleware';
import logger from '../utils/logger';

/**
 * Clase que gestiona la programación y ejecución del reseteo de cuotas
 */
export class QuotaResetService {
  private static instance: QuotaResetService;
  private timer: NodeJS.Timeout | null = null;
  private isRunning: boolean = false;

  /**
   * Constructor privado (singleton)
   */
  private constructor() {
    // Constructor privado para implementar el patrón singleton
  }

  /**
   * Obtiene la instancia única del servicio
   */
  public static getInstance(): QuotaResetService {
    if (!QuotaResetService.instance) {
      QuotaResetService.instance = new QuotaResetService();
    }
    return QuotaResetService.instance;
  }

  /**
   * Inicia el servicio de reseteo de cuotas
   */
  public start(): void {
    if (this.isRunning) {
      logger.info('El servicio de reseteo de cuotas ya está en ejecución');
      return;
    }

    // Programar el reseteo diario
    this.timer = setInterval(async () => {
      await this.checkAndResetQuotas();
    }, 24 * 60 * 60 * 1000); // Cada 24 horas

    // Ejecutar al inicio para comprobar si hay cuotas que resetear
    this.checkAndResetQuotas();
    
    this.isRunning = true;
    logger.info('Servicio de reseteo de cuotas iniciado');
  }

  /**
   * Detiene el servicio de reseteo de cuotas
   */
  public stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.isRunning = false;
    logger.info('Servicio de reseteo de cuotas detenido');
  }

  /**
   * Verifica y resetea las cuotas de todos los usuarios
   */
  private async checkAndResetQuotas(): Promise<void> {
    try {
      logger.info('Iniciando verificación de cuotas mensuales...');
      
      // Obtener todos los usuarios
      const users = await User.find({});
      let resetCount = 0;      // Para cada usuario, comprobar si debe resetearse su cuota
      for (const user of users) {
        try {
          // Convertir user._id a ObjectId antes de usarlo
          const userId = user._id;
          if (userId) {
            await resetMonthlyQuotas(userId as any);
            resetCount++;
          }
        } catch (error: any) {
          logger.error(`Error al resetear cuota del usuario ${user._id}: ${error.message}`);
        }
      }
      
      logger.info(`Verificación de cuotas completada. Se procesaron ${users.length} usuarios y se resetearon ${resetCount} cuotas.`);
    } catch (error: any) {
      logger.error(`Error en la verificación de cuotas: ${error.message}`);
    }
  }

  /**
   * Fuerza una comprobación inmediata de todas las cuotas
   */
  public async forceCheck(): Promise<void> {
    await this.checkAndResetQuotas();
  }
}

// Exportar la instancia única
export const quotaResetService = QuotaResetService.getInstance();
