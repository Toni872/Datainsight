import { Request, Response, NextFunction } from 'express';
import { Types } from 'mongoose';
import { User, IUser } from '../models/user.model';
import { getPlanById, getPlanPrice, hasAccessToModelType, SUBSCRIPTION_PLANS } from '../models/subscription.model';
import logger from '../utils/logger';
import { notificationService } from '../utils/notification.service';

// Extender Request con una interfaz para uso interno
interface RequestWithQuota extends Request {
  quotaLimits?: {
    apiCallsPerMonth: number;
    modelTrainingPerMonth: number;
    storageLimit: number;
    advancedModels?: boolean;
    customModels?: boolean;
    dedicatedResources?: boolean;
  };
  user?: any;
}

/**
 * Middleware para verificar los límites de cuota del usuario
 */
export const checkQuotaLimits = async (req: RequestWithQuota, res: Response, next: NextFunction): Promise<void> => {
  try {
    const user = req.user;
    
    // Si no hay usuario o no está autenticado, continuar
    if (!user) {
      next();
      return;
    }
    
    // Verificar si necesitamos resetear las cuotas mensuales
    await resetMonthlyQuotas(user._id);
    
    // Obtener información del plan actual
    const plan = getPlanById(user.subscriptionPlan);
    if (!plan) {
      // Por defecto usamos el plan gratuito
      const freePlan = SUBSCRIPTION_PLANS[0]; // El plan gratuito es el primero
      req.quotaLimits = {
        apiCallsPerMonth: freePlan.features.apiCallsPerMonth,
        modelTrainingPerMonth: freePlan.features.modelTrainingPerMonth,
        storageLimit: freePlan.features.storageLimit
      };
    } else {
      // Definimos los límites según el plan de suscripción
      req.quotaLimits = {
        apiCallsPerMonth: plan.features.apiCallsPerMonth,
        modelTrainingPerMonth: plan.features.modelTrainingPerMonth,
        storageLimit: plan.features.storageLimit,
        advancedModels: plan.features.advancedModels,
        customModels: plan.features.customModels,
        dedicatedResources: plan.features.dedicatedResources
      };
    }
    
    next();
  } catch (error) {
    logger.error(`Error en checkQuotaLimits: ${error}`);
    next();
  }
};

/**
 * Comprueba si una cuota mensual necesita ser reseteada
 */
export const resetMonthlyQuotas = async (userId: Types.ObjectId): Promise<void> => {
  try {
    const user = await User.findById(userId);
    if (!user) return;
    
    const now = new Date();
    let updateData: any = {};
    let needsUpdate = false;
    
    // Verificar cuotas de API
    if (user.apiCalls && shouldResetQuota(user.apiCalls.lastResetDate)) {
      updateData['apiCalls.used'] = 0;
      updateData['apiCalls.lastResetDate'] = now;
      needsUpdate = true;
    }
    
    // Verificar cuotas de entrenamiento de modelos
    if (user.modelTraining && shouldResetQuota(user.modelTraining.lastResetDate)) {
      updateData['modelTraining.used'] = 0;
      updateData['modelTraining.lastResetDate'] = now;
      needsUpdate = true;
    }
    
    // Si necesitamos actualizar, hacerlo
    if (needsUpdate) {
      await User.findByIdAndUpdate(userId, updateData);
    }
    
    logger.info(`Cuotas mensuales verificadas para el usuario ${userId}`);
  } catch (error) {
    logger.error(`Error en resetMonthlyQuotas: ${error}`);
  }
};

/**
 * Obtiene información básica sobre las cuotas y límites de un usuario
 */
export const getBasicQuotaInfo = async (userId: Types.ObjectId): Promise<any> => {
  try {
    const user = await User.findById(userId);
    if (!user) {
      throw new Error('Usuario no encontrado');
    }
    
    // Obtener plan actual
    const plan = getPlanById(user.subscriptionPlan);
    if (!plan) {
      // Si no hay plan, usar el gratuito por defecto
      const freePlan = SUBSCRIPTION_PLANS[0];
      return {
        apiCalls: {
          used: user.apiCalls?.used || 0,
          limit: freePlan.features.apiCallsPerMonth
        },
        modelTraining: {
          used: user.modelTraining?.used || 0,
          limit: freePlan.features.modelTrainingPerMonth
        },
        storage: {
          used: user.storage?.used || 0,
          limit: freePlan.features.storageLimit
        }
      };
    }
    
    // Devolver información basada en el plan
    return {
      apiCalls: {
        used: user.apiCalls?.used || 0,
        limit: plan.features.apiCallsPerMonth
      },
      modelTraining: {
        used: user.modelTraining?.used || 0,
        limit: plan.features.modelTrainingPerMonth
      },
      storage: {
        used: user.storage?.used || 0,
        limit: plan.features.storageLimit
      },
      advancedModels: plan.features.advancedModels,
      customModels: plan.features.customModels,
      dedicatedResources: plan.features.dedicatedResources
    };
  } catch (error) {
    logger.error(`Error en getBasicQuotaInfo: ${error}`);
    return null;
  }
};

/**
 * Incrementa el contador de uso de un recurso
 */
export const incrementResourceUsage = async (
  userId: Types.ObjectId, 
  resourceType: 'apiCalls' | 'modelTraining' | 'storage', 
  amount: number = 1
): Promise<boolean> => {
  try {
    // Primero resetear cuotas si es necesario
    await resetMonthlyQuotas(userId);
    
    // Obtener usuario actual
    const user = await User.findById(userId);
    if (!user) return false;
    
    // Obtener plan y límites
    const plan = getPlanById(user.subscriptionPlan);
    if (!plan) return false;
    
    let currentUsage = 0;
    let limit = 0;
    
    // Determinar el tipo de recurso y sus límites
    switch (resourceType) {
      case 'apiCalls':
        currentUsage = user.apiCalls?.used || 0;
        limit = plan.features.apiCallsPerMonth;
        break;
      case 'modelTraining':
        currentUsage = user.modelTraining?.used || 0;
        limit = plan.features.modelTrainingPerMonth;
        break;
      case 'storage':
        currentUsage = user.storage?.used || 0;
        limit = plan.features.storageLimit;
        break;
    }
    
    // Verificar si excedería el límite
    if (currentUsage + amount > limit) {
      return false; // Límite excedido
    }
    
    // Actualizar el uso
    const updateField = `${resourceType}.used`;
    await User.findByIdAndUpdate(userId, {
      $inc: { [updateField]: amount }
    });
    
    return true;
  } catch (error) {
    logger.error(`Error en incrementResourceUsage: ${error}`);
    return false;
  }
};

/**
 * Determina si una cuota debe ser reseteada (ha pasado un mes desde el último reset)
 */
function shouldResetQuota(lastResetDate: Date): boolean {
  if (!lastResetDate) return true;
  
  const now = new Date();
  const last = new Date(lastResetDate);
  
  // Si estamos en un mes diferente o un año diferente
  return now.getMonth() !== last.getMonth() || now.getFullYear() !== last.getFullYear();
}

// Funciones auxiliares
const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const getNextPlan = (currentPlanId: string): string | null => {
  const planIndex = SUBSCRIPTION_PLANS.findIndex(plan => plan.id === currentPlanId);
  if (planIndex === -1 || planIndex === SUBSCRIPTION_PLANS.length - 1) {
    return null;
  }
  return SUBSCRIPTION_PLANS[planIndex + 1].id;
};

// Definimos interfaces para controlar la información de cuota
interface IQuotaInfo {
  apiCalls: {
    used: number;
    limit: number;
  };
  modelTraining: {
    used: number;
    limit: number;
  };
  storage: {
    used: number;
    limit: number;
  };
  advancedModels?: boolean;
  customModels?: boolean;
  dedicatedResources?: boolean;
}

// Definimos una interfaz extendida para nuestro uso específico
interface IExtendedQuotaInfo extends IQuotaInfo {
  apiCalls: {
    used: number;
    limit: number;
    percentage: number;
  };
  modelTraining: {
    used: number;
    limit: number;
    percentage: number;
  };
  storage: {
    used: number;
    limit: number;
    percentage: number;
    formattedUsed: string;
    formattedLimit: string;
  };
  supportedModelTiers?: string[];
  daysUntilReset?: number;
}

/**
 * Obtiene la información de cuota para un usuario
 * @param userId ID del usuario
 * @returns Información de cuota y uso
 */
export const getQuotaInfo = async (userId: Types.ObjectId): Promise<IExtendedQuotaInfo> => {
  try {
    // Obtener el usuario con su información de suscripción
    const user = await User.findById(userId);
    if (!user) {
      throw new Error('Usuario no encontrado');
    }

    // Obtener el plan de suscripción
    const plan = getPlanById(user.subscriptionPlan);
    if (!plan) {
      throw new Error('Plan de suscripción no válido');
    }

    // Calcular días hasta el próximo reset mensual de cuota
    const apiCallsLastReset = new Date(user.apiCalls.lastResetDate);
    const now = new Date();
    const nextReset = new Date(apiCallsLastReset);
    nextReset.setMonth(nextReset.getMonth() + 1);
    const daysUntilReset = Math.ceil((nextReset.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));

    // Calcular porcentajes de uso
    const apiCallsPercentage = (user.apiCalls.used / plan.features.apiCallsPerMonth) * 100;
    const modelTrainingPercentage = (user.modelTraining.used / plan.features.modelTrainingPerMonth) * 100;
    const storagePercentage = (user.storage.used / plan.features.storageLimit) * 100;

    // Convertir MB a bytes para cálculo preciso (asumiendo que storage.used está en MB)
    const storageBytes = user.storage.used * 1024 * 1024; // Convertir de MB a bytes
    const storageLimitBytes = plan.features.storageLimit * 1024 * 1024; // Convertir de MB a bytes

    return {
      apiCalls: {
        used: user.apiCalls.used,
        limit: plan.features.apiCallsPerMonth,
        percentage: apiCallsPercentage
      },
      modelTraining: {
        used: user.modelTraining.used,
        limit: plan.features.modelTrainingPerMonth,
        percentage: modelTrainingPercentage
      },
      storage: {
        used: storageBytes,
        limit: storageLimitBytes,
        percentage: storagePercentage,
        formattedUsed: formatBytes(storageBytes),
        formattedLimit: formatBytes(storageLimitBytes)
      },
      supportedModelTiers: plan.features.advancedModels ? ['basic', 'advanced'] : ['basic'],
      daysUntilReset: daysUntilReset
    };
  } catch (error: any) {
    logger.error(`Error al obtener información de cuota: ${error.message}`);
    throw error;
  }
};

/**
 * Verifica si un usuario tiene una suscripción activa
 */
export const checkActiveSubscription = (req: RequestWithQuota, res: Response, next: NextFunction) => {
  try {
    const user = req.user as IUser;
    
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Usuario no autenticado'
      });
    }
    
    // Verificar si la suscripción está activa
    if (user.subscriptionStatus !== 'active' && user.subscriptionStatus !== 'trial') {
      return res.status(403).json({
        success: false,
        message: 'Suscripción inactiva o vencida',
        subscriptionStatus: user.subscriptionStatus
      });
    }
    
    // Verificar si la suscripción ha expirado
    if (user.subscriptionExpiresAt && new Date() > new Date(user.subscriptionExpiresAt)) {
      return res.status(403).json({
        success: false,
        message: 'Suscripción expirada',
        expirationDate: user.subscriptionExpiresAt
      });
    }
    
    next();
  } catch (error: any) {
    logger.error(`Error en checkActiveSubscription: ${error.message}`);
    res.status(500).json({
      success: false,
      message: 'Error al verificar el estado de la suscripción'
    });
  }
};

/**
 * Verifica si un usuario tiene disponibilidad de API calls
 */
export const checkApiQuota = async (req: RequestWithQuota, res: Response, next: NextFunction) => {
  try {
    const user = req.user as IUser;
    
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Usuario no autenticado'
      });
    }
    
    // Obtener el plan de suscripción
    const plan = getPlanById(user.subscriptionPlan);
    if (!plan) {
      return res.status(400).json({
        success: false,
        message: 'Plan de suscripción no válido'
      });
    }
    
    // Verificar si ha excedido el límite de llamadas API
    if (user.apiCalls.used >= plan.features.apiCallsPerMonth) {
      return res.status(429).json({
        success: false,
        message: 'Has alcanzado el límite de llamadas API para tu plan',
        limit: plan.features.apiCallsPerMonth,
        used: user.apiCalls.used,
        upgrade: {
          nextPlan: getNextPlan(user.subscriptionPlan),
          upgradeUrl: '/perfil/suscripcion.html'
        }
      });
    }
    
    // Incrementar contador de llamadas API
    await User.findByIdAndUpdate(user._id, {
      $inc: { 'apiCalls.used': 1 }
    });
    
    // Actualizar usuario con datos recientes
    const updatedUser = await User.findById(user._id);
    if (updatedUser) {
      // Verificar si está cerca del límite para enviar notificación
      const apiCallsPercentage = Math.round((updatedUser.apiCalls.used / plan.features.apiCallsPerMonth) * 100);
      if (apiCallsPercentage === 80 || apiCallsPercentage === 90) {
        // Ejecutar notificación en segundo plano para no bloquear la respuesta
        notificationService.notifyLowQuota(updatedUser, 'apiCalls', apiCallsPercentage)
          .catch(err => logger.error(`Error al enviar notificación de cuota: ${err.message}`));
      }
    }
    
    next();
  } catch (error: any) {
    logger.error(`Error en checkApiQuota: ${error.message}`);
    res.status(500).json({
      success: false,
      message: 'Error al verificar cuota de API'
    });
  }
};

/**
 * Verifica si un usuario tiene disponibilidad de entrenamientos de modelo
 */
export const checkModelTrainingQuota = async (req: RequestWithQuota, res: Response, next: NextFunction) => {
  try {
    const user = req.user as IUser;
    
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Usuario no autenticado'
      });
    }
    
    // Obtener el plan de suscripción
    const plan = getPlanById(user.subscriptionPlan);
    if (!plan) {
      return res.status(400).json({
        success: false,
        message: 'Plan de suscripción no válido'
      });
    }
    
    // Verificar si ha excedido el límite de entrenamientos
    if (user.modelTraining.used >= plan.features.modelTrainingPerMonth) {
      return res.status(429).json({
        success: false,
        message: 'Has alcanzado el límite de entrenamientos de modelos para tu plan',
        limit: plan.features.modelTrainingPerMonth,
        used: user.modelTraining.used,
        upgrade: {
          nextPlan: getNextPlan(user.subscriptionPlan),
          upgradeUrl: '/perfil/suscripcion.html'
        }
      });
    }
    
    // Incrementar contador de entrenamientos
    await User.findByIdAndUpdate(user._id, {
      $inc: { 'modelTraining.used': 1 }
    });
    
    // Actualizar usuario con datos recientes
    const updatedUser = await User.findById(user._id);
    if (updatedUser) {
      // Verificar si está cerca del límite para enviar notificación
      const trainingPercentage = Math.round((updatedUser.modelTraining.used / plan.features.modelTrainingPerMonth) * 100);
      if (trainingPercentage === 80 || trainingPercentage === 90) {
        // Ejecutar notificación en segundo plano para no bloquear la respuesta
        notificationService.notifyLowQuota(updatedUser, 'modelTraining', trainingPercentage)
          .catch(err => logger.error(`Error al enviar notificación de cuota: ${err.message}`));
      }
    }
    
    next();
  } catch (error: any) {
    logger.error(`Error en checkModelTrainingQuota: ${error.message}`);
    res.status(500).json({
      success: false,
      message: 'Error al verificar cuota de entrenamiento de modelos'
    });
  }
};

/**
 * Verifica si un usuario tiene espacio de almacenamiento disponible
 * @param sizeInBytes Tamaño del archivo a guardar en bytes
 */
export const checkStorageQuota = (sizeInBytes: number) => {
  return async (req: RequestWithQuota, res: Response, next: NextFunction) => {
    try {
      const user = req.user as IUser;
      
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
      }
      
      // Obtener el plan de suscripción
      const plan = getPlanById(user.subscriptionPlan);
      if (!plan) {
        return res.status(400).json({
          success: false,
          message: 'Plan de suscripción no válido'
        });
      }
      
      // Verificar si excederá el límite de almacenamiento
      const storageLimitBytes = plan.features.storageLimit * 1024 * 1024; // Convertir de MB a bytes
      if (user.storage.used + sizeInBytes > storageLimitBytes) {
        return res.status(429).json({
          success: false,
          message: 'Has alcanzado el límite de almacenamiento para tu plan',
          limit: storageLimitBytes,
          used: user.storage.used,
          requested: sizeInBytes,
          formattedLimit: formatBytes(storageLimitBytes),
          formattedUsed: formatBytes(user.storage.used),
          formattedRequested: formatBytes(sizeInBytes),
          upgrade: {
            nextPlan: getNextPlan(user.subscriptionPlan),
            upgradeUrl: '/perfil/suscripcion.html'
          }
        });
      }
      
      // Incrementar contador de almacenamiento
      await User.findByIdAndUpdate(user._id, {
        $inc: { 'storage.used': sizeInBytes }
      });
      
      // Actualizar usuario con datos recientes
      const updatedUser = await User.findById(user._id);
      if (updatedUser) {
        // Verificar si está cerca del límite para enviar notificación
        const storageLimitBytes = plan.features.storageLimit * 1024 * 1024;
        const storagePercentage = Math.round(((updatedUser.storage.used + sizeInBytes) / storageLimitBytes) * 100);
        if (storagePercentage === 80 || storagePercentage === 90) {
          // Ejecutar notificación en segundo plano para no bloquear la respuesta
          notificationService.notifyLowQuota(updatedUser, 'storage', storagePercentage)
            .catch(err => logger.error(`Error al enviar notificación de cuota: ${err.message}`));
        }
      }
      
      next();
    } catch (error: any) {
      logger.error(`Error en checkStorageQuota: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al verificar cuota de almacenamiento'
      });
    }
  };
};

/**
 * Verifica si un usuario tiene acceso a un tipo específico de modelo
 * @param modelType Tipo de modelo a verificar
 */
export const checkModelAccessPermission = (modelType: string) => {
  return async (req: RequestWithQuota, res: Response, next: NextFunction) => {
    try {
      const user = req.user as IUser;
      
      if (!user) {
        return res.status(401).json({
          success: false,
          message: 'Usuario no autenticado'
        });
      }
      
      // Convertir tipo de modelo a categoría
      const modelAccess = modelType === 'basic' ? 'basic' : 
                         (modelType === 'dbscan' || modelType === 'arima') ? 'advanced' : 'custom';
                         
      if (!hasAccessToModelType(user.subscriptionPlan, modelAccess as 'basic' | 'advanced' | 'custom')) {
        const plan = getPlanById(user.subscriptionPlan);
        let requiredPlanId = '';
        let requiredPlanName = '';
        
        // Determinar plan mínimo requerido
        if (modelAccess === 'advanced') {
          // Para modelos avanzados, al menos se necesita el plan básico
          requiredPlanId = 'basic';
          requiredPlanName = 'Plan Básico';
        } else if (modelAccess === 'custom') {
          // Para modelos personalizados, se necesita al menos el plan profesional
          requiredPlanId = 'professional';
          requiredPlanName = 'Plan Profesional';
        }
        
        return res.status(403).json({
          success: false,
          message: `Tu plan actual no tiene acceso a modelos de tipo "${modelType}"`,
          currentPlan: plan?.name,
          requiredPlan: requiredPlanName,
          upgrade: {
            requiredPlan: requiredPlanId,
            upgradeUrl: '/perfil/suscripcion.html'
          }
        });
      }
      
      next();
    } catch (error: any) {
      logger.error(`Error en checkModelAccessPermission: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Error al verificar permisos de acceso a modelos'
      });
    }
  };
};

// Nota: La inicialización del cron job se ha movido a src/utils/cron.service.ts

// Funciones auxiliares