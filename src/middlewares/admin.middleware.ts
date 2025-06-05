import { Request, Response, NextFunction } from 'express';
import { User } from '../models/user.model';
import logger from '../utils/logger';

/**
 * Middleware para verificar que el usuario tiene rol de administrador
 */
export const checkAdminRole = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    // El usuario ya debe estar autenticado por el middleware authenticateJWT
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Usuario no autenticado'
      });
      return;
    }
    
    // Verificar si el usuario tiene el rol de administrador
    const user = await User.findById(req.user._id);
    
    if (!user) {
      res.status(401).json({
        success: false,
        message: 'Usuario no encontrado'
      });
      return;
    }
    
    if (user.role !== 'admin') {
      logger.warn(`Usuario sin permisos de administrador intentó acceder: ${user._id}`);
      res.status(403).json({
        success: false,
        message: 'Acceso denegado: Se requieren permisos de administrador'
      });
      return;
    }
    
    // Usuario es admin, continuar
    next();
  } catch (error: any) {
    logger.error(`Error en checkAdminRole: ${error.message}`);
    res.status(500).json({
      success: false,
      message: 'Error al verificar permisos de administrador'
    });
  }
};
