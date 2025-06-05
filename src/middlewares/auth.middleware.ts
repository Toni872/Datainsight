import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { User, IUser } from '../models/user.model';
import logger from '../utils/logger';

// Extender la interfaz Request para incluir el usuario
declare global {
  namespace Express {
    interface Request {
      user?: IUser;
    }
  }
}

/**
 * Middleware para autenticar el token JWT
 */
export const authenticateJWT = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader) {
      return res.status(401).json({
        success: false,
        message: 'No se proporcionó token de autenticación'
      });
    }
    
    // El formato de authorization header debería ser "Bearer [token]"
    const token = authHeader.split(' ')[1];
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'Formato de token inválido'
      });
    }
    
    // Verificar el token
    const secret = process.env.JWT_SECRET || 'default_secret_key_change_in_production';
    const decoded = jwt.verify(token, secret) as { id: string };
    
    // Buscar el usuario
    const user = await User.findById(decoded.id);
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Usuario no encontrado'
      });
    }
    
    // Si la cuenta está desactivada
    if (!user.isActive) {
      return res.status(403).json({
        success: false,
        message: 'Cuenta desactivada'
      });
    }
    
    // Añadir el usuario a la solicitud
    req.user = user;
    
    next();
  } catch (error: any) {
    logger.error(`Error en authenticateJWT: ${error.message}`);
    
    if (error.name === 'TokenExpiredError') {
      return res.status(401).json({
        success: false,
        message: 'Token expirado',
        expired: true
      });
    }
    
    return res.status(401).json({
      success: false,
      message: 'Token inválido'
    });
  }
};

/**
 * Middleware para verificar roles de usuario
 * @param roles Roles permitidos
 */
export const authorizeRoles = (roles: string[]) => {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'Usuario no autenticado'
      });
    }
    
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({
        success: false,
        message: 'No tienes permisos para acceder a este recurso'
      });
    }
    
    next();
  };
};

/**
 * Middleware para verificar la autenticación por API key
 */
export const authenticateApiKey = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const apiKey = req.headers['x-api-key'];
    
    if (!apiKey) {
      return res.status(401).json({
        success: false,
        message: 'No se proporcionó API key'
      });
    }
    
    // Buscar el usuario por API key
    const user = await User.findOne({ apiKey });
    
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'API key inválida'
      });
    }
    
    // Si la cuenta está desactivada
    if (!user.isActive) {
      return res.status(403).json({
        success: false,
        message: 'Cuenta desactivada'
      });
    }
    
    // Añadir el usuario a la solicitud
    req.user = user;
    
    // Actualizar la fecha del último inicio de sesión
    await User.findByIdAndUpdate(user._id, { lastLogin: new Date() });
    
    next();
  } catch (error: any) {
    logger.error(`Error en authenticateApiKey: ${error.message}`);
    
    return res.status(500).json({
      success: false,
      message: 'Error en la autenticación con API key'
    });
  }
};