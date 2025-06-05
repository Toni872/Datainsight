import { Request, Response } from 'express';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { User } from '../models/user.model';
import logger from '../utils/logger';

export class AuthController {
  /**
   * Registra un nuevo usuario
   */
  public register = async (req: Request, res: Response): Promise<void> => {
    try {
      const { name, email, password } = req.body;

      // Validar datos obligatorios
      if (!email || !password) {
        res.status(400).json({
          success: false,
          message: 'Email y contraseña son obligatorios'
        });
        return;
      }

      // Comprobar si el email ya existe
      const existingUser = await User.findOne({ email });
      if (existingUser) {
        res.status(400).json({
          success: false,
          message: 'El email ya está registrado'
        });
        return;
      }

      // Crear nuevo usuario
      const user = new User({
        name: name || email.split('@')[0], // Usar parte del email como nombre si no se proporciona
        email,
        password, // Se encriptará automáticamente por el middleware pre-save
        role: 'user',
        subscriptionPlan: 'free',
        subscriptionStatus: 'active',
        billingPeriod: 'monthly',
        cancelAtPeriodEnd: false,
        apiCalls: { used: 0, limit: 100 },
        modelTraining: { used: 0, limit: 5 },
        storage: { used: 0, limit: 100 }
      });

      await user.save();

      // Generar token JWT
      const token = jwt.sign(
        { id: user._id, email: user.email, role: user.role },
        process.env.JWT_SECRET || 'datainsight-secret-key',
        { expiresIn: '1d' }
      );

      res.status(201).json({
        success: true,
        message: 'Usuario registrado correctamente',
        token,
        user: {
          id: user._id,
          email: user.email,
          name: user.name,
          role: user.role,
          subscription: {
            plan: user.subscriptionPlan,
            status: user.subscriptionStatus
          }
        }
      });
    } catch (error) {
      logger.error('Error al registrar usuario:', error);
      res.status(500).json({
        success: false,
        message: 'Error al registrar usuario',
        error: (error as Error).message
      });
    }
  };

  /**
   * Inicia sesión con un usuario existente
   */
  public login = async (req: Request, res: Response): Promise<void> => {
    try {
      const { email, password } = req.body;

      // Validar datos obligatorios
      if (!email || !password) {
        res.status(400).json({
          success: false,
          message: 'Email y contraseña son obligatorios'
        });
        return;
      }

      // Buscar usuario por email
      const user = await User.findOne({ email });
      if (!user) {
        res.status(401).json({
          success: false,
          message: 'Credenciales inválidas'
        });
        return;
      }

      // Verificar contraseña
      const isPasswordValid = await user.comparePassword(password);
      if (!isPasswordValid) {
        res.status(401).json({
          success: false,
          message: 'Credenciales inválidas'
        });
        return;
      }

      // Generar token JWT
      const token = jwt.sign(
        { id: user._id, email: user.email, role: user.role },
        process.env.JWT_SECRET || 'datainsight-secret-key',
        { expiresIn: '1d' }
      );

      res.status(200).json({
        success: true,
        message: 'Inicio de sesión exitoso',
        token,
        user: {
          id: user._id,
          email: user.email,
          name: user.name,
          role: user.role,
          subscription: {
            plan: user.subscriptionPlan,
            status: user.subscriptionStatus
          }
        }
      });
    } catch (error) {
      logger.error('Error al iniciar sesión:', error);
      res.status(500).json({
        success: false,
        message: 'Error al iniciar sesión',
        error: (error as Error).message
      });
    }
  };

  /**
   * Obtiene el perfil del usuario actual
   */
  public getProfile = async (req: Request, res: Response): Promise<void> => {
    try {
      const userId = (req as any).user.id;
      
      const user = await User.findById(userId).select('-password');
      if (!user) {
        res.status(404).json({
          success: false,
          message: 'Usuario no encontrado'
        });
        return;
      }

      res.status(200).json({
        success: true,
        user: {
          id: user._id,
          email: user.email,
          name: user.name,
          role: user.role,
          subscription: {
            plan: user.subscriptionPlan,
            status: user.subscriptionStatus,
            expiresAt: user.subscriptionExpiresAt,
            billingPeriod: user.billingPeriod
          },
          quotas: {
            apiCalls: user.apiCalls,
            modelTraining: user.modelTraining,
            storage: user.storage
          }
        }
      });
    } catch (error) {
      logger.error('Error al obtener perfil:', error);
      res.status(500).json({
        success: false,
        message: 'Error al obtener perfil',
        error: (error as Error).message
      });
    }
  };

  /**
   * Actualiza la información del perfil del usuario
   */
  public updateProfile = async (req: Request, res: Response): Promise<void> => {
    try {
      const userId = (req as any).user.id;
      const { name, email } = req.body;
      
      // Verificar si el email ya está en uso por otro usuario
      if (email) {
        const existingUser = await User.findOne({ email, _id: { $ne: userId } });
        if (existingUser) {
          res.status(400).json({
            success: false,
            message: 'El email ya está en uso por otro usuario'
          });
          return;
        }
      }
      
      // Actualizar usuario
      const updatedUser = await User.findByIdAndUpdate(
        userId,
        { $set: { name, email, updatedAt: new Date() } },
        { new: true }
      ).select('-password');
      
      if (!updatedUser) {
        res.status(404).json({
          success: false,
          message: 'Usuario no encontrado'
        });
        return;
      }
      
      res.status(200).json({
        success: true,
        message: 'Perfil actualizado correctamente',
        user: updatedUser
      });
    } catch (error) {
      logger.error('Error al actualizar perfil:', error);
      res.status(500).json({
        success: false,
        message: 'Error al actualizar perfil',
        error: (error as Error).message
      });
    }
  };

  /**
   * Cambia la contraseña del usuario
   */
  public changePassword = async (req: Request, res: Response): Promise<void> => {
    try {
      const userId = (req as any).user.id;
      const { currentPassword, newPassword } = req.body;
      
      if (!currentPassword || !newPassword) {
        res.status(400).json({
          success: false,
          message: 'Se requiere la contraseña actual y la nueva'
        });
        return;
      }
      
      // Obtener usuario con contraseña
      const user = await User.findById(userId);
      if (!user) {
        res.status(404).json({
          success: false,
          message: 'Usuario no encontrado'
        });
        return;
      }
      
      // Verificar contraseña actual
      const isCurrentPasswordValid = await user.comparePassword(currentPassword);
      if (!isCurrentPasswordValid) {
        res.status(401).json({
          success: false,
          message: 'La contraseña actual es incorrecta'
        });
        return;
      }
      
      // Actualizar contraseña
      user.password = newPassword;
      await user.save();
      
      res.status(200).json({
        success: true,
        message: 'Contraseña actualizada correctamente'
      });
    } catch (error) {
      logger.error('Error al cambiar contraseña:', error);
      res.status(500).json({
        success: false,
        message: 'Error al cambiar contraseña',
        error: (error as Error).message
      });
    }
  };
}