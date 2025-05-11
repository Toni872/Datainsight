import { Request, Response, NextFunction } from 'express';

export const authenticate = (req: Request, res: Response, next: NextFunction): void | Response => {
    // Middleware for authentication
    const token = req.headers['authorization'];
    if (!token) {
        return res.status(401).json({ message: 'No token provided' });
    }
    // Logic to verify token
    return next();
};

export const validateRequest = (req: Request, res: Response, next: NextFunction): void | Response => {
    // Middleware for request validation
    const { body } = req;
    if (!body || Object.keys(body).length === 0) {
        return res.status(400).json({ message: 'Request body is required' });
    }
    // Additional validation logic can go here
    next();
};

export const errorHandler = (err: Error, req: Request, res: Response, next: NextFunction): void => {
    console.error(err.stack);
    
    const statusCode = (err as any).statusCode || 500;
    const message = err.message || 'Error interno del servidor';
    
    res.status(statusCode).json({
        status: 'error',
        statusCode,
        message
    });
};