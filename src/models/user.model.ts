import mongoose, { Document, Schema } from 'mongoose';
import bcrypt from 'bcrypt';

export interface IUser extends Document {
    name?: string;
    email: string;
    password: string;
    createdAt: Date;
    updatedAt: Date;
    role: 'user' | 'admin';
    comparePassword(candidatePassword: string): Promise<boolean>;
    
    // Campos de suscripción
    subscriptionPlan: string;
    subscriptionStatus: 'active' | 'canceled' | 'past_due' | 'trial' | 'unpaid';
    subscriptionStartDate?: Date;
    subscriptionExpiresAt?: Date;
    cancelAtPeriodEnd: boolean;
    nextSubscriptionPlan?: string;
    nextBillingPeriod?: string;
    billingPeriod: 'monthly' | 'yearly';
    
    // Campos para integración con Stripe
    stripeCustomerId?: string;
    stripeSubscriptionId?: string;
    paymentFailCount?: number;
    lastPaymentDate?: Date;
    
    // Cuotas mensuales y su uso
    apiCalls: {
        used: number;
        lastResetDate: Date;
    };
    modelTraining: {
        used: number;
        lastResetDate: Date;
    };
    storage: {
        used: number;
        lastResetDate: Date;
    };

    isActive: boolean; // Nuevo campo para indicar si el usuario está activo
}

const UserSchema = new Schema<IUser>({
    name: {
        type: String,
        trim: true
    },
    email: {
        type: String,
        required: true,
        unique: true,
        trim: true,
        lowercase: true
    },
    password: {
        type: String,
        required: true
    },
    role: {
        type: String,
        enum: ['user', 'admin'],
        default: 'user'
    },
    createdAt: {
        type: Date,
        default: Date.now
    },
    updatedAt: {
        type: Date,
        default: Date.now
    },
    
    // Campos de suscripción
    subscriptionPlan: {
        type: String,
        default: 'free'
    },
    subscriptionStatus: {
        type: String,
        enum: ['active', 'canceled', 'past_due', 'trial', 'unpaid'],
        default: 'active'
    },
    subscriptionStartDate: {
        type: Date
    },
    subscriptionExpiresAt: {
        type: Date
    },
    cancelAtPeriodEnd: {
        type: Boolean,
        default: false
    },
    nextSubscriptionPlan: {
        type: String
    },
    nextBillingPeriod: {
        type: String,
        enum: ['monthly', 'yearly']
    },
    billingPeriod: {
        type: String,
        enum: ['monthly', 'yearly'],
        default: 'monthly'
    },
    
    // Campos para integración con Stripe
    stripeCustomerId: {
        type: String
    },
    stripeSubscriptionId: {
        type: String
    },
    paymentFailCount: {
        type: Number,
        default: 0
    },
    lastPaymentDate: {
        type: Date
    },
    
    // Cuotas mensuales y su uso
    apiCalls: {
        used: {
            type: Number,
            default: 0
        },
        lastResetDate: {
            type: Date,
            default: Date.now
        }
    },
    modelTraining: {
        used: {
            type: Number,
            default: 0
        },
        lastResetDate: {
            type: Date,
            default: Date.now
        }
    },
    storage: {
        used: {
            type: Number,
            default: 0
        },
        lastResetDate: {
            type: Date,
            default: Date.now
        }
    },

    isActive: {
        type: Boolean,
        default: true
    }
});

// Middleware de pre-guardado para hashear contraseñas
UserSchema.pre('save', async function(next) {
    const user = this;
    
    // Actualizar fecha de modificación
    user.updatedAt = new Date();
    
    // Solo hashear si la contraseña se ha modificado o es nueva
    if (!user.isModified('password')) return next();
    
    try {
        const salt = await bcrypt.genSalt(10);
        const hash = await bcrypt.hash(user.password, salt);
        user.password = hash;
        next();
    } catch (error: any) {
        return next(error);
    }
});

// Método para comparar contraseñas
UserSchema.methods.comparePassword = async function(candidatePassword: string): Promise<boolean> {
    try {
        return await bcrypt.compare(candidatePassword, this.password);
    } catch (error) {
        return false;
    }
};

export const User = mongoose.model<IUser>('User', UserSchema);