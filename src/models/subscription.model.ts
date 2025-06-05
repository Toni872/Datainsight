import mongoose, { Document, Schema, Types } from 'mongoose';
import { User } from './user.model';
import logger from '../utils/logger';

interface PriceTier {
    monthly: number;
    yearly: number;
}

export interface ISubscription extends Document {
    userId: Types.ObjectId;
    planId: string;
    status: 'active' | 'canceled' | 'past_due' | 'trial' | 'unpaid';
    startDate: Date;
    endDate: Date;
    cancelAtPeriodEnd: boolean;
    billingPeriod: 'monthly' | 'yearly';
    stripeSubscriptionId?: string;
    currentPeriodStart: Date;
    currentPeriodEnd: Date;
    createdAt: Date;
    updatedAt: Date;
}

export interface IInvoice extends Document {
    userId: Types.ObjectId;
    subscriptionId: Types.ObjectId;
    planId: string;
    amount: number;
    currency: string;
    status: 'paid' | 'unpaid' | 'refunded' | 'failed';
    paymentMethod: string;
    paymentDate?: Date;
    dueDate: Date;
    invoiceNumber: string;
    stripeInvoiceId?: string;
    billingPeriod: 'monthly' | 'yearly';
    createdAt: Date;
    updatedAt: Date;
}

export interface ISubscriptionPlan {
    id: string;
    name: string;
    description: string;
    price: PriceTier;
    features: {
        apiCallsPerMonth: number;
        modelTrainingPerMonth: number;
        storageLimit: number; // en MB
        advancedModels: boolean;
        prioritySupport: boolean;
        customModels: boolean;
        dedicatedResources: boolean;
    };
    popular?: boolean;
    stripeMonthlyPriceId?: string;
    stripeYearlyPriceId?: string;
}

export const SUBSCRIPTION_PLANS: ISubscriptionPlan[] = [
    {
        id: 'free',
        name: 'Plan Gratuito',
        description: 'Perfecto para empezar y explorar la plataforma',
        price: {
            monthly: 0,
            yearly: 0
        },
        features: {
            apiCallsPerMonth: 100,
            modelTrainingPerMonth: 5,
            storageLimit: 50, // 50MB
            advancedModels: false,
            prioritySupport: false,
            customModels: false,
            dedicatedResources: false
        }
    },
    {
        id: 'basic',
        name: 'Plan Básico',
        description: 'Ideal para investigadores, estudiantes y proyectos personales',
        price: {
            monthly: 29.99,
            yearly: 299.99 // ~2 meses gratis
        },
        features: {
            apiCallsPerMonth: 1500, // Aumentado de 1000
            modelTrainingPerMonth: 30, // Aumentado de 20
            storageLimit: 750, // Aumentado de 500MB a 750MB
            advancedModels: true,
            prioritySupport: false,
            customModels: false,
            dedicatedResources: false
        },
        popular: true,
        stripeMonthlyPriceId: 'price_basic_monthly',
        stripeYearlyPriceId: 'price_basic_yearly'
    },
    {
        id: 'pro',
        name: 'Plan Profesional',
        description: 'Potencia máxima para equipos de trabajo y empresas medianas',
        price: {
            monthly: 79.99,
            yearly: 799.99 // ~2 meses gratis
        },
        features: {
            apiCallsPerMonth: 15000, // Aumentado de 10000
            modelTrainingPerMonth: 150, // Aumentado de 100
            storageLimit: 3072, // Aumentado de 2GB a 3GB
            advancedModels: true,
            prioritySupport: true,
            customModels: true,
            dedicatedResources: false
        },
        stripeMonthlyPriceId: 'price_pro_monthly',
        stripeYearlyPriceId: 'price_pro_yearly'
    },
    {
        id: 'enterprise',
        name: 'Plan Empresarial',
        description: 'Solución completa con recursos dedicados para grandes organizaciones',
        price: {
            monthly: 199.99,
            yearly: 1999.99 // ~2 meses gratis
        },
        features: {
            apiCallsPerMonth: 100000, // Aumentado de 50000
            modelTrainingPerMonth: 1000, // Aumentado de 500
            storageLimit: 20480, // Aumentado de 10GB a 20GB
            advancedModels: true,
            prioritySupport: true,
            customModels: true,
            dedicatedResources: true
        },
        stripeMonthlyPriceId: 'price_enterprise_monthly',
        stripeYearlyPriceId: 'price_enterprise_yearly'
    }
];

// Define la interfaz y el esquema para las transacciones de suscripción
export interface ISubscriptionTransaction extends Document {
    userId: mongoose.Types.ObjectId;
    planId: string;
    amount: number;
    currency: string;
    billingPeriod: 'monthly' | 'yearly';
    status: 'pending' | 'completed' | 'failed' | 'refunded';
    paymentMethod: string;
    paymentId?: string;
    createdAt: Date;
    periodStart: Date;
    periodEnd: Date;
    metadata?: Record<string, any>;
}

const SubscriptionTransactionSchema = new Schema<ISubscriptionTransaction>({
    userId: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    planId: {
        type: String,
        required: true
    },
    amount: {
        type: Number,
        required: true
    },
    currency: {
        type: String,
        required: true,
        default: 'EUR'
    },
    billingPeriod: {
        type: String,
        enum: ['monthly', 'yearly'],
        required: true
    },
    status: {
        type: String,
        enum: ['pending', 'completed', 'failed', 'refunded'],
        default: 'pending'
    },
    paymentMethod: {
        type: String,
        required: true
    },
    paymentId: {
        type: String
    },
    createdAt: {
        type: Date,
        default: Date.now
    },
    periodStart: {
        type: Date,
        required: true
    },
    periodEnd: {
        type: Date,
        required: true
    },
    metadata: {
        type: Schema.Types.Mixed
    }
});

// Crear el esquema de Subscription
const SubscriptionSchema = new Schema<ISubscription>({
    userId: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    planId: {
        type: String,
        required: true
    },
    status: {
        type: String,
        enum: ['active', 'canceled', 'past_due', 'trial', 'unpaid'],
        default: 'active'
    },
    startDate: {
        type: Date,
        default: Date.now
    },
    endDate: {
        type: Date,
        required: true
    },
    cancelAtPeriodEnd: {
        type: Boolean,
        default: false
    },
    billingPeriod: {
        type: String,
        enum: ['monthly', 'yearly'],
        default: 'monthly'
    },
    stripeSubscriptionId: {
        type: String
    },
    currentPeriodStart: {
        type: Date,
        default: Date.now
    },
    currentPeriodEnd: {
        type: Date,
        required: true
    },
    createdAt: {
        type: Date,
        default: Date.now
    },
    updatedAt: {
        type: Date,
        default: Date.now
    }
});

// Crear el esquema de Invoice
const InvoiceSchema = new Schema<IInvoice>({
    userId: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    subscriptionId: {
        type: Schema.Types.ObjectId,
        ref: 'Subscription',
        required: true
    },
    planId: {
        type: String,
        required: true
    },
    amount: {
        type: Number,
        required: true
    },
    currency: {
        type: String,
        default: 'EUR'
    },
    status: {
        type: String,
        enum: ['paid', 'unpaid', 'refunded', 'failed'],
        default: 'unpaid'
    },
    paymentMethod: {
        type: String,
        required: true
    },
    paymentDate: {
        type: Date
    },
    dueDate: {
        type: Date,
        required: true
    },
    invoiceNumber: {
        type: String,
        required: true,
        unique: true
    },
    stripeInvoiceId: {
        type: String
    },
    billingPeriod: {
        type: String,
        enum: ['monthly', 'yearly'],
        default: 'monthly'
    },
    createdAt: {
        type: Date,
        default: Date.now
    },
    updatedAt: {
        type: Date,
        default: Date.now
    }
});

// Configurar pre-save hook para generar número de factura automáticamente
InvoiceSchema.pre('save', async function(next) {
    if (!this.invoiceNumber) {
        const date = new Date();
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        
        // Contar facturas existentes en este mes/año
        const invoiceCount = await Invoice.countDocuments({
            createdAt: {
                $gte: new Date(date.getFullYear(), date.getMonth(), 1),
                $lt: new Date(date.getFullYear(), date.getMonth() + 1, 0)
            }
        });
        
        // Generar número de factura (Formato: INV-AAAAMMDD-XXX)
        const day = String(date.getDate()).padStart(2, '0');
        const count = String(invoiceCount + 1).padStart(3, '0');
        this.invoiceNumber = `INV-${year}${month}${day}-${count}`;
    }
    next();
});

// Crear los modelos
export const Subscription = mongoose.model<ISubscription>('Subscription', SubscriptionSchema);
export const Invoice = mongoose.model<IInvoice>('Invoice', InvoiceSchema);

// Funciones de utilidad para manejar planes
export const getPlanById = (planId: string): ISubscriptionPlan | undefined => {
    return SUBSCRIPTION_PLANS.find(plan => plan.id === planId);
};

export const getPlanPrice = (planId: string, billingPeriod: 'monthly' | 'yearly'): number => {
    const plan = getPlanById(planId);
    if (!plan) return 0;
    return billingPeriod === 'monthly' ? plan.price.monthly : plan.price.yearly;
};

export const hasAccessToModelType = (planId: string, modelType: 'basic' | 'advanced' | 'custom'): boolean => {
    const plan = getPlanById(planId);
    if (!plan) return false;

    switch (modelType) {
        case 'basic':
            return true; // Todos los planes tienen acceso a modelos básicos
        case 'advanced':
            return !!plan.features.advancedModels;
        case 'custom':
            return !!plan.features.customModels;
        default:
            return false;
    }
};

export const SubscriptionTransaction = mongoose.model<ISubscriptionTransaction>(
    'SubscriptionTransaction', 
    SubscriptionTransactionSchema
);

/**
 * Procesa una nueva suscripción para un usuario
 */
export const processNewSubscription = async (
    userId: Types.ObjectId, 
    planId: string, 
    billingPeriod: 'monthly' | 'yearly', 
    paymentMethod: string,
    stripePriceId?: string,
    stripeSubscriptionId?: string,
    metadata?: any
): Promise<ISubscription> => {
    try {
        // Obtener información del plan
        const plan = getPlanById(planId);
        if (!plan) {
            throw new Error(`Plan no válido: ${planId}`);
        }
        
        // Calcular fechas de inicio y fin
        const startDate = new Date();
        let endDate = new Date(startDate);
        
        if (billingPeriod === 'monthly') {
            endDate.setMonth(endDate.getMonth() + 1);
        } else { // yearly
            endDate.setFullYear(endDate.getFullYear() + 1);
        }
        
        // Crear nueva suscripción
        const subscription = new Subscription({
            userId,
            planId,
            status: 'active',
            startDate,
            endDate,
            cancelAtPeriodEnd: false,
            billingPeriod,
            stripeSubscriptionId,
            currentPeriodStart: startDate,
            currentPeriodEnd: endDate
        });
        
        // Guardar suscripción
        const savedSubscription = await subscription.save();
        
        // Crear factura asociada a la suscripción
        const price = billingPeriod === 'monthly' ? plan.price.monthly : plan.price.yearly;
        
        const invoice = new Invoice({
            userId,
            subscriptionId: savedSubscription._id,
            planId,
            amount: price,
            currency: 'EUR', // Podrías parametrizar esto si tienes diferentes monedas
            status: 'paid',
            paymentMethod,
            paymentDate: new Date(),
            dueDate: new Date(),
            stripeInvoiceId: metadata?.stripeInvoiceId,
            billingPeriod
        });
        
        // Guardar factura
        await invoice.save();
        
        // Actualizar usuario con la información de suscripción
        await User.findByIdAndUpdate(userId, {
            subscriptionPlan: planId,
            subscriptionStatus: 'active',
            subscriptionStartDate: startDate,
            subscriptionExpiresAt: endDate,
            cancelAtPeriodEnd: false,
            billingPeriod,
            stripeCustomerId: metadata?.stripeCustomerId,
            stripeSubscriptionId
        });
        
        // Registrar transacción
        const transaction = new SubscriptionTransaction({
            userId,
            planId,
            amount: price,
            billingPeriod,
            status: 'completed',
            paymentMethod,
            paymentId: metadata?.paymentId,
            periodStart: startDate,
            periodEnd: endDate,
            metadata
        });
        
        await transaction.save();
        
        return savedSubscription;
    } catch (error: any) {
        logger.error(`Error al procesar nueva suscripción: ${error.message}`);
        throw error;
    }
};