import mongoose, { Document, Schema } from 'mongoose';

interface PriceTier {
    monthly: number;
    yearly: number;
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
        description: 'Para pequeños equipos y proyectos iniciales',
        price: {
            monthly: 29.99,
            yearly: 299.99 // ~2 meses gratis
        },
        features: {
            apiCallsPerMonth: 1000,
            modelTrainingPerMonth: 20,
            storageLimit: 250, // 250MB
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
        id: 'professional',
        name: 'Plan Profesional',
        description: 'Para equipos en crecimiento con necesidades avanzadas',
        price: {
            monthly: 79.99,
            yearly: 799.99 // ~2 meses gratis
        },
        features: {
            apiCallsPerMonth: 5000,
            modelTrainingPerMonth: 100,
            storageLimit: 1024, // 1GB
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
        description: 'Recursos dedicados para grandes organizaciones',
        price: {
            monthly: 199.99,
            yearly: 1999.99 // ~2 meses gratis
        },
        features: {
            apiCallsPerMonth: 50000,
            modelTrainingPerMonth: 500,
            storageLimit: 10240, // 10GB
            advancedModels: true,
            prioritySupport: true,
            customModels: true,
            dedicatedResources: true
        },
        stripeMonthlyPriceId: 'price_enterprise_monthly',
        stripeYearlyPriceId: 'price_enterprise_yearly'
    }
];

/**
 * Obtiene un plan por su ID
 * @param planId ID del plan o 'all' para todos los planes
 * @returns Plan o array de planes
 */
export const getPlanById = (planId: string): ISubscriptionPlan | ISubscriptionPlan[] => {
    if (planId === 'all') {
        return SUBSCRIPTION_PLANS;
    }
    const plan = SUBSCRIPTION_PLANS.find(p => p.id === planId);
    return plan || SUBSCRIPTION_PLANS[0]; // Por defecto el plan gratuito
};

/**
 * Obtiene el precio de un plan según su ID y período de facturación
 * @param planId ID del plan
 * @param billingPeriod Período de facturación ('monthly' o 'yearly')
 * @returns Precio del plan
 */
export const getPlanPrice = (planId: string, billingPeriod: 'monthly' | 'yearly'): number => {
    const plan = getPlanById(planId);
    if (Array.isArray(plan)) {
        return 0;
    }
    return plan.price[billingPeriod];
};

/**
 * Verifica si un usuario tiene acceso a un tipo específico de modelo
 * @param planId ID del plan de suscripción
 * @param modelType Tipo de modelo ('basic', 'advanced', 'custom')
 * @returns true si tiene acceso, false si no
 */
export const hasAccessToModelType = (planId: string, modelType: 'basic' | 'advanced' | 'custom'): boolean => {
    const plan = getPlanById(planId);
    if (Array.isArray(plan)) {
        return false;
    }
    
    // Todos los planes pueden acceder a modelos básicos
    if (modelType === 'basic') return true;
    
    // Solo planes que tienen la característica pueden acceder a modelos avanzados
    if (modelType === 'advanced') return plan.features.advancedModels;
    
    // Solo planes que tienen la característica pueden acceder a modelos personalizados
    if (modelType === 'custom') return plan.features.customModels;
    
    return false;
};

// Esquema para transacciones de suscripción
const subscriptionTransactionSchema = new Schema({
    userId: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    type: {
        type: String,
        enum: ['subscription_created', 'subscription_updated', 'subscription_canceled', 'payment_succeeded', 'payment_failed'],
        required: true
    },
    planId: {
        type: String,
        required: true
    },
    amount: {
        type: Number,
        required: false
    },
    currency: {
        type: String,
        default: 'USD'
    },
    stripeInvoiceId: {
        type: String
    },
    stripeSubscriptionId: {
        type: String
    },
    status: {
        type: String,
        enum: ['succeeded', 'failed', 'pending'],
        default: 'succeeded'
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

export interface ISubscriptionTransaction extends Document {
    userId: mongoose.Types.ObjectId;
    type: 'subscription_created' | 'subscription_updated' | 'subscription_canceled' | 'payment_succeeded' | 'payment_failed';
    planId: string;
    amount?: number;
    currency: string;
    stripeInvoiceId?: string;
    stripeSubscriptionId?: string;
    status: 'succeeded' | 'failed' | 'pending';
    createdAt: Date;
}

// Intentar obtener el modelo existente o crear uno nuevo
export const SubscriptionTransaction = mongoose.models.SubscriptionTransaction || 
  mongoose.model<ISubscriptionTransaction>('SubscriptionTransaction', subscriptionTransactionSchema);

export default {
    getPlanById,
    getPlanPrice,
    hasAccessToModelType,
    SubscriptionTransaction
};
