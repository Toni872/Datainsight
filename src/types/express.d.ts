import { Types } from 'mongoose';
import { IUser } from '../models/user.model';

declare global {
  namespace Express {
    interface Request {
      user?: IUser;
      quotaLimits?: {
        apiCallsPerMonth: number;
        modelTrainingPerMonth: number;
        storageLimit: number;
        advancedModels?: boolean;
        customModels?: boolean;
        dedicatedResources?: boolean;
      };
    }
  }
}

export interface IQuotaInfo {
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
