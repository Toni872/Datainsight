export interface UserModel {
    id?: number;
    username: string;
    email: string;
    password: string;
    createdAt?: Date;
    updatedAt?: Date;
}

export class User implements UserModel {
    id?: number;
    username: string;
    email: string;
    password: string;
    createdAt: Date;
    updatedAt: Date;

    constructor(data: UserModel) {
        this.id = data.id;
        this.username = data.username;
        this.email = data.email; // Corregido: data.email en lugar de this.email
        this.password = data.password;
        this.createdAt = data.createdAt || new Date();
        this.updatedAt = data.updatedAt || new Date();
    }

    // Método para validar un email
    static validateEmail(email: string): boolean {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
}

export class Product {
    constructor(public id: number, public name: string, public price: number) {}
}