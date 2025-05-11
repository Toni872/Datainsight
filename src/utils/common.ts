/**
 * Utilidades para formatear fechas
 */
export class DateUtils {
  /**
   * Formatea una fecha en formato español
   * @param date Fecha a formatear
   * @returns Fecha formateada en español
   */
  static formatDate(date: Date): string {
    return date.toLocaleDateString('es-ES', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric'
    });
  }
  
  /**
   * Obtiene la fecha actual formateada
   * @returns Fecha actual formateada en español
   */
  static getCurrentDate(): string {
    return this.formatDate(new Date());
  }
  
  /**
   * Calcula la diferencia en días entre dos fechas
   * @param date1 Primera fecha
   * @param date2 Segunda fecha
   * @returns Número de días de diferencia
   */
  static daysBetween(date1: Date, date2: Date): number {
    const oneDay = 24 * 60 * 60 * 1000; // milisegundos en un día
    return Math.round(Math.abs((date1.getTime() - date2.getTime()) / oneDay));
  }
}

/**
 * Utilidades para validar datos
 */
export class ValidationUtils {
  /**
   * Valida un correo electrónico
   * @param email Correo electrónico a validar
   * @returns true si el correo es válido
   */
  static isValidEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }
  
  /**
   * Valida que un string no esté vacío
   * @param value Valor a validar
   * @returns true si el valor no está vacío
   */
  static isNotEmpty(value: string): boolean {
    return value !== undefined && value !== null && value.trim() !== '';
  }
  
  /**
   * Valida que un número esté dentro de un rango
   * @param value Valor a validar
   * @param min Valor mínimo
   * @param max Valor máximo
   * @returns true si el valor está dentro del rango
   */
  static isInRange(value: number, min: number, max: number): boolean {
    return value >= min && value <= max;
  }
}