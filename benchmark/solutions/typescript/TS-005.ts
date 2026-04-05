export interface ApiResponse<T> {
  data: T;
  status: number;
  message: string;
  timestamp: Date;
}

export interface Pagination {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: Pagination;
}

export interface ApiError {
  status: number;
  code: string;
  message: string;
  details?: Record<string, string>;
}

export function createSuccessResponse<T>(data: T): ApiResponse<T> {
  return {
    data,
    status: 200,
    message: 'ok',
    timestamp: new Date()
  };
}

export function createErrorResponse(status: number, code: string, message: string): ApiError {
  return {
    status,
    code,
    message
  };
}

