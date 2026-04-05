import {
  ApiError,
  ApiResponse,
  PaginatedResponse,
  createErrorResponse,
  createSuccessResponse
} from '../../solutions/typescript/TS-005';
import { readSolution } from './helpers';

describe('TS-005: Generic API Response Wrapper', () => {
  test('ApiResponse is generic and accepts typed data', () => {
    const response: ApiResponse<string> = {
      data: 'ok',
      status: 200,
      message: 'success',
      timestamp: new Date()
    };

    expect(response.data).toBe('ok');
  });

  test('PaginatedResponse includes pagination metadata', () => {
    const response: PaginatedResponse<number> = {
      data: [1, 2, 3],
      status: 200,
      message: 'success',
      timestamp: new Date(),
      pagination: {
        page: 1,
        limit: 10,
        total: 3,
        totalPages: 1
      }
    };

    expect(response.pagination.total).toBe(3);
  });

  test('ApiError includes the required fields', () => {
    const error: ApiError = {
      status: 400,
      code: 'BAD_REQUEST',
      message: 'Something went wrong'
    };

    expect(error.code).toBe('BAD_REQUEST');
  });

  test('createSuccessResponse returns a success wrapper', () => {
    const response = createSuccessResponse({ id: '1' });
    expect(response.status).toBe(200);
    expect(response.data).toEqual({ id: '1' });
  });

  test('createSuccessResponse sets a timestamp', () => {
    const response = createSuccessResponse(42);
    expect(response.timestamp).toBeInstanceOf(Date);
  });

  test('createErrorResponse returns an ApiError', () => {
    const response = createErrorResponse(404, 'NOT_FOUND', 'Missing');
    expect(response).toEqual({
      status: 404,
      code: 'NOT_FOUND',
      message: 'Missing'
    });
  });

  test('generics work with multiple data shapes', () => {
    const numberResponse = createSuccessResponse(123);
    const objectResponse = createSuccessResponse({ active: true });
    expect(numberResponse.data).toBe(123);
    expect(objectResponse.data.active).toBe(true);
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-005');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});

