import { HttpRequest, RequestBuilder } from '../../solutions/typescript/TS-010';
import { readSolution } from './helpers';

describe('TS-010: Builder Pattern with Type Safety', () => {
  test('builder is chainable', () => {
    const builder = new RequestBuilder()
      .setUrl('/api/users')
      .setMethod('GET')
      .setTimeout(1000);

    expect(builder).toBeInstanceOf(RequestBuilder);
  });

  test('build returns an HttpRequest', () => {
    const request: HttpRequest = new RequestBuilder()
      .setUrl('/api/users')
      .setMethod('GET')
      .build();

    expect(request.method).toBe('GET');
  });

  test('setUrl and setMethod are required before build', () => {
    const content = readSolution('TS-010');
    expect(content).toMatch(/build\(this:\s*RequestBuilder<true,\s*true>\)/);
  });

  test('setHeader accumulates headers', () => {
    const request = new RequestBuilder()
      .setUrl('/api/users')
      .setMethod('GET')
      .setHeader('Accept', 'application/json')
      .setHeader('X-Test', 'yes')
      .build();

    expect(request.headers).toEqual({
      Accept: 'application/json',
      'X-Test': 'yes'
    });
  });

  test('setBody stores body values', () => {
    const request = new RequestBuilder()
      .setUrl('/api/users')
      .setMethod('POST')
      .setBody({ name: 'Ada' })
      .build();

    expect(request.body).toEqual({ name: 'Ada' });
  });

  test('setTimeout stores timeout values', () => {
    const request = new RequestBuilder()
      .setUrl('/api/users')
      .setMethod('GET')
      .setTimeout(2500)
      .build();

    expect(request.timeout).toBe(2500);
  });

  test('url and method values are preserved', () => {
    const request = new RequestBuilder()
      .setUrl('/health')
      .setMethod('PUT')
      .build();

    expect(request.url).toBe('/health');
    expect(request.method).toBe('PUT');
  });

  test('headers are optional', () => {
    const request = new RequestBuilder()
      .setUrl('/health')
      .setMethod('GET')
      .build();

    expect(request.headers).toBeUndefined();
  });

  test('body is optional', () => {
    const request = new RequestBuilder()
      .setUrl('/health')
      .setMethod('GET')
      .build();

    expect(request.body).toBeUndefined();
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-010');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});

