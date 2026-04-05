import { createUser, CreateUserInput, User } from '../../solutions/typescript/TS-001';
import { readSolution } from './helpers';

describe('TS-001: Basic User Interface', () => {
  test('User type accepts the required fields', () => {
    const user: User = {
      id: 'user-1',
      name: 'Ada Lovelace',
      email: 'ada@example.com',
      createdAt: new Date()
    };

    expect(user.email).toBe('ada@example.com');
  });

  test('CreateUserInput omits id and createdAt', () => {
    const input: CreateUserInput = {
      name: 'Grace Hopper',
      email: 'grace@example.com'
    };

    expect(input.name).toBe('Grace Hopper');
  });

  test('createUser returns a valid User object', () => {
    const result = createUser({
      name: 'Linus Torvalds',
      email: 'linus@example.com'
    });

    expect(result.name).toBe('Linus Torvalds');
    expect(result.email).toBe('linus@example.com');
  });

  test('createUser generates a string id', () => {
    const result = createUser({
      name: 'Margaret Hamilton',
      email: 'margaret@example.com'
    });

    expect(typeof result.id).toBe('string');
    expect(result.id.length).toBeGreaterThan(0);
  });

  test('createUser sets createdAt to a Date', () => {
    const result = createUser({
      name: 'Tim Berners-Lee',
      email: 'tim@example.com'
    });

    expect(result.createdAt).toBeInstanceOf(Date);
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-001');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});

