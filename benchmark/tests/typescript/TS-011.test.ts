import {
  AppEvents,
  TypedEventEmitter
} from '../../solutions/typescript/TS-011';
import { readSolution } from './helpers';

describe('TS-011: Event Emitter with Type-Safe Events', () => {
  test('AppEvents supports the expected payload shape', () => {
    const payload: AppEvents['user:login'] = {
      userId: 'user-1',
      timestamp: new Date()
    };

    expect(payload.userId).toBe('user-1');
  });

  test('on registers a handler and emit triggers it', () => {
    const emitter = new TypedEventEmitter<AppEvents>();
    const handler = jest.fn();

    emitter.on('user:logout', handler);
    emitter.emit('user:logout', { userId: 'user-1' });

    expect(handler).toHaveBeenCalledWith({ userId: 'user-1' });
  });

  test('off removes a registered handler', () => {
    const emitter = new TypedEventEmitter<AppEvents>();
    const handler = jest.fn();

    emitter.on('user:logout', handler);
    emitter.off('user:logout', handler);
    emitter.emit('user:logout', { userId: 'user-1' });

    expect(handler).not.toHaveBeenCalled();
  });

  test('once fires only one time', () => {
    const emitter = new TypedEventEmitter<AppEvents>();
    const handler = jest.fn();

    emitter.once('user:logout', handler);
    emitter.emit('user:logout', { userId: 'user-1' });
    emitter.emit('user:logout', { userId: 'user-1' });

    expect(handler).toHaveBeenCalledTimes(1);
  });

  test('supports multiple handlers per event', () => {
    const emitter = new TypedEventEmitter<AppEvents>();
    const first = jest.fn();
    const second = jest.fn();

    emitter.on('error', first);
    emitter.on('error', second);
    emitter.emit('error', { message: 'Oops', code: 500 });

    expect(first).toHaveBeenCalled();
    expect(second).toHaveBeenCalled();
  });

  test('emits strongly shaped login payloads', () => {
    const emitter = new TypedEventEmitter<AppEvents>();
    const handler = jest.fn();

    emitter.on('user:login', handler);
    emitter.emit('user:login', {
      userId: 'user-2',
      timestamp: new Date('2024-01-01T00:00:00.000Z')
    });

    expect(handler.mock.calls[0][0].userId).toBe('user-2');
  });

  test('error payload exposes message and code', () => {
    const emitter = new TypedEventEmitter<AppEvents>();
    const handler = jest.fn();

    emitter.on('error', handler);
    emitter.emit('error', { message: 'Boom', code: 500 });

    expect(handler).toHaveBeenCalledWith({ message: 'Boom', code: 500 });
  });

  test('logout payload contains userId', () => {
    const emitter = new TypedEventEmitter<AppEvents>();
    const handler = jest.fn();

    emitter.on('user:logout', handler);
    emitter.emit('user:logout', { userId: 'abc' });

    expect(handler.mock.calls[0][0].userId).toBe('abc');
  });

  test('listeners can be re-added after removal', () => {
    const emitter = new TypedEventEmitter<AppEvents>();
    const handler = jest.fn();

    emitter.on('user:logout', handler);
    emitter.off('user:logout', handler);
    emitter.on('user:logout', handler);
    emitter.emit('user:logout', { userId: 'abc' });

    expect(handler).toHaveBeenCalledTimes(1);
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-011');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});

