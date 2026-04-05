import { OrderState, transition } from '../../solutions/typescript/TS-006';
import { readSolution } from './helpers';

describe('TS-006: Discriminated Union for State Machine', () => {
  const createdAt = new Date('2024-01-01T00:00:00.000Z');

  test('pending can transition to confirmed', () => {
    const next = transition(
      { status: 'pending', createdAt },
      { type: 'confirm', confirmedAt: new Date('2024-01-02T00:00:00.000Z') }
    );

    expect(next.status).toBe('confirmed');
  });

  test('confirmed can transition to shipped', () => {
    const next = transition(
      {
        status: 'confirmed',
        createdAt,
        confirmedAt: new Date('2024-01-02T00:00:00.000Z')
      },
      { type: 'ship', trackingNumber: 'TRACK-123' }
    );

    expect(next.status).toBe('shipped');
  });

  test('shipped can transition to delivered', () => {
    const next = transition(
      {
        status: 'shipped',
        createdAt,
        confirmedAt: new Date('2024-01-02T00:00:00.000Z'),
        shippedAt: new Date('2024-01-03T00:00:00.000Z'),
        trackingNumber: 'TRACK-123'
      },
      { type: 'deliver' }
    );

    expect(next.status).toBe('delivered');
  });

  test('non-delivered orders can be cancelled', () => {
    const next = transition(
      {
        status: 'confirmed',
        createdAt,
        confirmedAt: new Date('2024-01-02T00:00:00.000Z')
      },
      { type: 'cancel', reason: 'Customer request' }
    );

    expect(next.status).toBe('cancelled');
  });

  test('invalid transitions throw an error', () => {
    expect(() =>
      transition(
        { status: 'pending', createdAt },
        { type: 'deliver' }
      )
    ).toThrow('Invalid transition');
  });

  test('state narrowing exposes shipped properties safely', () => {
    const state: OrderState = {
      status: 'shipped',
      createdAt,
      confirmedAt: new Date('2024-01-02T00:00:00.000Z'),
      shippedAt: new Date('2024-01-03T00:00:00.000Z'),
      trackingNumber: 'TRACK-1'
    };

    if (state.status === 'shipped') {
      const tracking: string = state.trackingNumber;
      expect(tracking).toBe('TRACK-1');
    }
  });

  test('cancelled state includes a reason', () => {
    const next = transition(
      { status: 'pending', createdAt },
      { type: 'cancel', reason: 'Out of stock' }
    );

    expect(next.status).toBe('cancelled');
    if (next.status === 'cancelled') {
      expect(next.reason).toBe('Out of stock');
    }
  });

  test('delivered orders cannot be cancelled again', () => {
    expect(() =>
      transition(
        {
          status: 'delivered',
          createdAt,
          confirmedAt: new Date('2024-01-02T00:00:00.000Z'),
          shippedAt: new Date('2024-01-03T00:00:00.000Z'),
          trackingNumber: 'TRACK-1',
          deliveredAt: new Date('2024-01-04T00:00:00.000Z')
        },
        { type: 'cancel', reason: 'Too late' }
      )
    ).toThrow('Invalid transition');
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-006');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});

