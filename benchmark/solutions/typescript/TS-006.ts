export type PendingOrder = {
  status: 'pending';
  createdAt: Date;
};

export type ConfirmedOrder = {
  status: 'confirmed';
  createdAt: Date;
  confirmedAt: Date;
};

export type ShippedOrder = {
  status: 'shipped';
  createdAt: Date;
  confirmedAt: Date;
  shippedAt: Date;
  trackingNumber: string;
};

export type DeliveredOrder = {
  status: 'delivered';
  createdAt: Date;
  confirmedAt: Date;
  shippedAt: Date;
  trackingNumber: string;
  deliveredAt: Date;
};

export type CancelledOrder = {
  status: 'cancelled';
  createdAt: Date;
  cancelledAt: Date;
  reason: string;
};

export type OrderState =
  | PendingOrder
  | ConfirmedOrder
  | ShippedOrder
  | DeliveredOrder
  | CancelledOrder;

export type OrderAction =
  | { type: 'confirm'; confirmedAt?: Date }
  | { type: 'ship'; trackingNumber: string; shippedAt?: Date }
  | { type: 'deliver'; deliveredAt?: Date }
  | { type: 'cancel'; reason: string; cancelledAt?: Date };

export function transition(state: OrderState, action: OrderAction): OrderState {
  switch (action.type) {
    case 'confirm':
      if (state.status !== 'pending') {
        throw new Error('Invalid transition');
      }
      return {
        status: 'confirmed',
        createdAt: state.createdAt,
        confirmedAt: action.confirmedAt ?? new Date()
      };
    case 'ship':
      if (state.status !== 'confirmed') {
        throw new Error('Invalid transition');
      }
      return {
        status: 'shipped',
        createdAt: state.createdAt,
        confirmedAt: state.confirmedAt,
        shippedAt: action.shippedAt ?? new Date(),
        trackingNumber: action.trackingNumber
      };
    case 'deliver':
      if (state.status !== 'shipped') {
        throw new Error('Invalid transition');
      }
      return {
        status: 'delivered',
        createdAt: state.createdAt,
        confirmedAt: state.confirmedAt,
        shippedAt: state.shippedAt,
        trackingNumber: state.trackingNumber,
        deliveredAt: action.deliveredAt ?? new Date()
      };
    case 'cancel':
      if (state.status === 'delivered' || state.status === 'cancelled') {
        throw new Error('Invalid transition');
      }
      return {
        status: 'cancelled',
        createdAt: state.createdAt,
        cancelledAt: action.cancelledAt ?? new Date(),
        reason: action.reason
      };
  }
}

