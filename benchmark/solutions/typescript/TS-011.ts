export type EventMap = Record<string, unknown>;

type Handler<T extends EventMap, K extends keyof T> = (payload: T[K]) => void;
type ListenerStore<T extends EventMap> = {
  [K in keyof T]?: Set<Handler<T, K>>;
};

export class TypedEventEmitter<T extends EventMap> {
  private listeners: ListenerStore<T> = {};

  private getHandlers<K extends keyof T>(event: K): Set<Handler<T, K>> {
    const existingHandlers = this.listeners[event] as Set<Handler<T, K>> | undefined;

    if (existingHandlers) {
      return existingHandlers;
    }

    const nextHandlers = new Set<Handler<T, K>>();
    this.listeners[event] = nextHandlers as ListenerStore<T>[K];
    return nextHandlers;
  }

  on<K extends keyof T>(event: K, handler: Handler<T, K>): void {
    this.getHandlers(event).add(handler);
  }

  off<K extends keyof T>(event: K, handler: Handler<T, K>): void {
    this.getHandlers(event).delete(handler);
  }

  emit<K extends keyof T>(event: K, payload: T[K]): void {
    this.getHandlers(event).forEach((handler) => {
      handler(payload);
    });
  }

  once<K extends keyof T>(event: K, handler: Handler<T, K>): void {
    const wrapper: Handler<T, K> = (payload) => {
      this.off(event, wrapper);
      handler(payload);
    };

    this.on(event, wrapper);
  }
}

export type AppEvents = {
  'user:login': { userId: string; timestamp: Date };
  'user:logout': { userId: string };
  error: { message: string; code: number };
};

