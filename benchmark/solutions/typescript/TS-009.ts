export interface Entity {
  id: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Repository<T extends Entity> {
  findAll(): T[];
  findById(id: string): T | undefined;
  create(input: Omit<T, 'id' | 'createdAt' | 'updatedAt'>): T;
  update(id: string, input: Partial<Omit<T, 'id' | 'createdAt' | 'updatedAt'>>): T | undefined;
  delete(id: string): boolean;
}

const generateId = (): string =>
  'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (character) => {
    const random = Math.floor(Math.random() * 16);
    const value = character === 'x' ? random : (random & 0x3) | 0x8;
    return value.toString(16);
  });

export class InMemoryRepository<T extends Entity> implements Repository<T> {
  private items: T[] = [];

  findAll(): T[] {
    return [...this.items];
  }

  findById(id: string): T | undefined {
    return this.items.find((item) => item.id === id);
  }

  create(input: Omit<T, 'id' | 'createdAt' | 'updatedAt'>): T {
    const now = new Date();
    const entity = {
      ...input,
      id: generateId(),
      createdAt: now,
      updatedAt: now
    } as T;

    this.items.push(entity);
    return entity;
  }

  update(id: string, input: Partial<Omit<T, 'id' | 'createdAt' | 'updatedAt'>>): T | undefined {
    const entity = this.findById(id);

    if (!entity) {
      return undefined;
    }

    const nextUpdatedAt = new Date(Math.max(Date.now(), entity.updatedAt.getTime() + 1));
    Object.assign(entity, input, { updatedAt: nextUpdatedAt });
    return entity;
  }

  delete(id: string): boolean {
    const index = this.items.findIndex((item) => item.id === id);

    if (index === -1) {
      return false;
    }

    this.items.splice(index, 1);
    return true;
  }
}

