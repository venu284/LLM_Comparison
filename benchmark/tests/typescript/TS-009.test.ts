import {
  Entity,
  InMemoryRepository,
  Repository
} from '../../solutions/typescript/TS-009';
import { readSolution } from './helpers';

interface Project extends Entity {
  name: string;
  active: boolean;
}

describe('TS-009: Generic Repository Pattern', () => {
  test('implements the Repository interface', () => {
    const repository: Repository<Project> = new InMemoryRepository<Project>();
    expect(repository.findAll()).toEqual([]);
  });

  test('create generates id and timestamps', () => {
    const repository = new InMemoryRepository<Project>();
    const created = repository.create({ name: 'Benchmark', active: true });

    expect(typeof created.id).toBe('string');
    expect(created.createdAt).toBeInstanceOf(Date);
    expect(created.updatedAt).toBeInstanceOf(Date);
  });

  test('findAll returns created items', () => {
    const repository = new InMemoryRepository<Project>();
    repository.create({ name: 'One', active: true });
    repository.create({ name: 'Two', active: false });

    expect(repository.findAll()).toHaveLength(2);
  });

  test('findById returns the matching item', () => {
    const repository = new InMemoryRepository<Project>();
    const created = repository.create({ name: 'One', active: true });

    expect(repository.findById(created.id)?.name).toBe('One');
  });

  test('findById returns undefined for missing ids', () => {
    const repository = new InMemoryRepository<Project>();
    expect(repository.findById('missing')).toBeUndefined();
  });

  test('update changes fields on the item', () => {
    const repository = new InMemoryRepository<Project>();
    const created = repository.create({ name: 'One', active: true });
    const updated = repository.update(created.id, { active: false });

    expect(updated?.active).toBe(false);
  });

  test('update changes updatedAt', () => {
    const repository = new InMemoryRepository<Project>();
    const created = repository.create({ name: 'One', active: true });
    const updated = repository.update(created.id, { name: 'Renamed' });

    expect(updated?.updatedAt.getTime()).toBeGreaterThanOrEqual(created.updatedAt.getTime());
  });

  test('update returns undefined for missing items', () => {
    const repository = new InMemoryRepository<Project>();
    expect(repository.update('missing', { name: 'Nope' })).toBeUndefined();
  });

  test('delete returns true when an item is removed', () => {
    const repository = new InMemoryRepository<Project>();
    const created = repository.create({ name: 'One', active: true });

    expect(repository.delete(created.id)).toBe(true);
  });

  test('delete returns false for missing items', () => {
    const repository = new InMemoryRepository<Project>();
    expect(repository.delete('missing')).toBe(false);
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-009');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});

