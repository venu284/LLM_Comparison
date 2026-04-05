import { Config, mergeConfig } from '../../solutions/typescript/TS-003';
import { readSolution } from './helpers';

describe('TS-003: Readonly and Optional Types', () => {
  test('Config has the correct required fields', () => {
    const config: Config = {
      host: 'localhost',
      port: 3000
    };

    expect(config.host).toBe('localhost');
    expect(config.port).toBe(3000);
  });

  test('debug and tags are optional', () => {
    const config: Config = {
      host: 'localhost',
      port: 3000,
      debug: true,
      tags: ['dev']
    };

    expect(config.debug).toBe(true);
    expect(config.tags).toEqual(['dev']);
  });

  test('ReadonlyConfig is defined using Readonly<Config>', () => {
    const content = readSolution('TS-003');
    expect(content).toContain('Readonly<Config>');
  });

  test('mergeConfig merges overrides into the base config', () => {
    const merged = mergeConfig(
      { host: 'localhost', port: 3000, debug: false },
      { port: 4000, debug: true }
    );

    expect(merged).toEqual({
      host: 'localhost',
      port: 4000,
      debug: true
    });
  });

  test('mergeConfig keeps untouched values from the base config', () => {
    const merged = mergeConfig(
      { host: 'localhost', port: 3000, tags: ['dev'] },
      {}
    );

    expect(merged.tags).toEqual(['dev']);
  });
});

