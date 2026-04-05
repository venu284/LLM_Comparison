export interface Config {
  host: string;
  port: number;
  debug?: boolean;
  tags?: string[];
}

export type ReadonlyConfig = Readonly<Config>;

export function mergeConfig(base: Config, overrides: Partial<Config>): Config {
  return {
    ...base,
    ...overrides
  };
}

