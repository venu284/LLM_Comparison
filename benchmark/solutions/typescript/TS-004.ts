export type StringRecord = Record<string, string>;

export function groupBy<T>(items: T[], key: keyof T): Record<string, T[]> {
  return items.reduce<Record<string, T[]>>((groups, item) => {
    const groupKey = String(item[key]);
    const currentGroup = groups[groupKey] ?? [];
    groups[groupKey] = [...currentGroup, item];
    return groups;
  }, {});
}

export function pluck<T, K extends keyof T>(items: T[], key: K): T[K][] {
  return items.map((item) => item[key]);
}

