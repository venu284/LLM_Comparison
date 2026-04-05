export type DeepPartial<T> = T extends Array<infer U>
  ? Array<DeepPartial<U>>
  : T extends object
    ? { [K in keyof T]?: DeepPartial<T[K]> }
    : T;

export type DeepReadonly<T> = T extends Array<infer U>
  ? ReadonlyArray<DeepReadonly<U>>
  : T extends object
    ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
    : T;

export type DeepRequired<T> = T extends Array<infer U>
  ? Array<DeepRequired<NonNullable<U>>>
  : T extends object
    ? { [K in keyof T]-?: DeepRequired<NonNullable<T[K]>> }
    : T;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    !(value instanceof Date)
  );
}

function mergeObjects(
  target: Record<string, unknown>,
  source: Record<string, unknown>
): Record<string, unknown> {
  const result: Record<string, unknown> = { ...target };

  for (const key of Object.keys(source)) {
    const sourceValue = source[key];

    if (sourceValue === undefined) {
      continue;
    }

    const targetValue = result[key];

    if (isPlainObject(targetValue) && isPlainObject(sourceValue)) {
      result[key] = mergeObjects(targetValue, sourceValue);
    } else {
      result[key] = sourceValue;
    }
  }

  return result;
}

export function deepMerge<T extends Record<string, unknown>>(
  target: T,
  source: DeepPartial<T>
): T {
  return mergeObjects(target, source as Record<string, unknown>) as T;
}

