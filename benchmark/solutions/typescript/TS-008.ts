export type Nullable<T> = {
  [K in keyof T]: T[K] | null;
};

export type IsString<T> = T extends string ? true : false;

export type ExtractStrings<T> = {
  [K in keyof T as T[K] extends string ? K : never]: T[K];
};

export function pickStrings<T extends Record<string, unknown>>(obj: T): ExtractStrings<T> {
  const result = {} as ExtractStrings<T>;

  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'string') {
      Object.assign(result, {
        [key]: value
      });
    }
  }

  return result;
}

