export function processValue(input: string | number): string {
  if (typeof input === 'string') {
    return input.toUpperCase();
  }

  return input.toFixed(2);
}

