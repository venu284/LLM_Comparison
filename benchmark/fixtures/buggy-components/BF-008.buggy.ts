export function processValue(input: string | number): string {
  if (typeof input === 'string') {
    return (input as unknown as number).toFixed(2);
  }

  return (input as unknown as string).toUpperCase();
}

