export interface Circle {
  kind: 'circle';
  radius: number;
}

export interface Rectangle {
  kind: 'rectangle';
  width: number;
  height: number;
}

export interface Triangle {
  kind: 'triangle';
  base: number;
  height: number;
}

export type Shape = Circle | Rectangle | Triangle;

export function isCircle(shape: Shape): shape is Circle {
  return shape.kind === 'circle';
}

export function isRectangle(shape: Shape): shape is Rectangle {
  return shape.kind === 'rectangle';
}

export function isTriangle(shape: Shape): shape is Triangle {
  return shape.kind === 'triangle';
}

function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${String(value)}`);
}

export function area(shape: Shape): number {
  switch (shape.kind) {
    case 'circle':
      return Math.PI * shape.radius * shape.radius;
    case 'rectangle':
      return shape.width * shape.height;
    case 'triangle':
      return (shape.base * shape.height) / 2;
  }

  return assertNever(shape);
}

export function describe(shape: Shape): string {
  switch (shape.kind) {
    case 'circle':
      return `Circle with radius ${shape.radius}`;
    case 'rectangle':
      return `Rectangle ${shape.width}x${shape.height}`;
    case 'triangle':
      return `Triangle ${shape.base}x${shape.height}`;
  }

  return assertNever(shape);
}
