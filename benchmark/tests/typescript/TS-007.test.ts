import {
  Circle,
  Rectangle,
  Triangle,
  area,
  describe as describeShape,
  isCircle,
  isRectangle,
  isTriangle
} from '../../solutions/typescript/TS-007';
import { readSolution } from './helpers';

describe('TS-007: Type Guards and Narrowing', () => {
  const circle: Circle = { kind: 'circle', radius: 3 };
  const rectangle: Rectangle = { kind: 'rectangle', width: 4, height: 6 };
  const triangle: Triangle = { kind: 'triangle', base: 8, height: 5 };

  test('isCircle narrows circle values', () => {
    expect(isCircle(circle)).toBe(true);
  });

  test('isRectangle narrows rectangle values', () => {
    expect(isRectangle(rectangle)).toBe(true);
  });

  test('isTriangle narrows triangle values', () => {
    expect(isTriangle(triangle)).toBe(true);
  });

  test('area computes circle area', () => {
    expect(area(circle)).toBeCloseTo(Math.PI * 9);
  });

  test('area computes rectangle and triangle areas', () => {
    expect(area(rectangle)).toBe(24);
    expect(area(triangle)).toBe(20);
  });

  test('describe returns expected strings', () => {
    expect(describeShape(circle)).toBe('Circle with radius 3');
    expect(describeShape(rectangle)).toBe('Rectangle 4x6');
  });

  test('describe handles triangles', () => {
    expect(describeShape(triangle)).toBe('Triangle 8x5');
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-007');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});
