import React from 'react';
import { act, render, screen } from '@testing-library/react';
const { loadBugfixDefaultExport, readBugfixSource } = require('./helpers');

const Timer = loadBugfixDefaultExport('BF-005');
const source = readBugfixSource('BF-005');

describe('BF-005: Stale Closure in useEffect', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('count increments after one second', () => {
    render(<Timer />);

    act(() => {
      jest.advanceTimersByTime(1000);
    });

    expect(screen.getByTestId('count')).toHaveTextContent('1');
  });

  test('count continues incrementing over time', () => {
    render(<Timer />);

    act(() => {
      jest.advanceTimersByTime(3000);
    });

    expect(screen.getByTestId('count')).toHaveTextContent('3');
  });

  test('display matches the actual count after five seconds', () => {
    render(<Timer />);

    act(() => {
      jest.advanceTimersByTime(5000);
    });

    expect(screen.getByTestId('count')).toHaveTextContent('5');
  });

  test('source includes clearInterval cleanup for the interval', () => {
    expect(source).toMatch(/clearInterval/);
  });

  test('component unmounts without throwing errors', () => {
    const { unmount } = render(<Timer />);
    expect(() => unmount()).not.toThrow();
  });

  test('advancing timers after unmount does not throw', () => {
    const { unmount } = render(<Timer />);
    unmount();

    expect(() => {
      act(() => {
        jest.advanceTimersByTime(3000);
      });
    }).not.toThrow();
  });
});
