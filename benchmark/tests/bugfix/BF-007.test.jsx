import React from 'react';
import { render } from '@testing-library/react';
const { loadBugfixDefaultExport } = require('./helpers');

const DataFetcher = loadBugfixDefaultExport('BF-007');

describe('BF-007: Async State Update After Unmount', () => {
  test('fetchData receives an AbortSignal', () => {
    let capturedSignal;

    const fetchData = jest.fn((signal) => {
      capturedSignal = signal;
      return new Promise(() => {});
    });

    render(<DataFetcher fetchData={fetchData} />);

    expect(capturedSignal).toBeInstanceOf(AbortSignal);
  });

  test('cleanup aborts the signal on unmount', () => {
    let capturedSignal;

    const fetchData = jest.fn((signal) => {
      capturedSignal = signal;
      return new Promise(() => {});
    });

    const { unmount } = render(<DataFetcher fetchData={fetchData} />);

    unmount();

    expect(capturedSignal.aborted).toBe(true);
  });
});
