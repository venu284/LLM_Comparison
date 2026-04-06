import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
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

  test('resolved fetch displays the loaded data', async () => {
    render(<DataFetcher fetchData={jest.fn().mockResolvedValue('Loaded')} />);

    await waitFor(() => {
      expect(screen.getByText('Loaded')).toBeInTheDocument();
    });
  });

  test('rejected non-abort fetch shows an error state', async () => {
    render(<DataFetcher fetchData={jest.fn().mockRejectedValue(new Error('Request failed'))} />);

    await waitFor(() => {
      expect(screen.getByText('Error')).toBeInTheDocument();
    });
  });

  test('component can unmount and remount cleanly', async () => {
    const fetchData = jest.fn().mockResolvedValue('Loaded');
    const firstRender = render(<DataFetcher fetchData={fetchData} />);

    firstRender.unmount();
    render(<DataFetcher fetchData={fetchData} />);

    await waitFor(() => {
      expect(screen.getByText('Loaded')).toBeInTheDocument();
    });
    expect(fetchData).toHaveBeenCalledTimes(2);
  });
});
