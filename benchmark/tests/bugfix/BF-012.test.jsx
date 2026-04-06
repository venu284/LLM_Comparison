import React from 'react';
import { render, screen } from '@testing-library/react';
const { loadBugfixDefaultExport } = require('./helpers');

const DataLoader = loadBugfixDefaultExport('BF-012');

describe('BF-012: Infinite Re-render Loop', () => {
  test('component renders fetched data', () => {
    render(<DataLoader fetchData={() => 'Loaded'} />);
    expect(screen.getByText('Loaded')).toBeInTheDocument();
  });

  test('fetchData is called only once', () => {
    const fetchData = jest.fn(() => 'Loaded');

    render(<DataLoader fetchData={fetchData} />);

    expect(fetchData).toHaveBeenCalledTimes(1);
  });

  test('rerendering with the same fetch function does not refetch', () => {
    const fetchData = jest.fn(() => 'Loaded');
    const { rerender } = render(<DataLoader fetchData={fetchData} />);

    rerender(<DataLoader fetchData={fetchData} />);

    expect(fetchData).toHaveBeenCalledTimes(1);
  });

  test('rerendering with a new fetch function updates the data once', () => {
    const initialFetch = jest.fn(() => 'Loaded');
    const nextFetch = jest.fn(() => 'Updated');
    const { rerender } = render(<DataLoader fetchData={initialFetch} />);

    rerender(<DataLoader fetchData={nextFetch} />);

    expect(screen.getByText('Updated')).toBeInTheDocument();
    expect(nextFetch).toHaveBeenCalledTimes(1);
  });

  test('empty string results render without crashing or looping', () => {
    const fetchData = jest.fn(() => '');
    const { container } = render(<DataLoader fetchData={fetchData} />);

    expect(fetchData).toHaveBeenCalledTimes(1);
    expect(container.firstChild).toBeEmptyDOMElement();
  });
});
