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
});
