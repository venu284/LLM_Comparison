import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
const { loadBugfixDefaultExport } = require('./helpers');

const Counter = loadBugfixDefaultExport('BF-001');

describe('BF-001: useState Not Updating', () => {
  test('increment updates the displayed count', () => {
    render(<Counter />);
    fireEvent.click(screen.getByRole('button', { name: 'Increment' }));
    expect(screen.getByTestId('count')).toHaveTextContent('1');
  });

  test('decrement updates the displayed count after incrementing', () => {
    render(<Counter />);
    fireEvent.click(screen.getByRole('button', { name: 'Increment' }));
    fireEvent.click(screen.getByRole('button', { name: 'Decrement' }));
    expect(screen.getByTestId('count')).toHaveTextContent('0');
  });

  test('display continues updating across multiple increments', () => {
    render(<Counter />);
    fireEvent.click(screen.getByRole('button', { name: 'Increment' }));
    fireEvent.click(screen.getByRole('button', { name: 'Increment' }));
    expect(screen.getByTestId('count')).toHaveTextContent('2');
  });
});
