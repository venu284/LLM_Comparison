import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
const { loadBugfixDefaultExport } = require('./helpers');

const Counter = loadBugfixDefaultExport('BF-001');

describe('BF-001: useState Not Updating', () => {
  const clickIncrement = () => {
    fireEvent.click(screen.getByRole('button', { name: 'Increment' }));
  };

  const clickDecrement = () => {
    fireEvent.click(screen.getByRole('button', { name: 'Decrement' }));
  };

  test('increment updates the displayed count', () => {
    render(<Counter />);
    clickIncrement();
    expect(screen.getByTestId('count')).toHaveTextContent('1');
  });

  test('decrement updates the displayed count after incrementing', () => {
    render(<Counter />);
    clickIncrement();
    clickDecrement();
    expect(screen.getByTestId('count')).toHaveTextContent('0');
  });

  test('display continues updating across multiple increments', () => {
    render(<Counter />);
    clickIncrement();
    clickIncrement();
    expect(screen.getByTestId('count')).toHaveTextContent('2');
  });

  test('decrement at zero keeps the count at zero', () => {
    render(<Counter />);
    clickDecrement();
    expect(screen.getByTestId('count')).toHaveTextContent('0');
  });

  test('mixed increment and decrement sequence updates after each click', () => {
    render(<Counter />);

    clickIncrement();
    expect(screen.getByTestId('count')).toHaveTextContent('1');

    clickIncrement();
    expect(screen.getByTestId('count')).toHaveTextContent('2');

    clickDecrement();
    expect(screen.getByTestId('count')).toHaveTextContent('1');

    clickIncrement();
    expect(screen.getByTestId('count')).toHaveTextContent('2');
  });

  test('repeated decrements never move the count below zero', () => {
    render(<Counter />);

    clickIncrement();
    clickDecrement();
    clickDecrement();

    expect(screen.getByTestId('count')).toHaveTextContent('0');
  });
});
