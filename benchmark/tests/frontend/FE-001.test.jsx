import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import Counter from '../../solutions/frontend/FE-001';

describe('FE-001: Counter Component', () => {
  test('renders without crashing', () => {
    render(<Counter />);
  });

  test('shows initial count of 0', () => {
    render(<Counter />);
    expect(screen.getByTestId('count')).toHaveTextContent('0');
  });

  test('increment button increases count by 1', () => {
    render(<Counter />);
    fireEvent.click(screen.getByRole('button', { name: 'Increment' }));
    expect(screen.getByTestId('count')).toHaveTextContent('1');
  });

  test('decrement button decreases count by 1', () => {
    render(<Counter />);
    fireEvent.click(screen.getByRole('button', { name: 'Increment' }));
    fireEvent.click(screen.getByRole('button', { name: 'Decrement' }));
    expect(screen.getByTestId('count')).toHaveTextContent('0');
  });

  test('count does not go below 0', () => {
    render(<Counter />);
    fireEvent.click(screen.getByRole('button', { name: 'Decrement' }));
    expect(screen.getByTestId('count')).toHaveTextContent('0');
  });

  test('has data-testid on count display', () => {
    render(<Counter />);
    expect(screen.getByTestId('count')).toBeInTheDocument();
  });

  test('both buttons are present and clickable', () => {
    render(<Counter />);
    expect(screen.getByRole('button', { name: 'Increment' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Decrement' })).toBeEnabled();
  });
});

