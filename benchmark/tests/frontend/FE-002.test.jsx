import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ToggleSwitch from '../../solutions/frontend/FE-002';

describe('FE-002: Toggle Switch', () => {
  test('renders without crashing', () => {
    render(<ToggleSwitch />);
  });

  test('initial text is OFF', () => {
    render(<ToggleSwitch />);
    expect(screen.getByTestId('toggle-btn')).toHaveTextContent('OFF');
  });

  test('clicking toggles to ON', () => {
    render(<ToggleSwitch />);
    fireEvent.click(screen.getByTestId('toggle-btn'));
    expect(screen.getByTestId('toggle-btn')).toHaveTextContent('ON');
  });

  test('clicking again toggles back to OFF', () => {
    render(<ToggleSwitch />);
    fireEvent.click(screen.getByTestId('toggle-btn'));
    fireEvent.click(screen.getByTestId('toggle-btn'));
    expect(screen.getByTestId('toggle-btn')).toHaveTextContent('OFF');
  });

  test('has data-testid toggle-btn', () => {
    render(<ToggleSwitch />);
    expect(screen.getByTestId('toggle-btn')).toBeInTheDocument();
  });

  test('only one button is rendered', () => {
    render(<ToggleSwitch />);
    expect(screen.getAllByRole('button')).toHaveLength(1);
  });
});

