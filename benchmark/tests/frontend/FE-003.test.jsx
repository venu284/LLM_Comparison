import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import CharCounter from '../../solutions/frontend/FE-003';

describe('FE-003: Character Counter Input', () => {
  test('renders without crashing', () => {
    render(<CharCounter />);
  });

  test('textarea is present', () => {
    render(<CharCounter />);
    expect(screen.getByRole('textbox', { name: /message/i })).toBeInTheDocument();
  });

  test('shows 0 / 200 initially', () => {
    render(<CharCounter />);
    expect(screen.getByText('0 / 200')).toBeInTheDocument();
  });

  test('updates count when typing', () => {
    render(<CharCounter />);
    fireEvent.change(screen.getByRole('textbox', { name: /message/i }), {
      target: { value: 'Hello' }
    });
    expect(screen.getByText('5 / 200')).toBeInTheDocument();
  });

  test('shows correct count for multi-character input', () => {
    render(<CharCounter />);
    fireEvent.change(screen.getByRole('textbox', { name: /message/i }), {
      target: { value: 'Benchmark suite' }
    });
    expect(screen.getByText('15 / 200')).toBeInTheDocument();
  });

  test('adds over-limit class when count is above 200', () => {
    render(<CharCounter />);
    fireEvent.change(screen.getByRole('textbox', { name: /message/i }), {
      target: { value: 'a'.repeat(201) }
    });
    expect(screen.getByText('201 / 200')).toHaveClass('over-limit');
  });

  test('removes over-limit class when count returns under the limit', () => {
    render(<CharCounter />);
    const textarea = screen.getByRole('textbox', { name: /message/i });
    fireEvent.change(textarea, {
      target: { value: 'a'.repeat(201) }
    });
    fireEvent.change(textarea, {
      target: { value: 'a'.repeat(12) }
    });
    expect(screen.getByText('12 / 200')).not.toHaveClass('over-limit');
  });
});

