import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import SearchFilter from '../../solutions/frontend/FE-006';

const items = ['Apple', 'Banana', 'Apricot', 'Grape'];

describe('FE-006: Search Filter List', () => {
  test('renders without crashing', () => {
    render(<SearchFilter items={items} />);
  });

  test('shows all items initially', () => {
    render(<SearchFilter items={items} />);
    expect(screen.getAllByRole('listitem')).toHaveLength(4);
  });

  test('search input is present', () => {
    render(<SearchFilter items={items} />);
    expect(screen.getByLabelText(/search/i)).toBeInTheDocument();
  });

  test('typing filters the list case-insensitively', () => {
    render(<SearchFilter items={items} />);
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'ap' } });
    expect(screen.getByText('Apple')).toBeInTheDocument();
    expect(screen.getByText('Apricot')).toBeInTheDocument();
    expect(screen.getByText('Grape')).toBeInTheDocument();
    expect(screen.queryByText('Banana')).not.toBeInTheDocument();
  });

  test('shows No results found when nothing matches', () => {
    render(<SearchFilter items={items} />);
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'zzz' } });
    expect(screen.getByText('No results found.')).toBeInTheDocument();
  });

  test('clearing input shows all items again', () => {
    render(<SearchFilter items={items} />);
    const input = screen.getByLabelText(/search/i);
    fireEvent.change(input, { target: { value: 'banana' } });
    fireEvent.change(input, { target: { value: '' } });
    expect(screen.getAllByRole('listitem')).toHaveLength(4);
  });

  test('shows correct Showing X of Y count', () => {
    render(<SearchFilter items={items} />);
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'ap' } });
    expect(screen.getByText('Showing 3 of 4')).toBeInTheDocument();
  });

  test('handles empty items array', () => {
    render(<SearchFilter items={[]} />);
    expect(screen.getByText('Showing 0 of 0')).toBeInTheDocument();
    expect(screen.getByText('No results found.')).toBeInTheDocument();
  });

  test('partial match works', () => {
    render(<SearchFilter items={items} />);
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'app' } });
    expect(screen.getByText('Apple')).toBeInTheDocument();
    expect(screen.queryByText('Apricot')).not.toBeInTheDocument();
  });
});

