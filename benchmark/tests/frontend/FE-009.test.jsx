import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import DataTable from '../../solutions/frontend/FE-009';

const rows = [
  { id: 1, name: 'Charlie', email: 'charlie@example.com', role: 'Editor' },
  { id: 2, name: 'Alice', email: 'alice@example.com', role: 'Admin' },
  { id: 3, name: 'Bob', email: 'bob@example.com', role: 'Viewer' },
  { id: 4, name: 'Dana', email: 'dana@example.com', role: 'Manager' },
  { id: 5, name: 'Eli', email: 'eli@example.com', role: 'Author' }
];

describe('FE-009: Paginated Data Table', () => {
  const renderTable = (data = rows) => render(<DataTable data={data} pageSize={2} />);

  test('renders without crashing', () => {
    renderTable();
  });

  test('shows the correct number of rows per page', () => {
    renderTable();
    expect(screen.getAllByRole('row')).toHaveLength(3);
  });

  test('shows Page 1 of Y initially', () => {
    renderTable();
    expect(screen.getByText('Page 1 of 3')).toBeInTheDocument();
  });

  test('Next button advances the page', () => {
    renderTable();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByText('Page 2 of 3')).toBeInTheDocument();
  });

  test('Previous button goes back', () => {
    renderTable();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    fireEvent.click(screen.getByRole('button', { name: 'Previous' }));
    expect(screen.getByText('Page 1 of 3')).toBeInTheDocument();
  });

  test('Previous is disabled on page 1', () => {
    renderTable();
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();
  });

  test('Next is disabled on the last page', () => {
    renderTable();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
  });

  test('clicking Name header sorts by name', () => {
    renderTable();
    fireEvent.click(screen.getByRole('button', { name: 'Name' }));
    expect(screen.getByText('Alice')).toBeInTheDocument();
  });

  test('clicking the header again reverses sort order', () => {
    renderTable();
    fireEvent.click(screen.getByRole('button', { name: 'Name' }));
    fireEvent.click(screen.getByRole('button', { name: /Name/ }));
    expect(screen.getByText('Eli')).toBeInTheDocument();
  });

  test('sort indicator appears on the sorted column', () => {
    renderTable();
    fireEvent.click(screen.getByRole('button', { name: 'Name' }));
    expect(screen.getByRole('button', { name: /Name ▲/ })).toBeInTheDocument();
  });

  test('sorting resets the table back to page 1', () => {
    renderTable();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    fireEvent.click(screen.getByRole('button', { name: 'Name' }));
    expect(screen.getByText('Page 1 of 3')).toBeInTheDocument();
  });

  test('handles empty data array', () => {
    renderTable([]);
    expect(screen.getByText('No data available.')).toBeInTheDocument();
    expect(screen.getByText('Page 1 of 1')).toBeInTheDocument();
  });
});
