import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import Accordion from '../../solutions/frontend/FE-007';

const sections = [
  { title: 'First', content: 'First content' },
  { title: 'Second', content: 'Second content' }
];

describe('FE-007: Accordion Component', () => {
  test('renders without crashing', () => {
    render(<Accordion sections={sections} />);
  });

  test('all section titles are visible', () => {
    render(<Accordion sections={sections} />);
    expect(screen.getByRole('button', { name: 'First' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Second' })).toBeInTheDocument();
  });

  test('no content is visible initially', () => {
    render(<Accordion sections={sections} />);
    expect(screen.queryByText('First content')).not.toBeInTheDocument();
    expect(screen.queryByText('Second content')).not.toBeInTheDocument();
  });

  test('clicking a title shows its content', () => {
    render(<Accordion sections={sections} />);
    fireEvent.click(screen.getByRole('button', { name: 'First' }));
    expect(screen.getByText('First content')).toBeInTheDocument();
  });

  test('clicking another title hides the first and shows the second', () => {
    render(<Accordion sections={sections} />);
    fireEvent.click(screen.getByRole('button', { name: 'First' }));
    fireEvent.click(screen.getByRole('button', { name: 'Second' }));
    expect(screen.queryByText('First content')).not.toBeInTheDocument();
    expect(screen.getByText('Second content')).toBeInTheDocument();
  });

  test('clicking the open section closes it', () => {
    render(<Accordion sections={sections} />);
    fireEvent.click(screen.getByRole('button', { name: 'First' }));
    fireEvent.click(screen.getByRole('button', { name: 'First' }));
    expect(screen.queryByText('First content')).not.toBeInTheDocument();
  });

  test('has correct data-testid attributes', () => {
    render(<Accordion sections={sections} />);
    expect(screen.getByTestId('section-0')).toBeInTheDocument();
    expect(screen.getByTestId('section-1')).toBeInTheDocument();
  });

  test('handles single section', () => {
    render(<Accordion sections={[sections[0]]} />);
    fireEvent.click(screen.getByRole('button', { name: 'First' }));
    expect(screen.getByText('First content')).toBeInTheDocument();
  });

  test('handles empty sections array', () => {
    render(<Accordion sections={[]} />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });
});

