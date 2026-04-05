import React from 'react';
import { render, screen } from '@testing-library/react';
import GreetingCard from '../../solutions/frontend/FE-004';

describe('FE-004: User Greeting Card', () => {
  test('renders without crashing', () => {
    render(<GreetingCard />);
  });

  test('displays Hello, Stranger! with no props', () => {
    render(<GreetingCard />);
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent('Hello, Stranger!');
  });

  test('displays Hello, Alice! when name is provided', () => {
    render(<GreetingCard name="Alice" />);
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent('Hello, Alice!');
  });

  test('displays age paragraph when age is provided', () => {
    render(<GreetingCard age={24} />);
    expect(screen.getByText('You are 24 years old.')).toBeInTheDocument();
  });

  test('does not render age paragraph when age is not provided', () => {
    render(<GreetingCard name="Chris" />);
    expect(screen.queryByText(/years old/i)).not.toBeInTheDocument();
  });

  test('uses h2 for greeting and p for age', () => {
    render(<GreetingCard name="Dana" age={33} />);
    expect(screen.getByRole('heading', { level: 2 })).toBeInTheDocument();
    expect(screen.getByText('You are 33 years old.').tagName).toBe('P');
  });
});

