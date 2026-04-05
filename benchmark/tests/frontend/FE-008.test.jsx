import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import StarRating from '../../solutions/frontend/FE-008';

describe('FE-008: Star Rating Component', () => {
  test('renders without crashing', () => {
    render(<StarRating />);
  });

  test('shows 5 stars', () => {
    render(<StarRating />);
    expect(screen.getAllByRole('button')).toHaveLength(5);
  });

  test('all stars are empty initially', () => {
    render(<StarRating />);
    expect(screen.getByTestId('star-1')).toHaveTextContent('☆');
    expect(screen.getByTestId('star-5')).toHaveTextContent('☆');
  });

  test('clicking star 3 fills stars 1 through 3', () => {
    render(<StarRating />);
    fireEvent.click(screen.getByTestId('star-3'));
    expect(screen.getByTestId('star-1')).toHaveTextContent('★');
    expect(screen.getByTestId('star-2')).toHaveTextContent('★');
    expect(screen.getByTestId('star-3')).toHaveTextContent('★');
    expect(screen.getByTestId('star-4')).toHaveTextContent('☆');
  });

  test('shows Rating: 3/5 after clicking star 3', () => {
    render(<StarRating />);
    fireEvent.click(screen.getByTestId('star-3'));
    expect(screen.getByText('Rating: 3/5')).toBeInTheDocument();
  });

  test('clicking star 1 changes rating to 1', () => {
    render(<StarRating />);
    fireEvent.click(screen.getByTestId('star-4'));
    fireEvent.click(screen.getByTestId('star-1'));
    expect(screen.getByText('Rating: 1/5')).toBeInTheDocument();
  });

  test('each star has the correct data-testid', () => {
    render(<StarRating />);
    for (let index = 1; index <= 5; index += 1) {
      expect(screen.getByTestId(`star-${index}`)).toBeInTheDocument();
    }
  });

  test('hovering previews the rating without committing it', () => {
    render(<StarRating />);
    fireEvent.mouseEnter(screen.getByTestId('star-4'));
    expect(screen.getByTestId('star-4')).toHaveTextContent('★');
    expect(screen.getByText('Rating: 0/5')).toBeInTheDocument();
  });

  test('rating persists after multiple clicks', () => {
    render(<StarRating />);
    fireEvent.click(screen.getByTestId('star-2'));
    fireEvent.click(screen.getByTestId('star-5'));
    expect(screen.getByText('Rating: 5/5')).toBeInTheDocument();
  });

  test('shows Rating: 0/5 initially', () => {
    render(<StarRating />);
    expect(screen.getByText('Rating: 0/5')).toBeInTheDocument();
  });
});

