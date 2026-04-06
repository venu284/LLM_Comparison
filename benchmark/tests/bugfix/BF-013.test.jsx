import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
const { loadBugfixDefaultExport } = require('./helpers');

const Summary = loadBugfixDefaultExport('BF-013');

describe('BF-013: Deep Equality vs Reference Equality', () => {
  test('initial render performs the calculation once and shows the total', () => {
    const onCalculate = jest.fn();

    render(<Summary items={[1, 2, 3]} multiplier={2} onCalculate={onCalculate} />);

    expect(screen.getByTestId('total')).toHaveTextContent('12');
    expect(onCalculate).toHaveBeenCalledTimes(1);
  });

  test('unrelated rerenders do not recompute the memoized value', () => {
    const onCalculate = jest.fn();

    render(<Summary items={[1, 2, 3]} multiplier={2} onCalculate={onCalculate} />);
    fireEvent.click(screen.getByRole('button', { name: /rerender/i }));

    expect(onCalculate).toHaveBeenCalledTimes(1);
  });

  test('changing inputs recomputes the value', () => {
    const onCalculate = jest.fn();
    const { rerender } = render(
      <Summary items={[1, 2, 3]} multiplier={2} onCalculate={onCalculate} />
    );

    rerender(<Summary items={[1, 2, 3]} multiplier={3} onCalculate={onCalculate} />);

    expect(screen.getByTestId('total')).toHaveTextContent('18');
    expect(onCalculate).toHaveBeenCalledTimes(2);
  });

  test('default multiplier of one computes the raw total correctly', () => {
    const onCalculate = jest.fn();

    render(<Summary items={[1, 2, 3]} onCalculate={onCalculate} />);

    expect(screen.getByTestId('total')).toHaveTextContent('6');
    expect(onCalculate).toHaveBeenCalledTimes(1);
  });

  test('repeated unrelated rerenders still do not recompute the memoized value', () => {
    const onCalculate = jest.fn();

    render(<Summary items={[1, 2, 3]} multiplier={2} onCalculate={onCalculate} />);
    fireEvent.click(screen.getByRole('button', { name: /rerender/i }));
    fireEvent.click(screen.getByRole('button', { name: /rerender/i }));
    fireEvent.click(screen.getByRole('button', { name: /rerender/i }));

    expect(onCalculate).toHaveBeenCalledTimes(1);
  });
});
