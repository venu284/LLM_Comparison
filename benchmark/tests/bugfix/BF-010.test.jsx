import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
const { loadBugfixDefaultExport } = require('./helpers');

const ResizeTracker = loadBugfixDefaultExport('BF-010');
const originalInnerWidth = window.innerWidth;

function setInnerWidth(value) {
  Object.defineProperty(window, 'innerWidth', {
    configurable: true,
    writable: true,
    value
  });
}

describe('BF-010: Memory Leak in Event Listener', () => {
  beforeEach(() => {
    setInnerWidth(1024);
  });

  afterEach(() => {
    setInnerWidth(originalInnerWidth);
    jest.restoreAllMocks();
  });

  test('component initially renders the current width', () => {
    render(<ResizeTracker />);
    expect(screen.getByTestId('width')).toHaveTextContent('1024');
  });

  test('only one resize listener is registered across rerenders', () => {
    const addSpy = jest.spyOn(window, 'addEventListener');

    render(<ResizeTracker />);
    fireEvent.click(screen.getByRole('button', { name: /rerender/i }));
    fireEvent.click(screen.getByRole('button', { name: /rerender/i }));

    const resizeRegistrations = addSpy.mock.calls.filter(
      ([eventName]) => eventName === 'resize'
    );

    expect(resizeRegistrations).toHaveLength(1);
  });

  test('cleanup removes the resize listener on unmount', () => {
    const removeSpy = jest.spyOn(window, 'removeEventListener');

    const { unmount } = render(<ResizeTracker />);
    unmount();

    const resizeRemovals = removeSpy.mock.calls.filter(
      ([eventName]) => eventName === 'resize'
    );

    expect(resizeRemovals.length).toBeGreaterThan(0);
  });

  test('resize events update the displayed width', () => {
    render(<ResizeTracker />);

    setInnerWidth(640);
    fireEvent(window, new Event('resize'));

    expect(screen.getByTestId('width')).toHaveTextContent('640');
  });

  test('width continues updating after multiple rerenders', () => {
    render(<ResizeTracker />);

    fireEvent.click(screen.getByRole('button', { name: /rerender/i }));
    fireEvent.click(screen.getByRole('button', { name: /rerender/i }));
    setInnerWidth(480);
    fireEvent(window, new Event('resize'));

    expect(screen.getByTestId('width')).toHaveTextContent('480');
  });
});
