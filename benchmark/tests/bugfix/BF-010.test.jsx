import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
const { loadBugfixDefaultExport } = require('./helpers');

const ResizeTracker = loadBugfixDefaultExport('BF-010');

describe('BF-010: Memory Leak in Event Listener', () => {
  test('only one resize listener is registered across rerenders', () => {
    const addSpy = jest.spyOn(window, 'addEventListener');

    render(<ResizeTracker />);
    fireEvent.click(screen.getByRole('button', { name: /rerender/i }));
    fireEvent.click(screen.getByRole('button', { name: /rerender/i }));

    const resizeRegistrations = addSpy.mock.calls.filter(
      ([eventName]) => eventName === 'resize'
    );

    expect(resizeRegistrations).toHaveLength(1);
    addSpy.mockRestore();
  });

  test('cleanup removes the resize listener on unmount', () => {
    const removeSpy = jest.spyOn(window, 'removeEventListener');

    const { unmount } = render(<ResizeTracker />);
    unmount();

    const resizeRemovals = removeSpy.mock.calls.filter(
      ([eventName]) => eventName === 'resize'
    );

    expect(resizeRemovals.length).toBeGreaterThan(0);
    removeSpy.mockRestore();
  });
});
