import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
const { loadBugfixDefaultExport } = require('./helpers');

const TodoList = loadBugfixDefaultExport('BF-002');

describe('BF-002: Missing Key Prop in List', () => {
  test('rendering the list does not produce the React key warning', () => {
    const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    render(<TodoList />);

    const keyWarning = errorSpy.mock.calls.find((call) =>
      call.some((value) => String(value).includes('unique "key" prop'))
    );

    expect(keyWarning).toBeUndefined();
    errorSpy.mockRestore();
  });

  test('deleting the middle item removes the correct todo', () => {
    render(<TodoList />);

    fireEvent.click(screen.getAllByRole('button', { name: 'Delete' })[1]);

    expect(screen.queryByText('Fix bugs')).not.toBeInTheDocument();
    expect(screen.getByText('Write tests')).toBeInTheDocument();
    expect(screen.getByText('Ship project')).toBeInTheDocument();
  });
});
