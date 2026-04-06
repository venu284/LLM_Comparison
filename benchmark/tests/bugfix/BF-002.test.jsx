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

  test('initial render shows the full three-item todo list', () => {
    render(<TodoList />);

    expect(screen.getAllByRole('listitem')).toHaveLength(3);
    expect(screen.getByText('Write tests')).toBeInTheDocument();
    expect(screen.getByText('Fix bugs')).toBeInTheDocument();
    expect(screen.getByText('Ship project')).toBeInTheDocument();
  });

  test('deleting the first item preserves the remaining todos', () => {
    render(<TodoList />);

    fireEvent.click(screen.getAllByRole('button', { name: 'Delete' })[0]);

    expect(screen.queryByText('Write tests')).not.toBeInTheDocument();
    expect(screen.getAllByRole('listitem')).toHaveLength(2);
    expect(screen.getByText('Fix bugs')).toBeInTheDocument();
    expect(screen.getByText('Ship project')).toBeInTheDocument();
  });

  test('sequential deletes reduce the rendered list to zero items', () => {
    render(<TodoList />);

    fireEvent.click(screen.getAllByRole('button', { name: 'Delete' })[0]);
    fireEvent.click(screen.getAllByRole('button', { name: 'Delete' })[0]);
    fireEvent.click(screen.getAllByRole('button', { name: 'Delete' })[0]);

    expect(screen.queryAllByRole('listitem')).toHaveLength(0);
    expect(screen.queryAllByRole('button', { name: 'Delete' })).toHaveLength(0);
  });
});
