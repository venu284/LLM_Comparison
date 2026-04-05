import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import TodoList from '../../solutions/frontend/FE-005';

describe('FE-005: Todo List', () => {
  const addTodo = (value) => {
    fireEvent.change(screen.getByLabelText(/todo/i), { target: { value } });
    fireEvent.click(screen.getByRole('button', { name: 'Add' }));
  };

  test('renders without crashing', () => {
    render(<TodoList />);
  });

  test('input field and Add button are present', () => {
    render(<TodoList />);
    expect(screen.getByLabelText(/todo/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add' })).toBeInTheDocument();
  });

  test('adding a todo shows it in the list', () => {
    render(<TodoList />);
    addTodo('Write tests');
    expect(screen.getByText('Write tests')).toBeInTheDocument();
  });

  test('empty input does not add a todo', () => {
    render(<TodoList />);
    addTodo('   ');
    expect(screen.queryByRole('listitem')).not.toBeInTheDocument();
  });

  test('checkbox toggles completion state', () => {
    render(<TodoList />);
    addTodo('Ship benchmark');
    const checkbox = screen.getByRole('checkbox');
    fireEvent.click(checkbox);
    expect(checkbox).toBeChecked();
  });

  test('completed todo has line-through style', () => {
    render(<TodoList />);
    addTodo('Review task');
    fireEvent.click(screen.getByRole('checkbox'));
    expect(screen.getByText('Review task')).toHaveStyle({ textDecoration: 'line-through' });
  });

  test('Delete button removes todo', () => {
    render(<TodoList />);
    addTodo('Delete me');
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(screen.queryByText('Delete me')).not.toBeInTheDocument();
  });

  test('items left count shows correct number', () => {
    render(<TodoList />);
    addTodo('One');
    addTodo('Two');
    expect(screen.getByText('2 items left')).toBeInTheDocument();
  });

  test('count updates when completing a todo', () => {
    render(<TodoList />);
    addTodo('One');
    addTodo('Two');
    fireEvent.click(screen.getAllByRole('checkbox')[0]);
    expect(screen.getByText('1 items left')).toBeInTheDocument();
  });

  test('can add multiple todos', () => {
    render(<TodoList />);
    addTodo('First');
    addTodo('Second');
    expect(screen.getAllByRole('listitem')).toHaveLength(2);
  });
});

