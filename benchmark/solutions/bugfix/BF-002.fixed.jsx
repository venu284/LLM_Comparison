import React, { useState } from 'react';

const initialTodos = [
  { id: '1', text: 'Write tests' },
  { id: '2', text: 'Fix bugs' },
  { id: '3', text: 'Ship project' }
];

export default function TodoList() {
  const [todos, setTodos] = useState(initialTodos);

  return (
    <ul>
      {todos.map((todo) => (
        <li key={todo.id} data-testid={`todo-${todo.id}`}>
          <span>{todo.text}</span>
          <button
            type="button"
            onClick={() => setTodos((current) => current.filter((item) => item.id !== todo.id))}
          >
            Delete
          </button>
        </li>
      ))}
    </ul>
  );
}

