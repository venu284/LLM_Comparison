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
      {todos.map((todo, index) => (
        <li>
          <span>{todo.text}</span>
          <button
            type="button"
            onClick={() => setTodos((current) => current.filter((_, itemIndex) => itemIndex !== index))}
          >
            Delete
          </button>
        </li>
      ))}
    </ul>
  );
}

