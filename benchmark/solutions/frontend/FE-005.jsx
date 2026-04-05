import React, { useState } from 'react';

export default function TodoList() {
  const [inputValue, setInputValue] = useState('');
  const [todos, setTodos] = useState([]);

  const addTodo = () => {
    const trimmedValue = inputValue.trim();
    if (!trimmedValue) {
      return;
    }

    setTodos((currentTodos) => [
      ...currentTodos,
      {
        id: `${Date.now()}-${currentTodos.length}`,
        text: trimmedValue,
        completed: false
      }
    ]);
    setInputValue('');
  };

  const remainingCount = todos.filter((todo) => !todo.completed).length;

  return (
    <div>
      <label htmlFor="todo-input">Todo</label>
      <input
        id="todo-input"
        value={inputValue}
        onChange={(event) => setInputValue(event.target.value)}
      />
      <button type="button" onClick={addTodo}>
        Add
      </button>

      <p>{remainingCount} items left</p>

      <ul>
        {todos.map((todo) => (
          <li key={todo.id}>
            <label>
              <input
                type="checkbox"
                checked={todo.completed}
                onChange={() =>
                  setTodos((currentTodos) =>
                    currentTodos.map((item) =>
                      item.id === todo.id ? { ...item, completed: !item.completed } : item
                    )
                  )
                }
              />
              <span
                style={{
                  textDecoration: todo.completed ? 'line-through' : 'none'
                }}
              >
                {todo.text}
              </span>
            </label>
            <button
              type="button"
              onClick={() =>
                setTodos((currentTodos) => currentTodos.filter((item) => item.id !== todo.id))
              }
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

