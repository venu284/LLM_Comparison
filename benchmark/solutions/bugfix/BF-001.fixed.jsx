import React, { useState } from 'react';

export default function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <div data-testid="count">{count}</div>
      <button type="button" onClick={() => setCount((current) => current + 1)}>
        Increment
      </button>
      <button
        type="button"
        onClick={() => setCount((current) => (current > 0 ? current - 1 : 0))}
      >
        Decrement
      </button>
    </div>
  );
}

