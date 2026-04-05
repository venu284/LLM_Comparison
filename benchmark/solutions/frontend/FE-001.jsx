import React, { useState } from 'react';

export default function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <div data-testid="count">{count}</div>
      <button type="button" onClick={() => setCount((value) => value + 1)}>
        Increment
      </button>
      <button
        type="button"
        onClick={() => setCount((value) => (value > 0 ? value - 1 : 0))}
      >
        Decrement
      </button>
    </div>
  );
}

