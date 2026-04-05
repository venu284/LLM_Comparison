import React, { useState } from 'react';

export default function Counter() {
  const [count] = useState(0);

  const increment = () => {
    let nextCount = count;
    nextCount++;
  };

  const decrement = () => {
    if (count === 0) {
      return;
    }

    let nextCount = count;
    nextCount--;
  };

  return (
    <div>
      <div data-testid="count">{count}</div>
      <button type="button" onClick={increment}>
        Increment
      </button>
      <button type="button" onClick={decrement}>
        Decrement
      </button>
    </div>
  );
}

