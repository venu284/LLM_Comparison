import React, { useEffect, useState } from 'react';

export default function Timer() {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setCount((current) => current + 1);
    }, 1000);

    return () => window.clearInterval(intervalId);
  }, []);

  return <div data-testid="count">{count}</div>;
}

