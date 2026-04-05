import React, { useEffect, useState } from 'react';

export default function ResizeTracker() {
  const [width, setWidth] = useState(window.innerWidth);
  const [ticks, setTicks] = useState(0);

  useEffect(() => {
    const handleResize = () => {
      setWidth(window.innerWidth);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <div>
      <div data-testid="width">{width}</div>
      <button type="button" onClick={() => setTicks((current) => current + 1)}>
        Rerender {ticks}
      </button>
    </div>
  );
}

