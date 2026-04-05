import React, { useMemo, useState } from 'react';

export default function Summary({ items, multiplier = 1, onCalculate = () => {} }) {
  const [ticks, setTicks] = useState(0);

  const total = useMemo(() => {
    onCalculate();
    return items.reduce((sum, item) => sum + item, 0) * multiplier;
  }, [items, multiplier, onCalculate]);

  return (
    <div>
      <div data-testid="total">{total}</div>
      <button type="button" onClick={() => setTicks((current) => current + 1)}>
        Rerender {ticks}
      </button>
    </div>
  );
}

