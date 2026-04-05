import React, { useState } from 'react';

export default function ToggleSwitch() {
  const [isOn, setIsOn] = useState(false);

  return (
    <button
      type="button"
      data-testid="toggle-btn"
      onClick={() => setIsOn((value) => !value)}
    >
      {isOn ? 'ON' : 'OFF'}
    </button>
  );
}

