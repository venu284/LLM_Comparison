import React, { useState } from 'react';

export default function CharCounter() {
  const [value, setValue] = useState('');
  const count = value.length;

  return (
    <div>
      <label htmlFor="char-counter-textarea">Message</label>
      <textarea
        id="char-counter-textarea"
        aria-label="Message"
        value={value}
        onChange={(event) => setValue(event.target.value)}
      />
      <p className={count > 200 ? 'over-limit' : ''}>{count} / 200</p>
    </div>
  );
}

