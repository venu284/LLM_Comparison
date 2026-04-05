import React from 'react';

export default function GreetingCard({ name = 'Stranger', age }) {
  return (
    <section>
      <h2>Hello, {name}!</h2>
      {typeof age === 'number' ? <p>You are {age} years old.</p> : null}
    </section>
  );
}

