import React, { useEffect, useState } from 'react';

export default function AsyncSearch({ searchFn }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  useEffect(() => {
    if (!query) {
      setResults([]);
      return;
    }

    searchFn(query).then((nextResults) => {
      setResults(nextResults);
    });
  }, [query, searchFn]);

  return (
    <div>
      <input
        aria-label="Search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <ul>
        {results.map((result) => (
          <li key={result}>{result}</li>
        ))}
      </ul>
    </div>
  );
}

