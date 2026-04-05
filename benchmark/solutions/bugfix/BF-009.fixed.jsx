import React, { useEffect, useRef, useState } from 'react';

async function defaultSearchFn() {
  return [];
}

export default function AsyncSearch({ searchFn = defaultSearchFn }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const latestRequestId = useRef(0);

  useEffect(() => {
    if (!query) {
      latestRequestId.current += 1;
      setResults([]);
      return;
    }

    const requestId = latestRequestId.current + 1;
    latestRequestId.current = requestId;

    searchFn(query).then((nextResults) => {
      if (latestRequestId.current === requestId) {
        setResults(nextResults);
      }
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

