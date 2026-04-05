import React, { useEffect, useState } from 'react';

export default function Autocomplete({ fetchSuggestions }) {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!query.trim()) {
      setDebouncedQuery('');
      setSuggestions([]);
      setIsOpen(false);
      setLoading(false);
      return undefined;
    }

    const timer = window.setTimeout(() => {
      setDebouncedQuery(query.trim());
    }, 300);

    return () => window.clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    if (!debouncedQuery) {
      return undefined;
    }

    let active = true;
    setLoading(true);
    setIsOpen(true);

    Promise.resolve(fetchSuggestions(debouncedQuery))
      .then((nextSuggestions) => {
        if (!active) {
          return;
        }

        setSuggestions(nextSuggestions);
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [debouncedQuery, fetchSuggestions]);

  const showDropdown = isOpen && (loading || suggestions.length > 0 || debouncedQuery);

  return (
    <div>
      <label htmlFor="autocomplete-input">Search</label>
      <input
        id="autocomplete-input"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            setIsOpen(false);
          }
        }}
      />

      {showDropdown ? (
        <ul role="listbox">
          {loading ? <li>Loading...</li> : null}
          {!loading && suggestions.length === 0 ? <li>No suggestions</li> : null}
          {!loading
            ? suggestions.map((suggestion) => (
                <li key={suggestion}>
                  <button
                    type="button"
                    onClick={() => {
                      setQuery(suggestion);
                      setIsOpen(false);
                    }}
                  >
                    {suggestion}
                  </button>
                </li>
              ))
            : null}
        </ul>
      ) : null}
    </div>
  );
}

