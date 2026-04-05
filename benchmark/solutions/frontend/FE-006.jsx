import React, { useState } from 'react';

export default function SearchFilter({ items = [] }) {
  const [query, setQuery] = useState('');

  const filteredItems = items.filter((item) =>
    item.toLowerCase().includes(query.trim().toLowerCase())
  );

  return (
    <div>
      <label htmlFor="search-filter-input">Search</label>
      <input
        id="search-filter-input"
        aria-label="Search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />

      <p>
        Showing {filteredItems.length} of {items.length}
      </p>

      {filteredItems.length === 0 ? (
        <p>No results found.</p>
      ) : (
        <ul>
          {filteredItems.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

