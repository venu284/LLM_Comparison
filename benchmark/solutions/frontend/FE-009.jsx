import React, { useMemo, useState } from 'react';

const columns = [
  { key: 'name', label: 'Name' },
  { key: 'email', label: 'Email' },
  { key: 'role', label: 'Role' }
];

export default function DataTable({ data = [], pageSize = 5 }) {
  const [page, setPage] = useState(1);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

  const sortedData = useMemo(() => {
    if (!sortConfig.key) {
      return [...data];
    }

    return [...data].sort((left, right) => {
      const leftValue = String(left[sortConfig.key]).toLowerCase();
      const rightValue = String(right[sortConfig.key]).toLowerCase();

      if (leftValue < rightValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }

      if (leftValue > rightValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }

      return 0;
    });
  }, [data, sortConfig]);

  const totalPages = Math.max(1, Math.ceil(sortedData.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const startIndex = (safePage - 1) * pageSize;
  const pageRows = sortedData.slice(startIndex, startIndex + pageSize);

  const toggleSort = (key) => {
    setSortConfig((current) => {
      if (current.key === key) {
        return {
          key,
          direction: current.direction === 'asc' ? 'desc' : 'asc'
        };
      }

      return { key, direction: 'asc' };
    });
    setPage(1);
  };

  const getSortIndicator = (key) => {
    if (sortConfig.key !== key) {
      return '';
    }

    return sortConfig.direction === 'asc' ? ' ▲' : ' ▼';
  };

  return (
    <div>
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>
                <button type="button" onClick={() => toggleSort(column.key)}>
                  {column.label}
                  {getSortIndicator(column.key)}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {pageRows.length === 0 ? (
            <tr>
              <td colSpan={3}>No data available.</td>
            </tr>
          ) : (
            pageRows.map((row) => (
              <tr key={row.id}>
                <td>{row.name}</td>
                <td>{row.email}</td>
                <td>{row.role}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>

      <p>
        Page {safePage} of {totalPages}
      </p>

      <button
        type="button"
        onClick={() => setPage((currentPage) => Math.max(1, currentPage - 1))}
        disabled={safePage === 1}
      >
        Previous
      </button>
      <button
        type="button"
        onClick={() => setPage((currentPage) => Math.min(totalPages, currentPage + 1))}
        disabled={safePage === totalPages}
      >
        Next
      </button>
    </div>
  );
}

