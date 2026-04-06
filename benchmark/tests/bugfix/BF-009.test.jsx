import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
const { loadBugfixDefaultExport } = require('./helpers');

const AsyncSearch = loadBugfixDefaultExport('BF-009');

describe('BF-009: Race Condition in Async Search', () => {
  const createControlledSearch = () => {
    const resolvers = {};

    const searchFn = jest.fn(
      (query) =>
        new Promise((resolve) => {
          resolvers[query] = resolve;
        })
    );

    return { searchFn, resolvers };
  };

  test('shows results for the final query only', async () => {
    const { searchFn, resolvers } = createControlledSearch();
    render(<AsyncSearch searchFn={searchFn} />);

    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'ab' } });
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'abc' } });

    resolvers.abc(['abc result']);

    await waitFor(() => expect(screen.getByText('abc result')).toBeInTheDocument());
  });

  test('stale intermediate results are discarded when they resolve late', async () => {
    const { searchFn, resolvers } = createControlledSearch();
    render(<AsyncSearch searchFn={searchFn} />);

    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'ab' } });
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'abc' } });

    resolvers.abc(['abc result']);

    await waitFor(() => expect(screen.getByText('abc result')).toBeInTheDocument());

    resolvers.ab(['ab result']);

    await waitFor(() => {
      expect(screen.queryByText('ab result')).not.toBeInTheDocument();
      expect(screen.getByText('abc result')).toBeInTheDocument();
    });
  });

  test('clearing the input removes existing results', async () => {
    const searchFn = jest.fn().mockResolvedValue(['alpha']);
    render(<AsyncSearch searchFn={searchFn} />);

    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'a' } });

    await waitFor(() => expect(screen.getByText('alpha')).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: '' } });

    await waitFor(() => {
      expect(screen.queryByText('alpha')).not.toBeInTheDocument();
    });
  });

  test('a single slow query eventually renders its results', async () => {
    const { searchFn, resolvers } = createControlledSearch();
    render(<AsyncSearch searchFn={searchFn} />);

    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'hello' } });
    expect(searchFn).toHaveBeenCalledWith('hello');

    resolvers.hello(['hello result']);

    await waitFor(() => {
      expect(screen.getByText('hello result')).toBeInTheDocument();
    });
  });

  test('clearing the query before a late response prevents stale results from reappearing', async () => {
    const { resolvers, searchFn } = createControlledSearch();
    render(<AsyncSearch searchFn={searchFn} />);

    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'ab' } });
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: '' } });

    resolvers.ab(['ab result']);

    await waitFor(() => {
      expect(screen.queryByText('ab result')).not.toBeInTheDocument();
    });
  });
});
