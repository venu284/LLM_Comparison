import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import Autocomplete from '../../solutions/frontend/FE-011';

describe('FE-011: Debounced Autocomplete', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('renders without crashing', () => {
    render(<Autocomplete fetchSuggestions={jest.fn().mockResolvedValue([])} />);
  });

  test('input field is present', () => {
    render(<Autocomplete fetchSuggestions={jest.fn().mockResolvedValue([])} />);
    expect(screen.getByLabelText(/search/i)).toBeInTheDocument();
  });

  test('no dropdown is shown initially', () => {
    render(<Autocomplete fetchSuggestions={jest.fn().mockResolvedValue([])} />);
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });

  test('typing triggers suggestions after the debounce delay', async () => {
    const fetchSuggestions = jest.fn().mockResolvedValue(['Alpha']);
    render(<Autocomplete fetchSuggestions={fetchSuggestions} />);
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'al' } });

    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => expect(fetchSuggestions).toHaveBeenCalledWith('al'));
  });

  test('shows Loading while fetching', async () => {
    let resolvePromise;
    const fetchSuggestions = jest.fn(
      () =>
        new Promise((resolve) => {
          resolvePromise = resolve;
        })
    );

    render(<Autocomplete fetchSuggestions={fetchSuggestions} />);
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'al' } });

    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    expect(screen.getByText('Loading...')).toBeInTheDocument();

    await act(async () => {
      resolvePromise(['Alpha']);
    });
  });

  test('displays returned suggestions', async () => {
    const fetchSuggestions = jest.fn().mockResolvedValue(['Alpha', 'Beta']);
    render(<Autocomplete fetchSuggestions={fetchSuggestions} />);
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'a' } });

    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument());
    expect(screen.getByText('Beta')).toBeInTheDocument();
  });

  test('shows No suggestions when the result is empty', async () => {
    render(<Autocomplete fetchSuggestions={jest.fn().mockResolvedValue([])} />);
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'x' } });

    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => expect(screen.getByText('No suggestions')).toBeInTheDocument());
  });

  test('clicking a suggestion fills the input', async () => {
    render(<Autocomplete fetchSuggestions={jest.fn().mockResolvedValue(['Alpha'])} />);
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'a' } });

    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: 'Alpha' }));
    expect(screen.getByLabelText(/search/i)).toHaveValue('Alpha');
  });

  test('clicking a suggestion closes the dropdown', async () => {
    render(<Autocomplete fetchSuggestions={jest.fn().mockResolvedValue(['Alpha'])} />);
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'a' } });

    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: 'Alpha' }));
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });

  test('Escape closes the dropdown', async () => {
    render(<Autocomplete fetchSuggestions={jest.fn().mockResolvedValue(['Alpha'])} />);
    const input = screen.getByLabelText(/search/i);
    fireEvent.change(input, { target: { value: 'a' } });

    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument());
    fireEvent.keyDown(input, { key: 'Escape' });
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });

  test('empty input closes the dropdown', async () => {
    render(<Autocomplete fetchSuggestions={jest.fn().mockResolvedValue(['Alpha'])} />);
    const input = screen.getByLabelText(/search/i);
    fireEvent.change(input, { target: { value: 'a' } });

    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument());
    fireEvent.change(input, { target: { value: '' } });
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });
});

