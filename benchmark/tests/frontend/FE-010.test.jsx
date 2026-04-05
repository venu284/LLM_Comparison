import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import FormWizard from '../../solutions/frontend/FE-010';

describe('FE-010: Multi-Step Form Wizard', () => {
  const fillStepOne = () => {
    fireEvent.change(screen.getByLabelText(/name/i), { target: { value: 'Alice' } });
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'alice@example.com' } });
  };

  const fillStepTwo = () => {
    fireEvent.change(screen.getByLabelText(/address/i), { target: { value: '123 Main St' } });
    fireEvent.change(screen.getByLabelText(/city/i), { target: { value: 'New York' } });
  };

  test('renders without crashing', () => {
    render(<FormWizard />);
  });

  test('shows Step 1 of 3 initially', () => {
    render(<FormWizard />);
    expect(screen.getByText('Step 1 of 3')).toBeInTheDocument();
  });

  test('step 1 has name and email inputs', () => {
    render(<FormWizard />);
    expect(screen.getByLabelText(/name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
  });

  test('cannot advance with empty fields', () => {
    render(<FormWizard />);
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByText('Step 1 of 3')).toBeInTheDocument();
  });

  test('shows error messages for empty fields', () => {
    render(<FormWizard />);
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getAllByText('This field is required')).toHaveLength(2);
  });

  test('filling fields and clicking Next goes to step 2', () => {
    render(<FormWizard />);
    fillStepOne();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByText('Step 2 of 3')).toBeInTheDocument();
  });

  test('step 2 has address and city inputs', () => {
    render(<FormWizard />);
    fillStepOne();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByLabelText(/address/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/city/i)).toBeInTheDocument();
  });

  test('Previous button goes back to step 1 with data preserved', () => {
    render(<FormWizard />);
    fillStepOne();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    fireEvent.click(screen.getByRole('button', { name: 'Previous' }));
    expect(screen.getByDisplayValue('Alice')).toBeInTheDocument();
    expect(screen.getByDisplayValue('alice@example.com')).toBeInTheDocument();
  });

  test('step 3 shows a review of all entered data', () => {
    render(<FormWizard />);
    fillStepOne();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    fillStepTwo();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByText('Name: Alice')).toBeInTheDocument();
    expect(screen.getByText('City: New York')).toBeInTheDocument();
  });

  test('step 3 has Submit button instead of Next', () => {
    render(<FormWizard />);
    fillStepOne();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    fillStepTwo();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByRole('button', { name: 'Submit' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Next' })).not.toBeInTheDocument();
  });

  test('Submit calls onSubmit with the correct data', () => {
    const onSubmit = jest.fn();
    render(<FormWizard onSubmit={onSubmit} />);
    fillStepOne();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    fillStepTwo();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    fireEvent.click(screen.getByRole('button', { name: 'Submit' }));
    expect(onSubmit).toHaveBeenCalledWith({
      name: 'Alice',
      email: 'alice@example.com',
      address: '123 Main St',
      city: 'New York'
    });
  });

  test('no Previous button is shown on step 1', () => {
    render(<FormWizard />);
    expect(screen.queryByRole('button', { name: 'Previous' })).not.toBeInTheDocument();
  });

  test('step indicator updates correctly', () => {
    render(<FormWizard />);
    fillStepOne();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByText('Step 2 of 3')).toBeInTheDocument();
  });
});

