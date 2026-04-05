import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import Modal from '../../solutions/frontend/FE-013';

const renderModal = (props = {}) =>
  render(
    <Modal isOpen onClose={jest.fn()} title="Dialog Title" {...props}>
      <button type="button">First action</button>
      <button type="button">Second action</button>
    </Modal>
  );

describe('FE-013: Modal with Focus Trap', () => {
  test('renders nothing when isOpen is false', () => {
    render(
      <Modal isOpen={false} onClose={jest.fn()} title="Closed">
        <button type="button">Child</button>
      </Modal>
    );
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('renders modal when isOpen is true', () => {
    renderModal();
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  test('shows the title in the modal', () => {
    renderModal();
    expect(screen.getByText('Dialog Title')).toBeInTheDocument();
  });

  test('renders children content', () => {
    renderModal();
    expect(screen.getByRole('button', { name: 'First action' })).toBeInTheDocument();
  });

  test('close button calls onClose', () => {
    const onClose = jest.fn();
    render(
      <Modal isOpen onClose={onClose} title="Dialog Title">
        <button type="button">First action</button>
      </Modal>
    );
    fireEvent.click(screen.getByRole('button', { name: /close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  test('Escape key calls onClose', () => {
    const onClose = jest.fn();
    render(
      <Modal isOpen onClose={onClose} title="Dialog Title">
        <button type="button">First action</button>
      </Modal>
    );
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });

  test('backdrop click calls onClose', () => {
    const onClose = jest.fn();
    render(
      <Modal isOpen onClose={onClose} title="Dialog Title">
        <button type="button">First action</button>
      </Modal>
    );
    fireEvent.click(screen.getByTestId('modal-backdrop'));
    expect(onClose).toHaveBeenCalled();
  });

  test('has role dialog', () => {
    renderModal();
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  test('has aria-modal true', () => {
    renderModal();
    expect(screen.getByRole('dialog')).toHaveAttribute('aria-modal', 'true');
  });

  test('focus moves to the first focusable element on open', () => {
    renderModal();
    expect(screen.getByRole('button', { name: /close/i })).toHaveFocus();
  });

  test('Tab key cycles focus within the modal', () => {
    renderModal();
    const closeButton = screen.getByRole('button', { name: /close/i });
    const firstAction = screen.getByRole('button', { name: 'First action' });
    const secondAction = screen.getByRole('button', { name: 'Second action' });

    expect(closeButton).toHaveFocus();
    fireEvent.keyDown(document, { key: 'Tab' });
    expect(firstAction).toHaveFocus();
    fireEvent.keyDown(document, { key: 'Tab' });
    expect(secondAction).toHaveFocus();
    fireEvent.keyDown(document, { key: 'Tab' });
    expect(closeButton).toHaveFocus();
  });
});
