import React, { useEffect, useRef } from 'react';

const focusableSelector =
  'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

export default function Modal({ isOpen, onClose, title, children }) {
  const dialogRef = useRef(null);

  useEffect(() => {
    if (!isOpen || !dialogRef.current) {
      return undefined;
    }

    const getFocusableElements = () =>
      Array.from(dialogRef.current.querySelectorAll(focusableSelector)).filter(
        (element) => !element.hasAttribute('disabled')
      );

    const focusableElements = getFocusableElements();
    if (focusableElements.length > 0) {
      focusableElements[0].focus();
    }

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        onClose();
        return;
      }

      if (event.key !== 'Tab') {
        return;
      }

      const elements = getFocusableElements();
      if (elements.length === 0) {
        return;
      }

      const currentIndex = elements.indexOf(document.activeElement);
      const nextIndex = event.shiftKey
        ? currentIndex <= 0
          ? elements.length - 1
          : currentIndex - 1
        : currentIndex === elements.length - 1
          ? 0
          : currentIndex + 1;

      event.preventDefault();
      elements[nextIndex].focus();
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  return (
    <div
      data-testid="modal-backdrop"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
      style={{
        position: 'fixed',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(0, 0, 0, 0.35)'
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        style={{ background: '#fff', padding: '1rem', minWidth: '320px' }}
      >
        <button type="button" aria-label="Close" onClick={onClose}>
          ×
        </button>
        <h2>{title}</h2>
        <div>{children}</div>
      </div>
    </div>
  );
}

