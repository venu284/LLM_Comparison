import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import DragDropList from '../../solutions/frontend/FE-012';

const items = [
  { id: 'a', text: 'Alpha' },
  { id: 'b', text: 'Beta' },
  { id: 'c', text: 'Gamma' }
];

const createDataTransfer = () => {
  const store = {};

  return {
    setData: jest.fn((type, value) => {
      store[type] = value;
    }),
    getData: jest.fn((type) => store[type]),
    dropEffect: 'move',
    effectAllowed: 'all'
  };
};

describe('FE-012: Drag and Drop List', () => {
  test('renders without crashing', () => {
    render(<DragDropList items={items} />);
  });

  test('all items are visible', () => {
    render(<DragDropList items={items} />);
    expect(screen.getByText('Alpha')).toBeInTheDocument();
    expect(screen.getByText('Beta')).toBeInTheDocument();
    expect(screen.getByText('Gamma')).toBeInTheDocument();
  });

  test('each item has draggable attribute', () => {
    render(<DragDropList items={items} />);
    expect(screen.getByTestId('item-a')).toHaveAttribute('draggable', 'true');
  });

  test('each item has the correct data-testid', () => {
    render(<DragDropList items={items} />);
    expect(screen.getByTestId('item-a')).toBeInTheDocument();
    expect(screen.getByTestId('item-b')).toBeInTheDocument();
    expect(screen.getByTestId('item-c')).toBeInTheDocument();
  });

  test('items render in the provided order', () => {
    render(<DragDropList items={items} />);
    const renderedItems = screen.getAllByRole('listitem');
    expect(renderedItems[0]).toHaveTextContent('Alpha');
    expect(renderedItems[1]).toHaveTextContent('Beta');
    expect(renderedItems[2]).toHaveTextContent('Gamma');
  });

  test('drag start sets drag data', () => {
    render(<DragDropList items={items} />);
    const dataTransfer = createDataTransfer();
    fireEvent.dragStart(screen.getByTestId('item-a'), { dataTransfer });
    expect(dataTransfer.setData).toHaveBeenCalledWith('text/plain', '0');
  });

  test('drop reorders items', () => {
    render(<DragDropList items={items} />);
    const dataTransfer = createDataTransfer();
    fireEvent.dragStart(screen.getByTestId('item-a'), { dataTransfer });
    fireEvent.dragOver(screen.getByTestId('item-c'), { dataTransfer });
    fireEvent.drop(screen.getByTestId('item-c'), { dataTransfer });
    const renderedItems = screen.getAllByRole('listitem');
    expect(renderedItems[2]).toHaveTextContent('Alpha');
  });

  test('onReorder is called with the new array', () => {
    const onReorder = jest.fn();
    render(<DragDropList items={items} onReorder={onReorder} />);
    const dataTransfer = createDataTransfer();
    fireEvent.dragStart(screen.getByTestId('item-a'), { dataTransfer });
    fireEvent.dragOver(screen.getByTestId('item-c'), { dataTransfer });
    fireEvent.drop(screen.getByTestId('item-c'), { dataTransfer });
    expect(onReorder).toHaveBeenCalledWith([
      { id: 'b', text: 'Beta' },
      { id: 'c', text: 'Gamma' },
      { id: 'a', text: 'Alpha' }
    ]);
  });

  test('handles single item without crashing', () => {
    render(<DragDropList items={[items[0]]} />);
    expect(screen.getAllByRole('listitem')).toHaveLength(1);
  });

  test('handles empty items array', () => {
    render(<DragDropList items={[]} />);
    expect(screen.queryByRole('listitem')).not.toBeInTheDocument();
  });
});

