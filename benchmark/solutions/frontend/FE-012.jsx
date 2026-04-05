import React, { useState } from 'react';

export default function DragDropList({ items = [], onReorder = () => {} }) {
  const [listItems, setListItems] = useState(items);
  const [draggedIndex, setDraggedIndex] = useState(null);
  const [dropIndex, setDropIndex] = useState(null);

  const reorderItems = (fromIndex, toIndex) => {
    if (fromIndex === toIndex || fromIndex == null || toIndex == null) {
      return;
    }

    const nextItems = [...listItems];
    const [movedItem] = nextItems.splice(fromIndex, 1);
    nextItems.splice(toIndex, 0, movedItem);
    setListItems(nextItems);
    onReorder(nextItems);
  };

  return (
    <ul>
      {listItems.map((item, index) => (
        <li
          key={item.id}
          data-testid={`item-${item.id}`}
          draggable="true"
          className={dropIndex === index ? 'drop-target' : ''}
          onDragStart={(event) => {
            event.dataTransfer.setData('text/plain', String(index));
            setDraggedIndex(index);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            setDropIndex(index);
          }}
          onDragEnd={() => {
            setDraggedIndex(null);
            setDropIndex(null);
          }}
          onDrop={(event) => {
            event.preventDefault();
            const fromIndex =
              draggedIndex != null ? draggedIndex : Number(event.dataTransfer.getData('text/plain'));

            reorderItems(fromIndex, index);
            setDraggedIndex(null);
            setDropIndex(null);
          }}
        >
          {item.text}
        </li>
      ))}
    </ul>
  );
}

