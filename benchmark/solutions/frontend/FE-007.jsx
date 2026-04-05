import React, { useState } from 'react';

export default function Accordion({ sections = [] }) {
  const [openIndex, setOpenIndex] = useState(null);

  return (
    <div>
      {sections.map((section, index) => (
        <div key={section.title} data-testid={`section-${index}`}>
          <button
            type="button"
            onClick={() => setOpenIndex((currentIndex) => (currentIndex === index ? null : index))}
          >
            {section.title}
          </button>
          {openIndex === index ? <p>{section.content}</p> : null}
        </div>
      ))}
    </div>
  );
}

