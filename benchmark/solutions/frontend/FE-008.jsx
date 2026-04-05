import React, { useState } from 'react';

const TOTAL_STARS = 5;

export default function StarRating() {
  const [rating, setRating] = useState(0);
  const [hoveredRating, setHoveredRating] = useState(0);
  const visibleRating = hoveredRating || rating;

  return (
    <div>
      <div>
        {Array.from({ length: TOTAL_STARS }, (_, index) => {
          const starNumber = index + 1;
          const isFilled = starNumber <= visibleRating;

          return (
            <button
              key={starNumber}
              type="button"
              data-testid={`star-${starNumber}`}
              aria-label={`Rate ${starNumber}`}
              onClick={() => setRating(starNumber)}
              onMouseEnter={() => setHoveredRating(starNumber)}
              onMouseLeave={() => setHoveredRating(0)}
            >
              {isFilled ? '★' : '☆'}
            </button>
          );
        })}
      </div>
      <p>Rating: {rating}/5</p>
    </div>
  );
}

