import React, { useEffect, useState } from 'react';

export default function DataLoader({ fetchData }) {
  const [data, setData] = useState('');

  useEffect(() => {
    setData(fetchData());
  }, [fetchData]);

  return <div>{data}</div>;
}

