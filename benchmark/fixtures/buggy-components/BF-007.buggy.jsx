import React, { useEffect, useState } from 'react';

export default function DataFetcher({ fetchData }) {
  const [data, setData] = useState('Loading...');

  useEffect(() => {
    fetchData().then((result) => {
      setData(result);
    });
  }, [fetchData]);

  return <div>{data}</div>;
}

