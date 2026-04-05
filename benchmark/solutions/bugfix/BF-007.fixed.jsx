import React, { useEffect, useState } from 'react';

async function defaultFetchData() {
  return 'Loaded';
}

export default function DataFetcher({ fetchData = defaultFetchData }) {
  const [data, setData] = useState('Loading...');

  useEffect(() => {
    const controller = new AbortController();

    fetchData(controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) {
          setData(result);
        }
      })
      .catch((error) => {
        const isAbortError =
          error instanceof DOMException && error.name === 'AbortError';

        if (!controller.signal.aborted && !isAbortError) {
          setData('Error');
        }
      });

    return () => {
      controller.abort();
    };
  }, [fetchData]);

  return <div>{data}</div>;
}

