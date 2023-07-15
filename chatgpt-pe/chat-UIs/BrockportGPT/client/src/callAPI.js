import fetch from 'node-fetch';

async function retrieveInformation(message) {
  try {
    const response = await fetch('http://localhost:3080', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
      }),
    });

    const data = await response.json();

    return data.message;
  } catch (error) {
    console.error('Error retrieving information:', error);
    return null;
  }
}

export default retrieveInformation;
