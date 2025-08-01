export async function POST() {
  try {
    const response = await fetch('http://localhost:8000/game/ai_action', {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    console.error('API Error:', error);
    return Response.json({ error: 'Failed to make AI action' }, { status: 500 });
  }
}