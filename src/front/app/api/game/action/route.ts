export async function POST(request: Request) {
  try {
    const { action_id } = await request.json();
    
    const response = await fetch(`http://localhost:8000/game/action/${action_id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    console.error('API Error:', error);
    return Response.json({ error: 'Failed to make action' }, { status: 500 });
  }
}