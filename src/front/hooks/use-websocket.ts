import { useEffect, useState } from 'react';

interface GameStats {
  episode: number;
  total_reward: number;
  win_rate: number;
  epsilon: number;
  memory_size: number;
  is_training: boolean;
}

export function useWebSocket(url: string) {
  const [stats, setStats] = useState<GameStats | null>(null);
  const [gameData, setGameData] = useState<any>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onopen = () => {
      setConnected(true);
      console.log('WebSocket connecté');
    };
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      switch (message.type) {
        case 'stats_update':
          setStats(message.data);
          break;
        case 'game_update':
          setGameData(message.data);
          break;
      }
    };
    
    ws.onclose = () => {
      setConnected(false);
      console.log('WebSocket déconnecté');
    };
    
    ws.onerror = (error) => {
      console.error('Erreur WebSocket:', error);
    };
    
    return () => ws.close();
  }, [url]);

  return { stats, gameData, connected };
}