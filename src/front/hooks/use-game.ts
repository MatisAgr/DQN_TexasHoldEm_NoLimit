import { useState, useCallback } from 'react';

interface GameState {
  pot: number;
  player_chips: number;
  opponent_chips: number;
  player_bet: number;
  opponent_bet: number;
  betting_round: number;
  community_cards: string[];
  player_cards: string[];
  done: boolean;
  winner?: string;
}

export function useGame() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchGameStatus = useCallback(async () => {
    console.log('🔄 Fetching game status...');
    setError(null);
    
    try {
      const response = await fetch('/api/game/status');
      console.log('📡 Response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('📦 Game data received:', data);
      
      if (data.game_state) {
        setGameState(data.game_state);
        console.log('✅ Game state updated');
      } else {
        console.error('❌ No game_state in response');
        setError('Invalid response format');
      }
      
      return data;
    } catch (error) {
      console.error('❌ Error fetching game status:', error);
      setError(error instanceof Error ? error.message : 'Unknown error');
      throw error;
    }
  }, []);

  const makeAction = useCallback(async (actionId: number) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/game/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action_id: actionId })
      });
      
      const result = await response.json();
      await fetchGameStatus();
      return result;
    } catch (error) {
      console.error('Error making action:', error);
      setError(error instanceof Error ? error.message : 'Action failed');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchGameStatus]);

  const resetGame = useCallback(async () => {
    setError(null);
    
    try {
      await fetch('/api/game/reset', { method: 'POST' });
      await fetchGameStatus();
    } catch (error) {
      console.error('Error resetting game:', error);
      setError(error instanceof Error ? error.message : 'Reset failed');
      throw error;
    }
  }, [fetchGameStatus]);

  const aiAction = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      await fetch('/api/game/ai-action', { method: 'POST' });
      await fetchGameStatus();
    } catch (error) {
      console.error('Error with AI action:', error);
      setError(error instanceof Error ? error.message : 'AI action failed');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchGameStatus]);

  return {
    gameState,
    loading,
    error,
    fetchGameStatus,
    makeAction,
    resetGame,
    aiAction
  };
}