"use client";

import { useEffect } from 'react';
import { useWebSocket } from '@/hooks/use-websocket';
import { useGame } from '@/hooks/use-game';
import { GameBoard } from '@/components/poker/game-board';
import { PlayerInfo } from '@/components/poker/player-info';
import { ActionPanel } from '@/components/poker/action-panel';
import { StatsPanel } from '@/components/poker/stats-panel';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';

export default function PokerGame() {
  const { stats, connected } = useWebSocket('ws://localhost:8000/ws');
  const { gameState, loading, error, fetchGameStatus, makeAction, resetGame, aiAction } = useGame();

  // Charger l'état initial
  useEffect(() => {
    console.log('🚀 PokerGame mounted, fetching initial state...');
    fetchGameStatus();
  }, [fetchGameStatus]);

  // Affichage d'erreur
  if (error) {
    return (
      <div className="container mx-auto p-4">
        <Alert className="border-red-200 bg-red-50">
          <AlertDescription className="text-red-600">
            <strong>Erreur:</strong> {error}
          </AlertDescription>
        </Alert>
        <div className="mt-4 text-center">
          <Button onClick={() => fetchGameStatus()}>
            🔄 Réessayer
          </Button>
        </div>
        
        {/* Debug info */}
        <div className="mt-4 p-4 bg-gray-100 rounded text-sm">
          <h3 className="font-bold">Debug Info:</h3>
          <p>API Backend: http://localhost:8000</p>
          <p>WebSocket: {connected ? '✅ Connecté' : '❌ Déconnecté'}</p>
          <p>Frontend API: /api/game/status</p>
        </div>
      </div>
    );
  }

  if (!gameState) {
    return (
      <div className="container mx-auto p-4">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Chargement du jeu...</p>
          
          {/* Debug info pendant le chargement */}
          <div className="mt-4 text-sm text-gray-500">
            <p>Connexion WebSocket: {connected ? '✅' : '❌'}</p>
            <p>Tentative de connexion à l'API...</p>
          </div>
        </div>
      </div>
    );
  }

  const getWinnerMessage = () => {
    if (!gameState.done) return null;
    
    switch (gameState.winner) {
      case 'player':
        return '🎉 L\'agent IA remporte la partie !';
      case 'opponent':
        return '😞 L\'adversaire remporte la partie';
      default:
        return '🤝 Match nul';
    }
  };

  return (
    <div className="container mx-auto p-4 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          🤖 DQN Poker Agent
        </h1>
        <p className="text-gray-600 mt-2">
          Intelligence Artificielle jouant au Texas Hold'em No Limit
        </p>
      </div>

      {/* Statistiques temps réel */}
      <StatsPanel stats={stats} connected={connected} />

      {/* Table de jeu */}
      <GameBoard 
        pot={gameState.pot}
        communityCards={gameState.community_cards}
        bettingRound={gameState.betting_round}
      />

      {/* Joueurs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PlayerInfo
          name="Agent DQN"
          chips={gameState.player_chips}
          bet={gameState.player_bet}
          cards={gameState.player_cards}
          isAI={true}
          maxChips={1000}
        />
        
        <PlayerInfo
          name="Adversaire"
          chips={gameState.opponent_chips}
          bet={gameState.opponent_bet}
          maxChips={1000}
        />
      </div>

      {/* Résultat de la partie */}
      {gameState.done && (
        <Alert className="border-yellow-200 bg-yellow-50">
          <AlertDescription className="text-center text-lg font-semibold">
            {getWinnerMessage()}
          </AlertDescription>
        </Alert>
      )}

      {/* Actions */}
      <ActionPanel
        onAction={makeAction}
        onAIAction={aiAction}
        onReset={resetGame}
        loading={loading}
        disabled={gameState.done}
      />

      {/* Footer */}
      <div className="text-center text-sm text-gray-500">
        <p>Projet DQN Texas Hold'em - Deep Q-Network avec exploration softmax</p>
      </div>
    </div>
  );
}