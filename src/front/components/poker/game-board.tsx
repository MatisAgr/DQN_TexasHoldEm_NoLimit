"use client";

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface GameBoardProps {
  pot: number;
  communityCards: string[];
  bettingRound: number;
}

export function GameBoard({ pot, communityCards, bettingRound }: GameBoardProps) {
  const roundNames = ['Pre-flop', 'Flop', 'Turn', 'River'];
  
  return (
    <Card className="bg-green-50 border-green-200">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>🃏 Table de Poker</span>
          <Badge variant="outline">{roundNames[bettingRound]}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-center space-y-4">
          {/* Pot */}
          <div>
            <div className="text-sm text-gray-600">Pot</div>
            <div className="text-3xl font-bold text-green-600">
              💰 {pot} jetons
            </div>
          </div>
          
          {/* Cartes communes */}
          {communityCards.length > 0 && (
            <div>
              <div className="text-sm text-gray-600 mb-2">Board</div>
              <div className="flex justify-center gap-2">
                {communityCards.map((card, i) => (
                  <Badge 
                    key={i} 
                    variant="secondary" 
                    className="text-lg p-3 bg-white border-2"
                  >
                    {card}
                  </Badge>
                ))}
                {/* Cartes à venir */}
                {Array.from({ length: 5 - communityCards.length }).map((_, i) => (
                  <div 
                    key={`empty-${i}`}
                    className="w-12 h-16 bg-gray-200 border-2 border-dashed border-gray-400 rounded flex items-center justify-center"
                  >
                    <span className="text-gray-400">?</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}