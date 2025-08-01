"use client";

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

interface PlayerInfoProps {
  name: string;
  chips: number;
  bet: number;
  cards?: string[];
  isAI?: boolean;
  maxChips?: number;
}

export function PlayerInfo({ 
  name, 
  chips, 
  bet, 
  cards, 
  isAI = false, 
  maxChips = 1000 
}: PlayerInfoProps) {
  const chipPercentage = (chips / maxChips) * 100;
  
  return (
    <Card className={isAI ? "border-blue-200 bg-blue-50" : "border-red-200 bg-red-50"}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span>{isAI ? "🤖" : "🎲"}</span>
          {name}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Jetons */}
        <div>
          <div className="flex justify-between text-sm">
            <span>Jetons</span>
            <span>{chips}</span>
          </div>
          <Progress value={chipPercentage} className="mt-1" />
        </div>
        
        {/* Mise actuelle */}
        <div>
          <div className="text-sm text-gray-600">Mise actuelle</div>
          <div className="font-bold text-lg">{bet} jetons</div>
        </div>
        
        {/* Cartes (si visible) */}
        {cards && (
          <div>
            <div className="text-sm text-gray-600 mb-2">Cartes</div>
            <div className="flex gap-1">
              {cards.map((card, i) => (
                <Badge 
                  key={i} 
                  variant="outline"
                  className="text-sm font-mono"
                >
                  {card}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}