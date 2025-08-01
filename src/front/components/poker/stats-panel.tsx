"use client";

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

interface StatsPanelProps {
  stats: {
    episode: number;
    total_reward: number;
    win_rate: number;
    epsilon: number;
    memory_size: number;
    is_training: boolean;
  } | null;
  connected: boolean;
}

export function StatsPanel({ stats, connected }: StatsPanelProps) {
  if (!stats) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>📊 Statistiques Agent</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-500">
            En attente des données...
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>📊 Statistiques Agent</span>
          <div className="flex gap-2">
            <Badge variant={connected ? "default" : "destructive"}>
              {connected ? "🟢 Live" : "🔴 Déco"}
            </Badge>
            {stats.is_training && (
              <Badge variant="secondary">🎯 Training</Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {stats.episode}
            </div>
            <div className="text-sm text-gray-500">Épisode</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {stats.total_reward.toFixed(1)}
            </div>
            <div className="text-sm text-gray-500">Récompense</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {stats.win_rate.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-500">Taux victoire</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {stats.memory_size}
            </div>
            <div className="text-sm text-gray-500">Mémoire</div>
          </div>
        </div>
        
        <div className="mt-4 space-y-2">
          <div>
            <div className="flex justify-between text-sm">
              <span>Epsilon (exploration)</span>
              <span>{stats.epsilon.toFixed(4)}</span>
            </div>
            <Progress 
              value={stats.epsilon * 100} 
              className="mt-1"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}