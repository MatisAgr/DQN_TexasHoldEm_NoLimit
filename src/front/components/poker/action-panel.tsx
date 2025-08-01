"use client";

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface ActionPanelProps {
  onAction: (actionId: number) => void;
  onAIAction: () => void;
  onReset: () => void;
  loading: boolean;
  disabled: boolean;
}

const ACTIONS = [
  { id: 0, name: 'FOLD', variant: 'destructive', description: 'Se coucher' },
  { id: 1, name: 'CALL', variant: 'secondary', description: 'Suivre' },
  { id: 2, name: 'RAISE SMALL', variant: 'default', description: '+50 jetons' },
  { id: 3, name: 'RAISE BIG', variant: 'default', description: '+100 jetons' },
  { id: 4, name: 'ALL IN', variant: 'destructive', description: 'Tapis' }
] as const;

export function ActionPanel({ 
  onAction, 
  onAIAction, 
  onReset, 
  loading, 
  disabled 
}: ActionPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>⚡ Actions disponibles</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {!disabled && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            {ACTIONS.map((action) => (
              <div key={action.id} className="text-center">
                <Button
                  variant={action.variant as any}
                  onClick={() => onAction(action.id)}
                  disabled={loading}
                  className="w-full mb-1"
                >
                  {action.name}
                </Button>
                <div className="text-xs text-gray-500">
                  {action.description}
                </div>
              </div>
            ))}
          </div>
        )}
        
        <div className="flex gap-2 pt-2 border-t">
          <Button 
            variant="outline" 
            onClick={onAIAction} 
            disabled={loading || disabled}
            className="flex-1"
          >
            🤖 IA Joue
          </Button>
          <Button 
            variant="outline" 
            onClick={onReset}
            className="flex-1"
          >
            🔄 Nouvelle partie
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}