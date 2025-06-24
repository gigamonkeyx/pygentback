import React from 'react';
import { Dna } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export const EvolutionPage: React.FC = () => {
  return (
    <div className="p-6">
      <div className="flex items-center space-x-3 mb-6">
        <Dna className="h-8 w-8 text-green-500" />
        <div>
          <h1 className="text-3xl font-bold">Recipe Evolution</h1>
          <p className="text-muted-foreground">
            AI-guided recipe optimization and genetic algorithms
          </p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Evolution Dashboard</CardTitle>
          <CardDescription>
            Monitor recipe evolution progress and fitness metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-96 flex items-center justify-center border-2 border-dashed border-muted-foreground/25 rounded-lg">
            <div className="text-center text-muted-foreground">
              <Dna className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">Evolution Interface</p>
              <p className="text-sm">Coming soon...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
