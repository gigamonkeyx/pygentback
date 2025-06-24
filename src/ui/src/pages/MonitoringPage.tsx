import React from 'react';
import { Activity } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export const MonitoringPage: React.FC = () => {
  return (
    <div className="p-6">
      <div className="flex items-center space-x-3 mb-6">
        <Activity className="h-8 w-8 text-orange-500" />
        <div>
          <h1 className="text-3xl font-bold">System Monitoring</h1>
          <p className="text-muted-foreground">
            Real-time system performance and health monitoring
          </p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Monitoring Dashboard</CardTitle>
          <CardDescription>
            System metrics, alerts, and performance trends
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-96 flex items-center justify-center border-2 border-dashed border-muted-foreground/25 rounded-lg">
            <div className="text-center text-muted-foreground">
              <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">Monitoring Interface</p>
              <p className="text-sm">Coming soon...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
