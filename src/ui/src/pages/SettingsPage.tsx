import React from 'react';
import { Settings } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export const SettingsPage: React.FC = () => {
  return (
    <div className="p-6">
      <div className="flex items-center space-x-3 mb-6">
        <Settings className="h-8 w-8 text-gray-500" />
        <div>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground">
            System configuration and user preferences
          </p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Configuration Panel</CardTitle>
          <CardDescription>
            Manage system settings and user preferences
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-96 flex items-center justify-center border-2 border-dashed border-muted-foreground/25 rounded-lg">
            <div className="text-center text-muted-foreground">
              <Settings className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">Settings Interface</p>
              <p className="text-sm">Coming soon...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
