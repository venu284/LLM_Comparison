import fs from 'fs';
import path from 'path';

export function readSolution(taskId: string): string {
  return fs.readFileSync(
    path.join(__dirname, '../../solutions/typescript', `${taskId}.ts`),
    'utf8'
  );
}

