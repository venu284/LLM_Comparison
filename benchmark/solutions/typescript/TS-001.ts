export interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

export interface CreateUserInput {
  name: string;
  email: string;
}

export function createUser(input: CreateUserInput): User {
  return {
    id: Math.random().toString(36).substr(2, 9),
    name: input.name,
    email: input.email,
    createdAt: new Date()
  };
}

