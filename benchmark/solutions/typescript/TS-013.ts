export type Context<T> = {
  data: T;
  metadata: Record<string, unknown>;
};

export type Middleware<T, U> = (
  ctx: Context<T>,
  next: () => Promise<Context<U>>
) => Promise<Context<U>>;

type UnknownContext = Context<unknown>;
type UnknownMiddleware = Middleware<unknown, unknown>;

export class Pipeline<TInput, TCurrent = TInput> {
  private readonly middlewares: UnknownMiddleware[];

  constructor(middlewares: UnknownMiddleware[] = []) {
    this.middlewares = middlewares;
  }

  use<U>(middleware: Middleware<TCurrent, U>): Pipeline<TInput, U> {
    return new Pipeline<TInput, U>([
      ...this.middlewares,
      middleware as UnknownMiddleware
    ]);
  }

  async execute(initialContext: Context<TInput>): Promise<Context<TCurrent>> {
    const dispatch = async (index: number, context: UnknownContext): Promise<UnknownContext> => {
      const middleware = this.middlewares[index];

      if (!middleware) {
        return context;
      }

      return middleware(context, async () => dispatch(index + 1, context));
    };

    return dispatch(0, initialContext as UnknownContext) as Promise<Context<TCurrent>>;
  }
}

export const loggerMiddleware = async <T>(
  ctx: Context<T>,
  next: () => Promise<Context<T>>
): Promise<Context<T>> => {
  const existingSteps = Array.isArray(ctx.metadata.steps)
    ? (ctx.metadata.steps as string[])
    : [];

  ctx.metadata.steps = [...existingSteps, 'logger'];
  ctx.metadata.logged = true;
  return next();
};

export const transformMiddleware: Middleware<string, string> = async (
  ctx,
  next
) => {
  const existingSteps = Array.isArray(ctx.metadata.steps)
    ? (ctx.metadata.steps as string[])
    : [];

  const transformedContext: Context<string> = {
    data: ctx.data.toUpperCase(),
    metadata: {
      ...ctx.metadata,
      transformed: true,
      steps: [...existingSteps, 'transform']
    }
  };

  const mutableContext = ctx as unknown as Context<string>;
  mutableContext.data = transformedContext.data;
  mutableContext.metadata = transformedContext.metadata;

  return next();
};
