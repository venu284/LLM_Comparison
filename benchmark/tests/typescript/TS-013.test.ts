import {
  Context,
  Pipeline,
  loggerMiddleware,
  transformMiddleware
} from '../../solutions/typescript/TS-013';
import { readSolution } from './helpers';

describe('TS-013: Middleware Pipeline Types', () => {
  test('Context stores data and metadata', () => {
    const context: Context<string> = {
      data: 'hello',
      metadata: {}
    };

    expect(context.data).toBe('hello');
  });

  test('single middleware pipeline works', async () => {
    const pipeline = new Pipeline<string>().use(transformMiddleware);
    const result = await pipeline.execute({
      data: 'hello',
      metadata: {}
    });

    expect(result.data).toBe('HELLO');
  });

  test('multiple middleware chain in order', async () => {
    const pipeline = new Pipeline<string>()
      .use(loggerMiddleware)
      .use(transformMiddleware);

    const result = await pipeline.execute({
      data: 'hello',
      metadata: {}
    });

    expect(result.metadata.steps).toEqual(['logger', 'transform']);
  });

  test('transformed data reaches downstream middleware', async () => {
    let observedLength = 0;
    const toLength = async (
      ctx: Context<string>,
      next: () => Promise<Context<number>>
    ): Promise<Context<number>> => {
      const nextContext = ctx as unknown as Context<number>;
      nextContext.data = ctx.data.length;
      nextContext.metadata = {
        ...ctx.metadata,
        transformedToLength: true
      };

      return next();
    };
    const observeLength = async (
      ctx: Context<number>,
      next: () => Promise<Context<number>>
    ): Promise<Context<number>> => {
      observedLength = ctx.data;
      return next();
    };
    const pipeline: Pipeline<string, number> = new Pipeline<string>()
      .use(toLength)
      .use(observeLength);
    const result = await pipeline.execute({
      data: 'hello',
      metadata: {}
    });

    expect(observedLength).toBe(5);
    expect(result.data).toBe(5);
    expect(result.metadata.transformedToLength).toBe(true);
  });

  test('execute runs middleware in order around next', async () => {
    const events: string[] = [];
    const first = async (
      ctx: Context<string>,
      next: () => Promise<Context<string>>
    ): Promise<Context<string>> => {
      events.push(`first:${ctx.data}`);
      const result = await next();
      events.push(`first:done:${result.data}`);
      return result;
    };
    const second = async (
      ctx: Context<string>,
      next: () => Promise<Context<string>>
    ): Promise<Context<string>> => {
      events.push(`second:${ctx.data}`);
      const result = await next();
      events.push(`second:done:${result.data}`);
      return result;
    };
    const pipeline = new Pipeline<string>().use(first).use(second);
    const result = await pipeline.execute({
      data: 'ordered',
      metadata: {}
    });

    expect(result.data).toBe('ordered');
    expect(events).toEqual([
      'first:ordered',
      'second:ordered',
      'second:done:ordered',
      'first:done:ordered'
    ]);
  });

  test('logger middleware does not change the data value', async () => {
    const pipeline = new Pipeline<number>().use(loggerMiddleware);
    const result = await pipeline.execute({
      data: 123,
      metadata: {}
    });

    expect(result.data).toBe(123);
  });

  test('transform middleware uppercases string data', async () => {
    let seenByNext = '';
    const result = await transformMiddleware(
      { data: 'world', metadata: {} },
      async () => {
        seenByNext = 'WORLD';
        return { data: 'WORLD', metadata: { forwarded: true } };
      }
    );

    expect(result.data).toBe('WORLD');
    expect(seenByNext).toBe('WORLD');
    expect(result.metadata.forwarded).toBe(true);
  });

  test('pipeline propagates middleware errors', async () => {
    const explodingPipeline = new Pipeline<string>().use(async () => {
      throw new Error('Pipeline failed');
    });

    await expect(
      explodingPipeline.execute({
        data: 'hello',
        metadata: {}
      })
    ).rejects.toThrow('Pipeline failed');
  });

  test('solution uses no any types', () => {
    const content = readSolution('TS-013');
    expect(content).not.toMatch(/:\s*any\b/);
  });
});
