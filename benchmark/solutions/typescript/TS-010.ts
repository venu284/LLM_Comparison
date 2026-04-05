export interface HttpRequest {
  url: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  headers?: Record<string, string>;
  body?: unknown;
  timeout?: number;
}

export class RequestBuilder<HasUrl extends boolean = false, HasMethod extends boolean = false> {
  private readonly request: Partial<HttpRequest>;

  constructor(request: Partial<HttpRequest> = {}) {
    this.request = request;
  }

  setUrl(url: string): RequestBuilder<true, HasMethod> {
    return new RequestBuilder<true, HasMethod>({
      ...this.request,
      url
    });
  }

  setMethod(method: HttpRequest['method']): RequestBuilder<HasUrl, true> {
    return new RequestBuilder<HasUrl, true>({
      ...this.request,
      method
    });
  }

  setHeader(key: string, value: string): RequestBuilder<HasUrl, HasMethod> {
    return new RequestBuilder<HasUrl, HasMethod>({
      ...this.request,
      headers: {
        ...this.request.headers,
        [key]: value
      }
    });
  }

  setBody(body: unknown): RequestBuilder<HasUrl, HasMethod> {
    return new RequestBuilder<HasUrl, HasMethod>({
      ...this.request,
      body
    });
  }

  setTimeout(timeout: number): RequestBuilder<HasUrl, HasMethod> {
    return new RequestBuilder<HasUrl, HasMethod>({
      ...this.request,
      timeout
    });
  }

  build(this: RequestBuilder<true, true>): HttpRequest {
    const { url, method, headers, body, timeout } = this.request;

    return {
      url,
      method,
      headers,
      body,
      timeout
    } as HttpRequest;
  }
}

