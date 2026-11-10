/** A transport failure while reading response bytes, distinct from parsing errors. */
export class ResponseReadError extends Error {
  constructor(cause: TypeError) {
    super('Failed to read response bytes', { cause })
  }
}

/** Incremental SSE line parser. */
class SSEParser {
  private buffer = ''
  private data: string[] = []

  push(text: string, end: boolean): string[] {
    this.buffer += text
    const result: string[] = []
    let newline = this.buffer.search(/[\r\n]/)
    while (newline !== -1) {
      const terminator = this.buffer[newline]
      if (terminator === '\r' && newline === this.buffer.length - 1 && !end) break
      const skip = terminator === '\r' && this.buffer[newline + 1] === '\n' ? 2 : 1
      const line = this.buffer.slice(0, newline)
      this.buffer = this.buffer.slice(newline + skip)
      this.line(line, result)
      newline = this.buffer.search(/[\r\n]/)
    }
    // EOF does not dispatch a pending event; the stream requires a blank line.
    return result
  }

  private line(line: string, result: string[]): void {
    if (line === '') {
      if (this.data.length) result.push(this.data.join('\n'))
      this.data = []
    } else if (line === 'data') {
      this.data.push('')
    } else if (line.startsWith('data:')) {
      this.data.push(line.slice(5).replace(/^ /, ''))
    }
  }
}

/** Parse SSE framing independently of byte, UTF-8 and newline boundaries. */
export async function* readSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  const parser = new SSEParser()
  let complete = false
  try {
    while (true) {
      let chunk: ReadableStreamReadResult<Uint8Array>
      try { chunk = await reader.read() } catch (error) {
        if (error instanceof TypeError) throw new ResponseReadError(error)
        throw error
      }
      complete = chunk.done
      const text = decoder.decode(chunk.value, { stream: !chunk.done })
      yield* parser.push(text, chunk.done)
      if (chunk.done) break
    }
  } finally {
    if (!complete) {
      try { void reader.cancel().catch(() => undefined) } catch { /* custom transport cleanup may throw */ }
    }
    reader.releaseLock()
  }
}
