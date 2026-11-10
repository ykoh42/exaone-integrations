/** A transport failure while reading response bytes, distinct from parsing errors. */
export class ResponseReadError extends Error {
  constructor(cause: TypeError) {
    super('Failed to read response bytes', { cause })
  }
}

/** Incremental SSE line parser. */
class SSEParser {
  private readonly fragments: string[] = []
  private data: string[] = []
  private afterCR = false

  private startOffset(text: string): number {
    if (!this.afterCR || !text.length) return 0
    this.afterCR = false
    return text[0] === '\n' ? 1 : 0
  }

  push(text: string): string[] {
    const result: string[] = []
    let start = this.startOffset(text)
    for (let index = start; index < text.length; index++) {
      const char = text[index]
      if (char !== '\r' && char !== '\n') continue
      if (index > start) this.fragments.push(text.slice(start, index))
      this.line(this.fragments.join(''), result)
      this.fragments.length = 0
      if (char === '\r') {
        if (text[index + 1] === '\n') index++
        else if (index + 1 === text.length) this.afterCR = true
      }
      start = index + 1
    }
    if (start < text.length) this.fragments.push(text.slice(start))
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
      yield* parser.push(text)
      if (chunk.done) break
    }
  } finally {
    if (!complete) {
      try { void reader.cancel().catch(() => undefined) } catch { /* custom transport cleanup may throw */ }
    }
    reader.releaseLock()
  }
}
