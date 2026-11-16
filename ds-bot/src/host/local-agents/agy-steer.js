// Lossless step-boundary steering for Antigravity (`agy acp`) via native
// Lifecycle Hooks (`PreInvocation`, `PostInvocation`, `Stop`).
//
// On POSIX, DS Bot listens on a private 0600 Unix domain socket inside a 0700
// directory (`DSBOT_STEER_DIR/steer.sock`, no TCP port) and touches a per-session
// `<conversationId>.steer` marker file only while a steer is queued. `hooks.json`
// checks that marker in pure `/bin/sh`, so 99.9% of tool/model steps exit in
// <0.3ms without spawning `node` at all (both in standalone terminal `agy` and
// inside DS Bot when no steer is waiting).
import { createServer, request as httpRequest } from 'node:http'
import { randomBytes } from 'node:crypto'
import { chmodSync, closeSync, existsSync, fstatSync, mkdirSync, mkdtempSync, openSync, readFileSync, readSync, rmSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

const TERMINAL_STOP_REASONS = new Set([
  'USER_CANCELED',
  'ERROR',
  'MAX_INVOCATIONS',
  'MAX_FORCED_INVOCATIONS',
  'MAX_TOKEN_BUDGET_EXCEEDED',
  'TERMINAL_CUSTOM_HOOK',
])

export const STEER_HOOK_SCRIPT = `import { readFileSync } from 'node:fs'
import { request as httpRequest } from 'node:http'

const sock = process.env.DSBOT_STEER_SOCK
const url = process.env.DSBOT_STEER_URL
const token = process.env.DSBOT_STEER_TOKEN
let wrote = false
const finish = (text) => {
  if (wrote) return
  wrote = true
  process.stdout.write(text)
}

if ((!sock && !url) || !token) {
  finish('{}')
  process.exit(0)
}

let input = {}
try {
  input = JSON.parse(readFileSync(0, 'utf8'))
} catch {
  finish('{}')
  process.exit(0)
}

const body = JSON.stringify({
  conversationId: input.conversationId,
  hookEventName: input.hookEventName,
  terminationReason: input.terminationReason,
  fullyIdle: input.fullyIdle,
  transcriptPath: input.transcriptPath,
})

let target
try {
  if (sock) target = { socketPath: sock, path: '/steer' }
  else {
    const parsed = new URL(url)
    target = { hostname: parsed.hostname, port: parsed.port, path: parsed.pathname }
  }
} catch {
  finish('{}')
  process.exit(0)
}

const req = httpRequest({
  ...target,
  method: 'POST',
  headers: {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(body),
    authorization: \`Bearer \${token}\`,
  },
  timeout: 3_000,
}, (res) => {
  let out = ''
  res.setEncoding('utf8')
  res.on('data', (chunk) => { out += chunk })
  res.on('end', () => {
    const trimmed = res.statusCode === 200 ? out.trim() : ''
    finish(trimmed !== '' ? trimmed : '{}')
  })
})
req.on('timeout', () => req.destroy(new Error('timeout')))
req.on('error', () => finish('{}'))
req.end(body)
`

function writeIfChanged(filePath, content) {
  try {
    if (readFileSync(filePath, 'utf8') === content) return
  } catch { /* missing file is written below */ }
  writeFileSync(filePath, content, 'utf8')
}

const safeMarkerName = id => (typeof id === 'string' && /^[\w.-]{1,128}$/.test(id) ? `${id}.steer` : undefined)

export function ensureAgySteerPlugin({ homeDir = homedir(), nodePath = process.execPath, platform = process.platform } = {}) {
  try {
    const dir = join(homeDir, '.gemini', 'config', 'plugins', 'dsbot-steer')
    mkdirSync(dir, { recursive: true })
    const quotedNode = `'${nodePath.replaceAll("'", "'\\''")}'`
    const stepCommand = platform === 'win32'
      ? `"${nodePath}" ./steer-hook.mjs`
      : `if [ -z "$DSBOT_STEER_DIR" ] || { [ -n "$ANTIGRAVITY_CONVERSATION_ID" ] && [ ! -f "$DSBOT_STEER_DIR/$ANTIGRAVITY_CONVERSATION_ID.steer" ]; }; then printf '{}'; else ${quotedNode} ./steer-hook.mjs; fi`
    const stopCommand = platform === 'win32'
      ? `"${nodePath}" ./steer-hook.mjs`
      : `if [ -z "$DSBOT_STEER_DIR" ]; then printf '{}'; else ${quotedNode} ./steer-hook.mjs; fi`
    const stepHandlers = [{ type: 'command', command: stepCommand, timeout: 5 }]
    const stopHandlers = [{ type: 'command', command: stopCommand, timeout: 5 }]
    const hooksJson = `${JSON.stringify({
      'dsbot-steer': {
        PreInvocation: stepHandlers,
        PostInvocation: stepHandlers,
        Stop: stopHandlers,
      },
    }, null, 2)}\n`
    const pluginJson = `${JSON.stringify({ name: 'dsbot-steer' }, null, 2)}\n`
    writeIfChanged(join(dir, 'plugin.json'), pluginJson)
    writeIfChanged(join(dir, 'hooks.json'), hooksJson)
    writeIfChanged(join(dir, 'steer-hook.mjs'), STEER_HOOK_SCRIPT)
    return dir
  } catch {
    return undefined
  }
}

export function postAgySteer({ socketPath, url, token }, payload) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload)
    const target = socketPath
      ? { socketPath, path: '/steer' }
      : (() => { const parsed = new URL(url); return { hostname: parsed.hostname, port: parsed.port, path: parsed.pathname } })()
    const req = httpRequest({
      ...target,
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'content-length': Buffer.byteLength(body),
        authorization: `Bearer ${token}`,
      },
      timeout: 3_000,
    }, (res) => {
      let raw = ''
      res.setEncoding('utf8')
      res.on('data', chunk => { raw += chunk })
      res.on('end', () => {
        let parsed = {}
        try { parsed = JSON.parse(raw) } catch {}
        resolve({ status: res.statusCode ?? 0, body: parsed })
      })
    })
    req.on('timeout', () => req.destroy(new Error('timeout')))
    req.on('error', reject)
    req.end(body)
  })
}

export class AgySteerServer {
  constructor({ drain, hasPending, onStopState, platform = process.platform } = {}) {
    this.token = randomBytes(32).toString('hex')
    this.drain = drain
    this.hasPending = hasPending
    this.onStopState = onStopState
    this.platform = platform
    this.dir = undefined
    this.socketPath = undefined
    this.url = undefined
    this.onExit = undefined
    this.server = createServer((request, response) => void this.serve(request, response).catch(() => {
      if (!response.headersSent) response.writeHead(500, { 'content-type': 'application/json' })
      response.end('{}')
    }))
  }

  async listen() {
    if (this.dir !== undefined) return this.env()
    // Keep Unix socket path short (<104 bytes SUN_PATH limit on macOS).
    const baseTmp = this.platform !== 'win32' && existsSync('/tmp') ? '/tmp' : tmpdir()
    const dir = mkdtempSync(join(baseTmp, 'dsb-steer-'))
    try { chmodSync(dir, 0o700) } catch {}
    this.dir = dir
    this.onExit = () => { try { rmSync(dir, { recursive: true, force: true }) } catch {} }
    process.once('exit', this.onExit)
    if (this.platform === 'win32') {
      await new Promise((resolve, reject) => this.server.listen(0, '127.0.0.1', () => resolve()).once('error', reject))
      this.server.unref?.()
      const { port } = this.server.address()
      this.url = `http://127.0.0.1:${port}/steer`
    } else {
      const sock = join(dir, 'steer.sock')
      await new Promise((resolve, reject) => this.server.listen(sock, () => resolve()).once('error', reject))
      try { chmodSync(sock, 0o600) } catch {}
      this.server.unref?.()
      this.socketPath = sock
    }
    return this.env()
  }

  env() {
    return {
      ...(this.dir ? { DSBOT_STEER_DIR: this.dir } : {}),
      ...(this.socketPath ? { DSBOT_STEER_SOCK: this.socketPath } : {}),
      ...(this.url ? { DSBOT_STEER_URL: this.url } : {}),
      DSBOT_STEER_TOKEN: this.token,
    }
  }

  setPending(conversationId, pending) {
    if (this.dir === undefined) return
    const name = safeMarkerName(conversationId)
    if (name === undefined) return
    const marker = join(this.dir, name)
    if (pending) {
      try { writeFileSync(marker, '', { mode: 0o600 }) } catch {}
    } else {
      try { rmSync(marker, { force: true }) } catch {}
    }
  }

  close() {
    const dir = this.dir
    this.dir = undefined
    this.socketPath = undefined
    this.url = undefined
    if (this.onExit !== undefined) {
      process.removeListener('exit', this.onExit)
      this.onExit = undefined
    }
    try { this.server.close() } catch {}
    try { this.server.closeAllConnections?.() } catch {}
    if (dir !== undefined) {
      try { rmSync(dir, { recursive: true, force: true }) } catch {}
    }
  }

  async serve(request, response) {
    if (request.headers.authorization !== `Bearer ${this.token}`) {
      response.writeHead(401, { 'content-type': 'application/json' })
      response.end(JSON.stringify({ error: 'unauthorized' }))
      return
    }
    if (request.method !== 'POST') {
      response.writeHead(405, { 'content-type': 'application/json' })
      response.end('{}')
      return
    }
    const raw = await readBody(request)
    let payload
    try { payload = JSON.parse(raw) } catch {
      response.writeHead(400, { 'content-type': 'application/json' })
      response.end('{}')
      return
    }
    const conversationId = typeof payload?.conversationId === 'string' ? payload.conversationId : ''
    const hookEventName = payload?.hookEventName
    let result = {}
    if (conversationId !== '') {
      if (hookEventName === 'PreInvocation' || hookEventName === 'PostInvocation') {
        const items = (this.drain?.(conversationId) ?? []).filter(text => typeof text === 'string' && text !== '')
        if (this.hasPending?.(conversationId) !== true) this.setPending(conversationId, false)
        if (items.length > 0) {
          result = {
            injectSteps: items.map(userMessage => ({ userMessage })),
            ...(hookEventName === 'PostInvocation' ? { terminationBehavior: 'force_continue' } : {}),
          }
        }
      } else if (hookEventName === 'Stop') {
        if (!TERMINAL_STOP_REASONS.has(payload?.terminationReason) && this.hasPending?.(conversationId) === true) {
          this.setPending(conversationId, true)
          result = {
            decision: 'continue',
            reason: 'A new message arrived while finishing this step; read and address it before ending the turn.',
          }
        } else {
          this.setPending(conversationId, false)
          const fullyIdle = typeof payload?.fullyIdle === 'boolean' ? payload.fullyIdle : undefined
          const transcriptPath = typeof payload?.transcriptPath === 'string' && payload.transcriptPath !== '' ? payload.transcriptPath : undefined
          const terminationReason = typeof payload?.terminationReason === 'string' && payload.terminationReason !== '' ? payload.terminationReason : undefined
          if (fullyIdle !== undefined || transcriptPath !== undefined || terminationReason !== undefined) {
            await this.onStopState?.(conversationId, fullyIdle, transcriptPath, terminationReason)
          }
        }
      }
    }
    response.writeHead(200, { 'content-type': 'application/json' })
    response.end(JSON.stringify(result))
  }
}

const readBody = request => new Promise((resolve, reject) => {
  let body = ''
  request.setEncoding('utf8')
  request.on('data', chunk => { body += chunk })
  request.on('end', () => resolve(body))
  request.on('error', reject)
})

const TAIL_WATERMARK_BYTES = 64 * 1024
const TAIL_COLD_BYTES = 512 * 1024
const FULL_SCAN_CHUNK_BYTES = 2 * 1024 * 1024

function isCompleteJsonObjectLine(line) {
  if (line === '') return false
  try {
    const parsed = JSON.parse(line)
    return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)
  } catch {
    return false
  }
}

function readTailSlice(filePath, fromOffset, maxBytes) {
  let fd
  try {
    fd = openSync(filePath, 'r')
    const { size } = fstatSync(fd)
    if (size === 0) return { size: 0, start: 0, consumedEnd: 0, text: '', lineOffsets: [] }
    const validOffset = typeof fromOffset === 'number' && fromOffset >= 0 && fromOffset <= size
    if (validOffset) {
      const start = fromOffset
      const length = size - start
      if (length === 0) return { size, start, consumedEnd: size, text: '', lineOffsets: [] }
      const buf = Buffer.allocUnsafe(length)
      const bytesRead = readSync(fd, buf, 0, length, start)
      return finalizeAlignedSlice(size, start, buf.subarray(0, bytesRead))
    }
    let windowBytes = Math.max(4096, maxBytes)
    for (;;) {
      const probeStart = Math.max(0, size - windowBytes)
      const length = size - probeStart
      const buf = Buffer.allocUnsafe(length)
      const bytesRead = readSync(fd, buf, 0, length, probeStart)
      const probeBuf = buf.subarray(0, bytesRead)
      let alignedStart = probeStart
      let alignedBuf = probeBuf
      if (probeStart > 0) {
        const firstNl = probeBuf.indexOf(0x0a)
        if (firstNl === -1) {
          windowBytes *= 2
          continue
        }
        alignedStart = probeStart + firstNl + 1
        alignedBuf = probeBuf.subarray(firstNl + 1)
      }
      const finalized = finalizeAlignedSlice(size, alignedStart, alignedBuf)
      // If an incomplete trailing record hid every complete line in this window and
      // we have not reached BOF yet, expand backward so we still capture the last
      // complete step's index alongside its exact trailing boundary.
      if (probeStart > 0 && finalized.text.trim() === '' && finalized.consumedEnd < size) {
        windowBytes *= 2
        continue
      }
      return finalized
    }
  } catch {
    return undefined
  } finally {
    if (fd !== undefined) {
      try { closeSync(fd) } catch {}
    }
  }
}

function finalizeAlignedSlice(size, alignedStart, rawBuf) {
  const lastNl = rawBuf.lastIndexOf(0x0a)
  let consumedEnd = alignedStart + rawBuf.length
  let completeBuf = rawBuf
  if (lastNl !== rawBuf.length - 1) {
    const tailLine = rawBuf.subarray(lastNl + 1).toString('utf8').trim()
    const tailComplete = tailLine === '' || isCompleteJsonObjectLine(tailLine)
    if (!tailComplete) {
      consumedEnd = lastNl === -1 ? alignedStart : alignedStart + lastNl + 1
      completeBuf = lastNl === -1 ? rawBuf.subarray(0, 0) : rawBuf.subarray(0, lastNl + 1)
    }
  }
  const lineOffsets = []
  let lineStart = 0
  for (let i = 0; i < completeBuf.length; i += 1) {
    if (completeBuf[i] === 0x0a) {
      lineOffsets.push({ start: alignedStart + lineStart, end: alignedStart + i + 1, buf: completeBuf.subarray(lineStart, i) })
      lineStart = i + 1
    }
  }
  if (lineStart < completeBuf.length) {
    lineOffsets.push({ start: alignedStart + lineStart, end: alignedStart + completeBuf.length, buf: completeBuf.subarray(lineStart) })
  }
  return { size, start: alignedStart, consumedEnd, text: completeBuf.toString('utf8'), lineOffsets }
}

function parseJsonlStepsWithOffsets(lineOffsets) {
  const entries = []
  for (const item of lineOffsets) {
    const line = item.buf.toString('utf8').trim()
    if (line === '') continue
    try {
      const parsed = JSON.parse(line)
      if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
        entries.push({ step: parsed, startOffset: item.start, endOffset: item.end })
      }
    } catch { /* ignore incomplete or malformed line */ }
  }
  return entries
}

export function readAgyTranscriptWatermark(transcriptPath) {
  if (typeof transcriptPath !== 'string' || !transcriptPath.endsWith('.jsonl')) return undefined
  const slice = readTailSlice(transcriptPath, undefined, TAIL_WATERMARK_BYTES)
  if (slice === undefined) return undefined
  let maxStepIndex = -1
  for (const { step } of parseJsonlStepsWithOffsets(slice.lineOffsets)) {
    if (typeof step.step_index === 'number' && step.step_index > maxStepIndex) maxStepIndex = step.step_index
  }
  return { filePath: transcriptPath, byteOffset: slice.consumedEnd, maxStepIndex }
}

function loadFullStepMap(transcriptPath, neededIndices) {
  if (neededIndices.size === 0) return new Map()
  const fullPath = join(dirname(transcriptPath), 'transcript_full.jsonl')
  let fd
  try {
    fd = openSync(fullPath, 'r')
    const { size } = fstatSync(fd)
    if (size === 0) return undefined
    const byIndex = new Map()
    let pos = size
    let carry = Buffer.alloc(0)
    while (pos > 0 && byIndex.size < neededIndices.size) {
      const chunkStart = Math.max(0, pos - FULL_SCAN_CHUNK_BYTES)
      const chunkLen = pos - chunkStart
      const chunk = Buffer.allocUnsafe(chunkLen)
      const bytesRead = readSync(fd, chunk, 0, chunkLen, chunkStart)
      const combined = carry.length > 0
        ? Buffer.concat([chunk.subarray(0, bytesRead), carry])
        : chunk.subarray(0, bytesRead)
      let linesBuf = combined
      if (chunkStart > 0) {
        const firstNl = combined.indexOf(0x0a)
        if (firstNl === -1) {
          carry = combined
          pos = chunkStart
          continue
        }
        carry = combined.subarray(0, firstNl)
        linesBuf = combined.subarray(firstNl + 1)
      } else {
        carry = Buffer.alloc(0)
      }
      for (const raw of linesBuf.toString('utf8').split('\n')) {
        const line = raw.trim()
        if (line === '') continue
        try {
          const step = JSON.parse(line)
          if (step !== null && typeof step === 'object' && typeof step.step_index === 'number' && neededIndices.has(step.step_index)) {
            byIndex.set(step.step_index, step)
          }
        } catch { /* ignore partial trailing line in transcript_full.jsonl */ }
      }
      pos = chunkStart
    }
    return byIndex
  } catch {
    return undefined
  } finally {
    if (fd !== undefined) {
      try { closeSync(fd) } catch {}
    }
  }
}

export function readAgyTranscriptWakeup(transcriptPath, watermark, { allowIncomplete = false } = {}) {
  if (typeof transcriptPath !== 'string' || !transcriptPath.endsWith('.jsonl')) return undefined
  const hasMark = watermark !== undefined && watermark.filePath === transcriptPath
  const slice = readTailSlice(transcriptPath, hasMark ? watermark.byteOffset : undefined, TAIL_COLD_BYTES)
  if (slice === undefined) return undefined
  const parsedEntries = parseJsonlStepsWithOffsets(slice.lineOffsets)
  const floor = hasMark && typeof watermark.maxStepIndex === 'number' ? watermark.maxStepIndex : -1
  let candidateEntries
  if (hasMark) {
    candidateEntries = parsedEntries.filter(({ step }) => typeof step.step_index !== 'number' || step.step_index > floor)
  } else {
    // Cold fallback when no foreground Stop watermark was recorded yet: take only
    // steps following a SYSTEM wakeup after the latest USER_INPUT / USER_EXPLICIT.
    const lastUserAt = parsedEntries.findLastIndex(({ step }) => step.type === 'USER_INPUT' || step.source === 'USER_EXPLICIT')
    const wakeAt = parsedEntries.findIndex(({ step }, idx) => idx > lastUserAt && step.source === 'SYSTEM')
    candidateEntries = wakeAt === -1 ? [] : parsedEntries.slice(wakeAt + 1)
  }
  const neededFull = new Set()
  for (const { step } of candidateEntries) {
    if (step.type !== 'PLANNER_RESPONSE' || step.status === 'ERROR') continue
    const truncated = Array.isArray(step.truncated_fields) ? step.truncated_fields : []
    if ((truncated.includes('content') || truncated.includes('thinking')) && typeof step.step_index === 'number') {
      neededFull.add(step.step_index)
    }
  }
  const fullMap = neededFull.size > 0 ? loadFullStepMap(transcriptPath, neededFull) : undefined
  const events = []
  let maxStepIndex = floor
  let consumedEnd = slice.consumedEnd
  let unresolvedTruncated = false
  if (!hasMark) {
    // Advance past any pre-wake steps even if a post-wake step pauses for hydration.
    for (const { step, endOffset } of parsedEntries) {
      if (candidateEntries.length > 0 && step === candidateEntries[0].step) break
      if (typeof step.step_index === 'number' && step.step_index > maxStepIndex) maxStepIndex = step.step_index
      consumedEnd = endOffset
    }
  }
  for (const { step: rawStep, startOffset, endOffset } of candidateEntries) {
    if (rawStep.type === 'PLANNER_RESPONSE' && rawStep.status !== 'ERROR') {
      let step = rawStep
      const truncated = Array.isArray(rawStep.truncated_fields) ? rawStep.truncated_fields : []
      const needsHydration = (truncated.includes('content') || truncated.includes('thinking')) && typeof rawStep.step_index === 'number'
      let missingFull = false
      if (needsHydration) {
        const full = fullMap?.get(rawStep.step_index)
        if (full !== undefined) step = full
        else if (!allowIncomplete) {
          // Hold the watermark right before this unresolved step so a retry or
          // subsequent Stop hook hydrates the full record instead of committing
          // a truncated preview permanently.
          consumedEnd = startOffset
          unresolvedTruncated = true
          break
        } else {
          missingFull = true
        }
      }
      if (typeof step.thinking === 'string' && step.thinking.trim() !== '') {
        events.push({ type: 'thought', text: step.thinking })
      }
      if (typeof step.content === 'string' && step.content.trim() !== '') {
        const suffix = missingFull && truncated.includes('content') ? '\n\n[Output truncated — full transcript unavailable]' : ''
        const contentText = `${step.content}${suffix}`
        const prev = events.at(-1)
        events.push({ type: 'text', text: prev?.type === 'text' ? `\n\n${contentText}` : contentText })
      } else if (missingFull && truncated.includes('content')) {
        const prev = events.at(-1)
        const notice = '[Output truncated — full transcript unavailable]'
        events.push({ type: 'text', text: prev?.type === 'text' ? `\n\n${notice}` : notice })
      }
      if (Array.isArray(step.tool_calls)) {
        for (const call of step.tool_calls) {
          const name = call?.name ?? call?.tool_name ?? 'tool'
          events.push({ type: 'activity', text: `▸ ${name}` })
        }
      }
    }
    if (typeof rawStep.step_index === 'number' && rawStep.step_index > maxStepIndex) maxStepIndex = rawStep.step_index
    consumedEnd = endOffset
  }
  if (!unresolvedTruncated) {
    for (const { step } of parsedEntries) {
      if (typeof step.step_index === 'number' && step.step_index > maxStepIndex) maxStepIndex = step.step_index
    }
    consumedEnd = slice.consumedEnd
  }
  return {
    events,
    unresolvedTruncated,
    watermark: { filePath: transcriptPath, byteOffset: consumedEnd, maxStepIndex },
  }
}
