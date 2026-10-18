/**
 * /provider — connect / disconnect AI providers.
 *
 * Two-view state machine:
 *   list       → overview of all manageable providers with auth badges
 *   configure  → per-provider options: Activate OAuth / Activate API Key / Deactivate
 *
 * Core rules (from spec):
 *   - Multiple providers can be connected simultaneously.
 *   - OAuth and API Key on the SAME provider are mutually exclusive —
 *     switching one deactivates the other (mutex is already enforced
 *     inside saveProviderKey / OAuth finish paths, we just call them).
 *   - The user only sees "OpenAI" and "Google Gemini" — CLIProxyAPI /
 *     Antigravity / Codex are implicit engines behind OAuth and never
 *     surface as provider rows.
 *   - OAuth flow is one step: browser opens, user picks account, done.
 *   - API Key flow: prompt → paste → validate → done.
 *
 * This command generally manages credentials. The Anthropic login path also
 * switches routing to firstParty after a successful login.
 */

import * as React from 'react'
import { useEffect, useState } from 'react'
import chalk from 'chalk'
import { Box, Text, useInput } from '../../ink.js'
import type { CommandResultDisplay } from '../../commands.js'
import type {
  LocalJSXCommandCall,
  LocalJSXCommandContext,
} from '../../types/command.js'
import { stripSignatureBlocks } from '../../utils/messages.js'
import {
  getAPIProvider,
  PROVIDER_DISPLAY_NAMES,
  setActiveProvider,
  type APIProvider,
} from '../../utils/model/providers.js'
import {
  deleteAllProviderCredentials,
  deleteProviderKey,
  hasStoredKey,
  loadProviderKey,
  saveProviderKey,
} from '../../services/api/auth/api_key_manager.js'
import TextInput from '../../components/TextInput.js'
import {
  getClaudeAIOAuthTokens,
  hasAnthropicApiKeyAuth,
} from '../../utils/auth.js'
import { performLogout } from '../logout/logout.js'
import {
  E2BSecurityLogin,
  Login as AnthropicLogin,
} from '../login/login.js'
import { ProviderLoginFlow } from '../../components/ProviderLoginFlow.js'
import {
  E2B_SECURITY_DISPLAY_NAME,
  E2B_SECURITY_PROVIDER,
  clearE2BSecurityCredentials,
  formatE2BSecurityBadge,
  hasE2BSecurityAuth,
} from '../../utils/safetest/e2bSecurity.js'
import {
  addUserCustomProvider,
  getCustomProvider,
  getCustomProviderApiKey,
  getCustomProviders,
  getUserCustomProviders,
  isCustomProvider,
  isPolicyOnlyCustomProvider,
  isReservedProviderId,
  removeUserCustomProvider,
  resetCustomProviderCache,
  updateUserCustomProvider,
  type CustomProvider,
} from '../../utils/customProviders.js'

// ─── Config ──────────────────────────────────────────────────────

/**
 * Providers the user can connect via /provider.
 *
 * Anthropic (firstParty) delegates to /login so the OAuth choices stay
 * Tau-compatible.
 *
 * Excluded on purpose:
 *   - bedrock/vertex/foundry  → env/IAM-based, no credentials to manage here
 *
 * Ollama is a special case: no OAuth, no API key — the "credential" is
 * whether the local daemon is reachable at the configured base URL, and
 * the only configurable bit is that base URL.
 *
 * CLIProxyAPI/Antigravity/Codex are implicit engines behind Gemini &
 * OpenAI OAuth — not listed as separate rows.
 */
// `iflow` is hidden from user-facing provider management after the iFlow
// CLI shutdown announcement on April 17, 2026. `modelrouter` is also hidden
// from /provider while its auth/routing code stays wired for compatibility.
const MANAGEABLE_PROVIDERS = [
  // Anthropic surfaces here alongside the third-party providers. The
  // configure view hands Anthropic off to the shared /login OAuth flow
  // (subscription / console / 3rd-party platform) instead of a deactivate
  // prompt, so the credentials path stays Tau-compatible.
  'firstParty',
  'openai',
  'commandcode',
  'antigravity',
  'openrouter',
  'agentrouter',
  'vercel',
  'requesty',
  'opencode',
  'opencodego',
  'lxd',
  'mimo',
  'fireworks',
  'cloudflare',
  'mistral',
  'nim',
  'deepseek',
  'glm',
  'moonshot',
  'minimax',
  'alibaba',
  'xai',
  'meta',
  'ollama',
  'lmstudio',
  // Phase 4 (v0.4.0) — 3 full-chat + 3 login-only stubs.
  'kilocode',
  'cline',
  'clinepass',
  'copilot',
  'kiro',
] as const satisfies readonly APIProvider[]

const MANAGEABLE_PROVIDER_ROWS = [
  ...MANAGEABLE_PROVIDERS,
  E2B_SECURITY_PROVIDER,
] as const

type BuiltinManageableProvider = (typeof MANAGEABLE_PROVIDER_ROWS)[number]

/**
 * A row in /provider: either a built-in manageable provider or a
 * user-defined custom provider id from settings.json (`customProviders`).
 * Custom ids are arbitrary strings, so the union is widened to string.
 */
type ManageableProvider = BuiltinManageableProvider | string

/**
 * The full provider row list: built-ins followed by custom providers declared
 * in trusted settings. Custom ids are appended so users can activate them the
 * same way they activate a built-in provider.
 */
function getProviderRows(): string[] {
  // Re-read settings so a custom provider added since startup shows up.
  resetCustomProviderCache()
  return [...MANAGEABLE_PROVIDER_ROWS, ...getCustomProviders().map(p => p.id)]
}

/** Storage key for the user-supplied Ollama base URL (persisted in provider-keys.json). */
const OLLAMA_BASE_URL_KEY = 'ollama_base_url'
const OLLAMA_DEFAULT_BASE = 'http://localhost:11434'
const LMSTUDIO_BASE_URL_KEY = 'lmstudio_base_url'
const LMSTUDIO_DEFAULT_BASE = 'http://localhost:1234/v1'

/**
 * Synthetic row id for the "Add custom provider" action appended to the list.
 * Not a real provider — selecting it opens the add wizard.
 */
const ADD_CUSTOM_SENTINEL = '__add_custom_provider__'

type KeyedProvider = Exclude<
  ManageableProvider,
  | 'ollama'
  | 'lmstudio'
  | 'firstParty'
  | typeof E2B_SECURITY_PROVIDER
>

function getManageableProviderName(provider: ManageableProvider): string {
  if (provider === E2B_SECURITY_PROVIDER) return E2B_SECURITY_DISPLAY_NAME
  if (provider === ADD_CUSTOM_SENTINEL) return '＋ Add custom provider'
  if (isCustomProvider(provider)) {
    return getCustomProvider(provider)?.name ?? provider
  }
  return PROVIDER_DISPLAY_NAMES[provider as APIProvider]
}

// ─── Auth state helpers ──────────────────────────────────────────

type AuthState = 'oauth' | 'api_key' | 'inactive'

function getFirstPartyAuthState(): AuthState {
  if (getClaudeAIOAuthTokens()?.accessToken) return 'oauth'
  if (hasAnthropicApiKeyAuth()) return 'api_key'
  return 'inactive'
}

function hasFirstPartyAuth(): boolean {
  return getFirstPartyAuthState() !== 'inactive'
}

function getAuthState(provider: KeyedProvider): AuthState {
  // Custom providers store their key under their own id; the "configured"
  // state is whether a key is resolvable (env `<ID>_API_KEY` or saved key).
  if (isCustomProvider(provider)) {
    return getCustomProviderApiKey(provider) ? 'api_key' : 'inactive'
  }
  // Gemini row = CLI-tier OAuth (free flash/lite) or AI Studio API key.
  if (provider === 'gemini') {
    if (hasStoredKey('gemini_oauth_cli') || hasStoredKey('gemini_oauth')) return 'oauth'
    if (hasStoredKey('gemini')) return 'api_key'
    return 'inactive'
  }
  // Antigravity row = its own Google-login OAuth pool.
  if (provider === 'antigravity') {
    if (hasStoredKey('gemini_oauth_antigravity')) return 'oauth'
    return 'inactive'
  }
  if (hasStoredKey(`${provider}_oauth`)) return 'oauth'
  if (hasStoredKey(provider)) return 'api_key'
  return 'inactive'
}

/** Detailed Gemini auth state — shows CLI-tier OAuth + AI Studio API key. */
function getGeminiDetailedState(): { cliOAuth: boolean; apiKey: boolean } {
  return {
    cliOAuth: hasStoredKey('gemini_oauth_cli') || hasStoredKey('gemini_oauth'),
    apiKey: hasStoredKey('gemini'),
  }
}

function formatBadge(state: AuthState): string {
  switch (state) {
    case 'oauth':
      return chalk.green('[OAuth ✅]')
    case 'api_key':
      return chalk.green('[API Key ✅]')
    case 'inactive':
      return chalk.dim('[   –   ]')
  }
}

function formatGeminiBadge(): string {
  const { cliOAuth, apiKey } = getGeminiDetailedState()
  const parts: string[] = []
  if (cliOAuth) parts.push('CLI')
  if (apiKey) parts.push('Key')
  if (parts.length === 0) return chalk.dim('[   –   ]')
  return chalk.green(`[${parts.join(' + ')} ✅]`)
}

// ─── Ollama reachability ──────────────────────────────────────────
//
// Ollama has no credentials — we treat the "state" as whether the
// daemon actually answers. The reachability probe is async so the
// list view holds it in React state and the badge reflects the last
// result.

type OllamaStatus = 'unknown' | 'running' | 'offline'

function getOllamaBaseUrl(): string {
  // Order of precedence matches ollamaCatalog.ts:
  //   OLLAMA_HOST → OLLAMA_BASE_URL → stored override → default.
  // We normalise to a bare origin (no /v1 suffix) because /api/tags
  // lives under the root, not under the OpenAI-compat path.
  const envHost = process.env.OLLAMA_HOST ?? process.env.OLLAMA_BASE_URL
  const stored = loadProviderKey(OLLAMA_BASE_URL_KEY)
  const raw = envHost ?? stored ?? OLLAMA_DEFAULT_BASE
  const withScheme = /^https?:/i.test(raw) ? raw : `http://${raw}`
  return withScheme.replace(/\/+$/, '').replace(/\/v1$/i, '')
}

async function probeOllama(baseUrl: string, signal?: AbortSignal): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/api/tags`, { signal })
    return res.ok
  } catch {
    return false
  }
}

function formatOllamaBadge(status: OllamaStatus): string {
  switch (status) {
    case 'running':
      return chalk.green('[🟢 Running]')
    case 'offline':
      return chalk.red('[🔴 Offline]')
    case 'unknown':
      return chalk.dim('[   ?    ]')
  }
}

type LmStudioStatus = 'unknown' | 'running' | 'offline'

function normalizeLmStudioBaseUrl(raw: string): string {
  const withScheme = /^https?:\/\//i.test(raw) ? raw : `http://${raw}`
  const trimmed = withScheme.replace(/\/+$/, '')
  return /\/v1$/i.test(trimmed) ? trimmed : `${trimmed}/v1`
}

function getLmStudioBaseUrl(): string {
  const raw =
    process.env.LMSTUDIO_BASE_URL
    ?? process.env.LM_STUDIO_BASE_URL
    ?? loadProviderKey(LMSTUDIO_BASE_URL_KEY)
    ?? LMSTUDIO_DEFAULT_BASE
  return normalizeLmStudioBaseUrl(raw)
}

async function probeLmStudio(baseUrl: string, signal?: AbortSignal): Promise<boolean> {
  const rootUrl = normalizeLmStudioBaseUrl(baseUrl).replace(/\/(?:api\/)?v1$/i, '')
  const urls = [
    `${normalizeLmStudioBaseUrl(baseUrl).replace(/\/+$/, '')}/models`,
    `${rootUrl}/api/v1/models`,
  ]

  for (const url of urls) {
    try {
      const res = await fetch(url, { signal })
      if (res.ok || res.status === 401 || res.status === 403) {
        return true
      }
    } catch {
      if (signal?.aborted) return false
    }
  }
  return false
}

function formatLmStudioBadge(status: LmStudioStatus): string {
  const active = getAPIProvider() === 'lmstudio' ? `${chalk.green('[Active]')} ` : ''
  switch (status) {
    case 'running':
      return `${active}${chalk.green('[Running]')}`
    case 'offline':
      return `${active}${chalk.red('[Offline]')}`
    case 'unknown':
      return `${active}${chalk.dim('[   ?   ]')}`
  }
}

// ─── View state machine ─────────────────────────────────────────

type View =
  | { kind: 'list'; selectedIndex: number }
  | { kind: 'configure'; provider: ManageableProvider; selectedIndex: number }
  | {
      kind: 'ollama_url_input'
      error?: string
    }
  | {
      kind: 'lmstudio_url_input'
      error?: string
    }
  | {
      kind: 'custom_form'
      mode: 'add' | 'edit'
      id?: string
      /** Ordered field names for this wizard run (add vs edit differ). */
      steps: CustomFormStep[]
      /** Index into `steps`. */
      step: number
      /** Cursor for the protocol menu (0 = openai, 1 = anthropic). */
      protocolCursor: number
      draft: CustomProviderDraft
      error?: string
    }
  | { kind: 'custom_delete'; id: string; selectedIndex: number }
  | { kind: 'provider_login'; provider: KeyedProvider }
  | { kind: 'anthropic_login' }
  | { kind: 'e2b_login' }
  | {
      kind: 'result'
      provider: ManageableProvider
      message: string
      tone: 'success' | 'error'
    }

/** Fields of the custom provider add/edit wizard, in flow order. */
type CustomFormStep = 'id' | 'name' | 'protocol' | 'baseUrl' | 'apiKey'

function customFormSteps(mode: 'add' | 'edit'): CustomFormStep[] {
  return mode === 'add'
    ? ['id', 'name', 'protocol', 'baseUrl', 'apiKey']
    : ['name', 'protocol', 'baseUrl', 'apiKey']
}

/** In-progress custom provider definition in the add/edit wizard. */
type CustomProviderDraft = {
  id: string
  name: string
  protocol: 'anthropic' | 'openai'
  baseUrl: string
  /** API key typed in the wizard ('' = skip / keep existing). */
  apiKey: string
}

type ConfigureOption =
  | { kind: 'activate' }
  | { kind: 'login' }
  | { kind: 'deactivate' }
  | { kind: 'edit_custom' }
  | { kind: 'remove_custom' }
  | { kind: 'set_ollama_url' }
  | { kind: 'reset_ollama_url' }
  | { kind: 'test_ollama' }
  | { kind: 'set_lmstudio_url' }
  | { kind: 'reset_lmstudio_url' }
  | { kind: 'test_lmstudio' }
  | { kind: 'back' }

function usesEmbeddedProviderLogin(provider: ManageableProvider): provider is KeyedProvider {
  return (
    provider === 'modelrouter' ||
    provider === 'vercel' ||
    provider === 'requesty' ||
    provider === 'opencode' ||
    provider === 'opencodego' ||
    provider === 'commandcode' ||
    provider === 'fireworks' ||
    provider === 'cloudflare' ||
    provider === 'cline' ||
    provider === 'clinepass' ||
    provider === 'copilot' ||
    provider === 'kilocode' ||
    provider === 'kiro'
  )
}

function buildConfigureOptions(
  provider: ManageableProvider,
  ollamaStatus: OllamaStatus,
  lmStudioStatus: LmStudioStatus,
): ConfigureOption[] {
  // Custom providers: activate / edit / remove. There is no OAuth flow.
  if (isCustomProvider(provider)) {
    const options: ConfigureOption[] = []
    if (getAPIProvider() !== provider) {
      options.push({ kind: 'activate' })
    }
    // Policy-managed providers are read-only here — hide edit/remove so the
    // user is not told a change succeeded when it cannot be persisted.
    if (!isPolicyOnlyCustomProvider(provider)) {
      options.push({ kind: 'edit_custom' })
    }
    if (getAuthState(provider) !== 'inactive') {
      options.push({ kind: 'deactivate' })
    }
    if (!isPolicyOnlyCustomProvider(provider)) {
      options.push({ kind: 'remove_custom' })
    }
    options.push({ kind: 'back' })
    return options
  }

  if (provider === E2B_SECURITY_PROVIDER) {
    const options: ConfigureOption[] = []
    options.push({ kind: 'login' })
    if (hasE2BSecurityAuth()) {
      options.push({ kind: 'deactivate' })
    }
    options.push({ kind: 'back' })
    return options
  }

  // Ollama has its own option set.
  if (provider === 'ollama') {
    const options: ConfigureOption[] = []
    options.push({ kind: 'test_ollama' })
    options.push({ kind: 'set_ollama_url' })
    if (hasStoredKey(OLLAMA_BASE_URL_KEY)) {
      options.push({ kind: 'reset_ollama_url' })
    }
    options.push({ kind: 'back' })
    void ollamaStatus
    return options
  }

  if (provider === 'lmstudio') {
    const options: ConfigureOption[] = []
    if (getAPIProvider() !== 'lmstudio') {
      options.push({ kind: 'activate' })
    }
    options.push({ kind: 'test_lmstudio' })
    options.push({ kind: 'set_lmstudio_url' })
    if (hasStoredKey(LMSTUDIO_BASE_URL_KEY)) {
      options.push({ kind: 'reset_lmstudio_url' })
    }
    options.push({ kind: 'back' })
    void lmStudioStatus
    return options
  }

  // Anthropic (firstParty): hand off to the shared /login OAuth flow.
  if (provider === 'firstParty') {
    const options: ConfigureOption[] = []
    options.push({ kind: 'login' })
    if (hasFirstPartyAuth()) {
      options.push({ kind: 'deactivate' })
    }
    options.push({ kind: 'back' })
    return options
  }

  const options: ConfigureOption[] = []

  if (usesEmbeddedProviderLogin(provider)) {
    options.push({ kind: 'login' })
  }

  // Gemini: check CLI-tier OAuth + API key (Antigravity has its own row).
  if (provider === 'gemini') {
    const gemini = getGeminiDetailedState()
    if (gemini.cliOAuth || gemini.apiKey) {
      options.push({ kind: 'deactivate' })
    }
  } else {
    const state = getAuthState(provider)
    if (
      provider === 'agentrouter' &&
      state !== 'inactive' &&
      getAPIProvider() !== 'agentrouter'
    ) {
      options.push({ kind: 'activate' })
    }
    if (state !== 'inactive') {
      options.push({ kind: 'deactivate' })
    }
  }

  options.push({ kind: 'back' })
  return options
}

function labelConfigureOption(
  option: ConfigureOption,
  provider: ManageableProvider,
): string {
  switch (option.kind) {
    case 'activate':
      return provider === 'agentrouter'
        ? 'Activate AgentRouter'
        : `Activate ${getManageableProviderName(provider)}`
    case 'login':
      return provider === 'firstParty'
        ? 'Log in with Anthropic (subscription / Console API / platform)'
        : provider === E2B_SECURITY_PROVIDER
        ? 'Log in with E2B API key or auth token'
        : 'Log in'
    case 'deactivate':
      return provider === E2B_SECURITY_PROVIDER
        ? 'Clear E2B credentials'
        : 'Deactivate (clear all credentials)'
    case 'edit_custom':
      return 'Edit name / protocol / base URL'
    case 'remove_custom':
      return 'Remove provider (delete from settings)'
    case 'set_ollama_url':
      return 'Set custom base URL'
    case 'reset_ollama_url':
      return 'Reset base URL to default (http://localhost:11434)'
    case 'test_ollama':
      return 'Test connection'
    case 'set_lmstudio_url':
      return 'Set custom base URL'
    case 'reset_lmstudio_url':
      return 'Reset base URL to default (http://localhost:1234/v1)'
    case 'test_lmstudio':
      return 'Test connection'
    case 'back':
      return '← Back to provider list'
  }
}

// ─── Component ──────────────────────────────────────────────────

type OnDone = (
  result?: string,
  options?: { display?: CommandResultDisplay },
) => void

function ProviderManager({
  onDone,
  setMessages,
}: {
  onDone: OnDone
  setMessages: LocalJSXCommandContext['setMessages']
}) {
  const [view, setView] = useState<View>({ kind: 'list', selectedIndex: 0 })
  // Refresh tick forces re-read of auth state after saves/deletes.
  const [refreshTick, setRefreshTick] = useState(0)
  const refresh = () => setRefreshTick(t => t + 1)

  // Built-in rows + custom provider ids + the "Add custom provider" action.
  // Recomputed on refresh so a provider added/edited below shows up without
  // restarting the CLI.
  const providerRows = React.useMemo(
    () => [...getProviderRows(), ADD_CUSTOM_SENTINEL],
    [refreshTick],
  )

  const [ollamaUrlInput, setOllamaUrlInput] = useState('')
  const [ollamaUrlCursorOffset, setOllamaUrlCursorOffset] = useState(0)
  const [lmStudioUrlInput, setLmStudioUrlInput] = useState('')
  const [lmStudioUrlCursorOffset, setLmStudioUrlCursorOffset] = useState(0)
  // Text input state for the custom provider add/edit wizard.
  const [customFieldInput, setCustomFieldInput] = useState('')
  const [customFieldCursorOffset, setCustomFieldCursorOffset] = useState(0)
  const inputColumns = Math.max(20, (process.stdout.columns ?? 80) - 14)

  // Live reachability status for Ollama, computed when we first render
  // the list or configure view and refreshed when the user asks for it.
  // The badge starts at "unknown" and flips after the probe resolves.
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus>('unknown')
  const [lmStudioStatus, setLmStudioStatus] = useState<LmStudioStatus>('unknown')

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 1500)
    setOllamaStatus('unknown')
    probeOllama(getOllamaBaseUrl(), controller.signal).then(ok => {
      if (cancelled) return
      setOllamaStatus(ok ? 'running' : 'offline')
    })
    return () => {
      cancelled = true
      clearTimeout(timer)
      controller.abort()
    }
  }, [refreshTick])

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 1500)
    setLmStudioStatus('unknown')
    probeLmStudio(getLmStudioBaseUrl(), controller.signal).then(ok => {
      if (cancelled) return
      setLmStudioStatus(ok ? 'running' : 'offline')
    })
    return () => {
      cancelled = true
      clearTimeout(timer)
      controller.abort()
    }
  }, [refreshTick])

  // ─── Handlers ─────────────────────────────────────────────────

  function enterConfigure(provider: ManageableProvider) {
    if (provider === ADD_CUSTOM_SENTINEL) {
      // Open the add wizard with an empty draft.
      setCustomFieldInput('')
      setCustomFieldCursorOffset(0)
      setView({
        kind: 'custom_form',
        mode: 'add',
        steps: customFormSteps('add'),
        step: 0,
        protocolCursor: 0,
        draft: { id: '', name: '', protocol: 'openai', baseUrl: '', apiKey: '' },
      })
      return
    }
    setView({ kind: 'configure', provider, selectedIndex: 0 })
  }

  /** Open the edit wizard pre-filled from an existing custom provider. */
  function enterEditCustom(id: string) {
    const existing = getCustomProvider(id)
    if (!existing) return
    // Edit starts at the name field (id is immutable), pre-filled so the user
    // only retypes the field they actually want to change.
    setCustomFieldInput(existing.name)
    setCustomFieldCursorOffset(existing.name.length)
    setView({
      kind: 'custom_form',
      mode: 'edit',
      id,
      steps: customFormSteps('edit'),
      step: 0,
      protocolCursor: existing.protocol === 'anthropic' ? 1 : 0,
      draft: {
        id: existing.id,
        name: existing.name,
        protocol: existing.protocol,
        baseUrl: existing.baseUrl,
        apiKey: '',
      },
    })
  }

  function backToList() {
    // Preserve the list selection on the provider the user just configured.
    const lastProvider =
      view.kind === 'configure' || view.kind === 'result'
        ? view.provider
        : view.kind === 'custom_delete'
          ? view.id
          : view.kind === 'custom_form' && view.id
            ? view.id
            : undefined
    const idx = lastProvider ? providerRows.indexOf(lastProvider) : 0
    setView({ kind: 'list', selectedIndex: idx >= 0 ? idx : 0 })
  }

  function handleDeactivate(provider: KeyedProvider) {
    deleteAllProviderCredentials(provider)
    if (provider === 'gemini') {
      // Gemini row = CLI-tier only; Antigravity has its own row.
      deleteProviderKey('gemini_oauth_cli')
      deleteProviderKey('gemini_oauth')
    }
    if (provider === 'antigravity') {
      deleteProviderKey('gemini_oauth_antigravity')
    }
    refresh()
    setView({
      kind: 'result',
      provider,
      tone: 'success',
      message: `${getManageableProviderName(provider)} disconnected.`,
    })
  }

  function handleActivateAgentRouter() {
    setActiveProvider('agentrouter')
    setMessages(stripSignatureBlocks)
    refresh()
    setView({
      kind: 'result',
      provider: 'agentrouter',
      tone: 'success',
      message: 'AgentRouter activated.',
    })
  }

  function handleActivateLmStudio() {
    setActiveProvider('lmstudio')
    refresh()
    setView({
      kind: 'result',
      provider: 'lmstudio',
      tone: 'success',
      message: 'LM Studio activated.',
    })
  }

  function handleActivateCustom(id: string) {
    setActiveProvider(id as APIProvider)
    // Strip thinking-block signatures bound to the previously-active provider.
    setMessages(stripSignatureBlocks)
    refresh()
    setView({
      kind: 'result',
      provider: id,
      tone: 'success',
      message: `${getManageableProviderName(id)} activated.`,
    })
  }

  // ─── Custom provider add/edit wizard ──────────────────────────
  //
  // Field order (add):   id → name → protocol → baseUrl → apiKey
  // Field order (edit):        name → protocol → baseUrl → apiKey
  // Text steps use a single TextInput; `protocol` is a two-option menu;
  // `apiKey` is optional (blank skips on add, keeps existing on edit).

  function currentCustomStep(
    form: Extract<View, { kind: 'custom_form' }>,
  ): CustomFormStep | undefined {
    return form.steps[form.step]
  }

  /**
   * The value to pre-fill into a step's input on edit. On add, steps start
   * empty; on edit, the existing value is shown so the user can adjust a
   * single field and press Enter through the rest without retyping.
   */
  function customStepPrefill(
    form: Extract<View, { kind: 'custom_form' }>,
    step: CustomFormStep | undefined,
  ): string {
    if (form.mode !== 'edit') return ''
    switch (step) {
      case 'name':
        return form.draft.name
      case 'baseUrl':
        return form.draft.baseUrl
      case 'apiKey':
        return form.draft.apiKey
      default:
        return ''
    }
  }

  function validateCustomField(
    form: Extract<View, { kind: 'custom_form' }>,
    raw: string,
  ): { value?: string; error?: string } {
    const step = currentCustomStep(form)
    const value = raw.trim()
    switch (step) {
      case 'id': {
        if (!value) return { error: 'Provider id is required.' }
        if (!/^[a-z0-9][a-z0-9_-]*$/.test(value)) {
          return {
            error:
              'Id must be lowercase letters, digits, "-" or "_" (e.g. myproxy).',
          }
        }
        if (isReservedProviderId(value)) {
          return {
            error: `"${value}" is a built-in provider id. Pick a different id.`,
          }
        }
        if (isCustomProvider(value)) {
          return { error: `A provider with id "${value}" already exists.` }
        }
        return { value }
      }
      case 'name':
        if (!value) return { error: 'Name is required.' }
        return { value }
      case 'baseUrl': {
        if (!value) return { error: 'Base URL is required.' }
        try {
          const url = new URL(value)
          if (url.protocol !== 'http:' && url.protocol !== 'https:') {
            return { error: 'URL must start with http:// or https://' }
          }
        } catch {
          return {
            error: 'Not a valid URL (e.g. https://api.example.com/v1).',
          }
        }
        return { value }
      }
      case 'apiKey':
      case 'protocol':
      default:
        return { value }
    }
  }

  function advanceCustomForm(
    form: Extract<View, { kind: 'custom_form' }>,
    draft: CustomProviderDraft,
  ): void {
    if (form.step >= form.steps.length - 1) {
      finishCustomForm(form, draft)
      return
    }
    const nextStep = form.steps[form.step + 1]
    // Pre-fill the next step on edit so unchanged fields just need Enter.
    const prefill = customStepPrefill({ ...form, draft }, nextStep)
    setCustomFieldInput(prefill)
    setCustomFieldCursorOffset(prefill.length)
    setView({ ...form, step: form.step + 1, draft, error: undefined })
  }

  function finishCustomForm(
    form: Extract<View, { kind: 'custom_form' }>,
    draft: CustomProviderDraft,
  ): void {
    const provider: CustomProvider = {
      id: draft.id,
      name: draft.name,
      protocol: draft.protocol,
      baseUrl: draft.baseUrl,
    }
    const error =
      form.mode === 'add'
        ? addUserCustomProvider(provider)
        : updateUserCustomProvider(form.id!, {
            name: provider.name,
            protocol: provider.protocol,
            baseUrl: provider.baseUrl,
          })
    if (error) {
      setView({ ...form, error })
      return
    }
    // Persist the API key when the user typed one (blank = skip / keep).
    let keyNote = ''
    if (draft.apiKey) {
      try {
        saveProviderKey(provider.id, draft.apiKey)
        process.env[`${provider.id.toUpperCase().replace(/-/g, '_')}_API_KEY`] =
          draft.apiKey
        keyNote = ' API key saved.'
      } catch (e) {
        keyNote = ` (provider saved, but the API key could not be stored: ${
          e instanceof Error ? e.message : String(e)
        })`
      }
    }
    refresh()
    setCustomFieldInput('')
    setCustomFieldCursorOffset(0)
    setView({
      kind: 'result',
      provider: provider.id,
      tone: 'success',
      message:
        (form.mode === 'add'
          ? `Custom provider "${provider.name}" added.`
          : `Custom provider "${provider.name}" updated.`) + keyNote,
    })
  }

  function handleCustomFieldSubmit(raw: string) {
    if (view.kind !== 'custom_form') return
    const form = view
    const step = currentCustomStep(form)
    // The protocol step is a menu, not a text field.
    if (step === 'protocol') return
    const { value, error } = validateCustomField(form, raw)
    if (error !== undefined) {
      setView({ ...form, error })
      return
    }
    const draft: CustomProviderDraft = { ...form.draft }
    if (step === 'id') draft.id = value!
    else if (step === 'name') draft.name = value!
    else if (step === 'baseUrl') draft.baseUrl = value!
    else if (step === 'apiKey') draft.apiKey = value!
    advanceCustomForm(form, draft)
  }

  function handleCustomProtocolConfirm() {
    if (view.kind !== 'custom_form') return
    const form = view
    const draft: CustomProviderDraft = {
      ...form.draft,
      protocol: form.protocolCursor === 1 ? 'anthropic' : 'openai',
    }
    advanceCustomForm(form, draft)
  }

  function cancelCustomForm() {
    if (view.kind !== 'custom_form') return
    setCustomFieldInput('')
    setCustomFieldCursorOffset(0)
    if (view.mode === 'edit' && view.id) {
      setView({ kind: 'configure', provider: view.id, selectedIndex: 0 })
    } else {
      backToList()
    }
  }

  function handleRemoveCustom(id: string) {
    const name = getManageableProviderName(id)
    // Guard against a policy-managed provider: removeUserCustomProvider only
    // edits userSettings, so it would silently no-op and falsely report success.
    if (isPolicyOnlyCustomProvider(id)) {
      setView({
        kind: 'result',
        provider: id,
        tone: 'error',
        message: `${name} is managed by policy and cannot be removed here.`,
      })
      return
    }
    const error = removeUserCustomProvider(id)
    refresh()
    if (error) {
      setView({ kind: 'result', provider: id, tone: 'error', message: error })
      return
    }
    // If the removed provider was active, fall back to Anthropic.
    if (getAPIProvider() === id) {
      setActiveProvider('firstParty')
      setMessages(stripSignatureBlocks)
    }
    setView({
      kind: 'result',
      provider: id,
      tone: 'success',
      message: `${name} removed.`,
    })
  }

  // ─── Anthropic (firstParty) handlers ──────────────────────────

  function handleAnthropicLoginDone(success: boolean) {
    if (success) {
      setActiveProvider('firstParty')
      // Strip any thinking-block signatures bound to the previously-active
      // provider so the next turn doesn't 400 with "Invalid signature".
      setMessages(stripSignatureBlocks)
      refresh()
      setView({
        kind: 'result',
        provider: 'firstParty',
        tone: 'success',
        message: `${PROVIDER_DISPLAY_NAMES.firstParty} connected.`,
      })
      return
    }
    // Cancelled: return to the Anthropic configure screen.
    setView({ kind: 'configure', provider: 'firstParty', selectedIndex: 0 })
  }

  function handleAnthropicDeactivate() {
    void performLogout({ provider: 'firstParty' }).finally(() => {
      refresh()
      setView({
        kind: 'result',
        provider: 'firstParty',
        tone: 'success',
        message: `${PROVIDER_DISPLAY_NAMES.firstParty} disconnected.`,
      })
    })
  }

  function handleE2BLoginDone(success: boolean) {
    if (success) {
      refresh()
      setView({
        kind: 'result',
        provider: E2B_SECURITY_PROVIDER,
        tone: 'success',
        message: `${E2B_SECURITY_DISPLAY_NAME} connected.`,
      })
      return
    }
    setView({
      kind: 'configure',
      provider: E2B_SECURITY_PROVIDER,
      selectedIndex: 0,
    })
  }

  function handleE2BDeactivate() {
    clearE2BSecurityCredentials()
    refresh()
    setView({
      kind: 'result',
      provider: E2B_SECURITY_PROVIDER,
      tone: 'success',
      message: `${E2B_SECURITY_DISPLAY_NAME} credentials cleared.`,
    })
  }

  // ─── Ollama handlers ──────────────────────────────────────────

  function handleTestOllama() {
    // Refreshing forces the reachability effect to re-run.
    setOllamaStatus('unknown')
    refresh()
  }

  function handleOllamaUrlSubmit(value: string) {
    const raw = value.trim()
    if (!raw) return

    // Accept either a full URL or a host:port shorthand, matching
    // ollamaCatalog.ts's ollamaBase().
    const withScheme = /^https?:\/\//i.test(raw) ? raw : `http://${raw}`
    try {
      const parsed = new URL(withScheme)
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        throw new Error('bad protocol')
      }
    } catch {
      setView({
        kind: 'ollama_url_input',
        error: 'Not a valid URL. Try "http://localhost:11434" or "my-box:11434".',
      })
      return
    }

    const normalised = withScheme.replace(/\/+$/, '').replace(/\/v1$/i, '')
    saveProviderKey(OLLAMA_BASE_URL_KEY, normalised)
    process.env.OLLAMA_BASE_URL = normalised

    setOllamaUrlInput('')
    setOllamaUrlCursorOffset(0)
    setOllamaStatus('unknown')
    refresh()
    setView({
      kind: 'result',
      provider: 'ollama',
      tone: 'success',
      message: `Ollama base URL set to ${normalised}. Testing connection…`,
    })
  }

  function handleResetOllamaUrl() {
    deleteProviderKey(OLLAMA_BASE_URL_KEY)
    // Clear the env override too, so we fall back to the default.
    delete process.env.OLLAMA_BASE_URL
    delete process.env.OLLAMA_HOST
    setOllamaStatus('unknown')
    refresh()
    setView({
      kind: 'result',
      provider: 'ollama',
      tone: 'success',
      message: `Ollama base URL reset to ${OLLAMA_DEFAULT_BASE}.`,
    })
  }

  function reloadLmStudioInRuntime() {
    void import('../../services/api/providers/providerShim.js')
      .then(({ reloadOpenAICompatProviderAuth }) =>
        reloadOpenAICompatProviderAuth('lmstudio'),
      )
      .catch(() => {})
  }

  function handleTestLmStudio() {
    setLmStudioStatus('unknown')
    refresh()
  }

  function handleLmStudioUrlSubmit(value: string) {
    const raw = value.trim()
    if (!raw) return

    const withScheme = /^https?:\/\//i.test(raw) ? raw : `http://${raw}`
    try {
      const parsed = new URL(withScheme)
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        throw new Error('bad protocol')
      }
    } catch {
      setView({
        kind: 'lmstudio_url_input',
        error: 'Not a valid URL. Try "http://localhost:1234/v1" or "my-box:1234".',
      })
      return
    }

    const normalised = normalizeLmStudioBaseUrl(withScheme)
    saveProviderKey(LMSTUDIO_BASE_URL_KEY, normalised)
    process.env.LMSTUDIO_BASE_URL = normalised
    delete process.env.LM_STUDIO_BASE_URL
    reloadLmStudioInRuntime()

    setLmStudioUrlInput('')
    setLmStudioUrlCursorOffset(0)
    setLmStudioStatus('unknown')
    refresh()
    setView({
      kind: 'result',
      provider: 'lmstudio',
      tone: 'success',
      message: `LM Studio base URL set to ${normalised}. Testing connection...`,
    })
  }

  function handleResetLmStudioUrl() {
    deleteProviderKey(LMSTUDIO_BASE_URL_KEY)
    delete process.env.LMSTUDIO_BASE_URL
    delete process.env.LM_STUDIO_BASE_URL
    reloadLmStudioInRuntime()
    setLmStudioStatus('unknown')
    refresh()
    setView({
      kind: 'result',
      provider: 'lmstudio',
      tone: 'success',
      message: `LM Studio base URL reset to ${LMSTUDIO_DEFAULT_BASE}.`,
    })
  }

  // ─── Input routing ────────────────────────────────────────────

  useInput((input: string, key: {
    upArrow?: boolean
    downArrow?: boolean
    return?: boolean
    escape?: boolean
  }) => {
    // Global: Esc cancels the whole flow from any non-input view.
    // TextInput-backed views handle their own Esc.
    if (
      key.escape &&
      view.kind !== 'ollama_url_input' &&
      view.kind !== 'lmstudio_url_input' &&
      view.kind !== 'custom_form'
    ) {
      if (view.kind === 'list') {
        onDone('Provider setup closed.', { display: 'system' })
        return
      }
      backToList()
      return
    }

    // custom_form: Esc cancels the wizard (TextInput's own Esc is disabled via
    // the component's escape handling, so we route it here).
    if (view.kind === 'custom_form' && key.escape) {
      cancelCustomForm()
      return
    }

    // custom_form protocol step: up/down pick the protocol, Enter confirms.
    if (view.kind === 'custom_form' && currentCustomStep(view) === 'protocol') {
      if (key.upArrow || key.downArrow) {
        setView({
          ...view,
          protocolCursor: view.protocolCursor === 0 ? 1 : 0,
        })
        return
      }
      if (key.return) {
        handleCustomProtocolConfirm()
        return
      }
      return
    }

    // ollama_url_input: Esc goes back to the ollama configure screen.
    if (view.kind === 'ollama_url_input' && key.escape) {
      setOllamaUrlInput('')
      setOllamaUrlCursorOffset(0)
      setView({ kind: 'configure', provider: 'ollama', selectedIndex: 0 })
      return
    }

    if (view.kind === 'lmstudio_url_input' && key.escape) {
      setLmStudioUrlInput('')
      setLmStudioUrlCursorOffset(0)
      setView({ kind: 'configure', provider: 'lmstudio', selectedIndex: 0 })
      return
    }

    // ─── list view ───
    if (view.kind === 'list') {
      if (key.upArrow) {
        setView({
          kind: 'list',
          selectedIndex:
            view.selectedIndex > 0
              ? view.selectedIndex - 1
              : providerRows.length - 1,
        })
        return
      }
      if (key.downArrow) {
        setView({
          kind: 'list',
          selectedIndex:
            view.selectedIndex < providerRows.length - 1
              ? view.selectedIndex + 1
              : 0,
        })
        return
      }
      if (key.return) {
        const provider = providerRows[view.selectedIndex]
        if (provider) enterConfigure(provider)
        return
      }
      return
    }

    // ─── configure view ───
    if (view.kind === 'configure') {
      const options = buildConfigureOptions(view.provider, ollamaStatus, lmStudioStatus)
      if (key.upArrow) {
        setView({
          ...view,
          selectedIndex:
            view.selectedIndex > 0
              ? view.selectedIndex - 1
              : options.length - 1,
        })
        return
      }
      if (key.downArrow) {
        setView({
          ...view,
          selectedIndex:
            view.selectedIndex < options.length - 1
              ? view.selectedIndex + 1
              : 0,
        })
        return
      }
      if (key.return) {
        const chosen = options[view.selectedIndex]
        if (!chosen) return
        switch (chosen.kind) {
          case 'activate':
            if (view.provider === 'agentrouter') {
              handleActivateAgentRouter()
            }
            if (view.provider === 'lmstudio') {
              handleActivateLmStudio()
            }
            if (isCustomProvider(view.provider)) {
              handleActivateCustom(view.provider)
            }
            return
          case 'login':
            if (view.provider === 'firstParty') {
              setView({ kind: 'anthropic_login' })
            }
            if (view.provider === E2B_SECURITY_PROVIDER) {
              setView({ kind: 'e2b_login' })
            }
            if (usesEmbeddedProviderLogin(view.provider)) {
              setView({ kind: 'provider_login', provider: view.provider })
            }
            return
          case 'deactivate':
            if (view.provider === 'firstParty') {
              handleAnthropicDeactivate()
              return
            }
            if (view.provider === E2B_SECURITY_PROVIDER) {
              handleE2BDeactivate()
              return
            }
            if (view.provider !== 'ollama' && view.provider !== 'lmstudio') {
              handleDeactivate(view.provider)
            }
            return
          case 'edit_custom':
            enterEditCustom(view.provider)
            return
          case 'remove_custom':
            setView({ kind: 'custom_delete', id: view.provider, selectedIndex: 0 })
            return
          case 'set_ollama_url':
            setOllamaUrlInput('')
            setOllamaUrlCursorOffset(0)
            setView({ kind: 'ollama_url_input' })
            return
          case 'reset_ollama_url':
            handleResetOllamaUrl()
            return
          case 'test_ollama':
            handleTestOllama()
            return
          case 'set_lmstudio_url':
            setLmStudioUrlInput('')
            setLmStudioUrlCursorOffset(0)
            setView({ kind: 'lmstudio_url_input' })
            return
          case 'reset_lmstudio_url':
            handleResetLmStudioUrl()
            return
          case 'test_lmstudio':
            handleTestLmStudio()
            return
          case 'back':
            backToList()
            return
        }
      }
      return
    }

    // ─── custom provider delete confirmation ───
    if (view.kind === 'custom_delete') {
      if (key.upArrow || key.downArrow) {
        setView({ ...view, selectedIndex: view.selectedIndex === 0 ? 1 : 0 })
        return
      }
      if (key.return) {
        if (view.selectedIndex === 0) {
          handleRemoveCustom(view.id)
        } else {
          setView({ kind: 'configure', provider: view.id, selectedIndex: 0 })
        }
        return
      }
      return
    }

    // ─── result view ───
    if (view.kind === 'result') {
      if (key.return) backToList()
      return
    }
  })

  // ─── Render ───────────────────────────────────────────────────

  const header = (
    <Box marginBottom={1}>
      <Text bold color="claude">
        🔌 Providers
      </Text>
    </Box>
  )

  if (view.kind === 'list') {
    return (
      <Box flexDirection="column" paddingLeft={1}>
        {header}
        <Text dimColor>
          Connect the AI accounts you want Tau to use. Multiple providers
          can be active at once.
        </Text>
        <Box marginTop={1} flexDirection="column">
          {providerRows.map((provider, i) => {
            const isSelected = i === view.selectedIndex
            const isAddRow = provider === ADD_CUSTOM_SENTINEL
            const name = getManageableProviderName(provider)
            const prefix = isSelected ? '>' : ' '
            const badge = isAddRow
              ? chalk.dim('[ new ]')
              : provider === E2B_SECURITY_PROVIDER
                ? chalk.green(formatE2BSecurityBadge())
                : provider === 'ollama'
                  ? formatOllamaBadge(ollamaStatus)
                  : provider === 'lmstudio'
                    ? formatLmStudioBadge(lmStudioStatus)
                    : provider === 'gemini'
                      ? formatGeminiBadge()
                      : provider === 'firstParty'
                        ? formatBadge(getFirstPartyAuthState())
                        : formatBadge(getAuthState(provider))
            return (
              <Box key={provider}>
                <Text
                  bold={isSelected}
                  color={isSelected ? 'claude' : undefined}
                  dimColor={!isSelected}
                >
                  {prefix} {name.padEnd(24)}
                </Text>
                <Text> {badge}</Text>
              </Box>
            )
          })}
        </Box>
        <Box marginTop={1}>
          <Text dimColor>
            ↑↓ navigate · Enter to configure · Esc to close
          </Text>
        </Box>
      </Box>
    )
  }

  if (view.kind === 'configure') {
    const provider = view.provider
    const name = getManageableProviderName(provider)
    const options = buildConfigureOptions(provider, ollamaStatus, lmStudioStatus)
    const badge =
      provider === E2B_SECURITY_PROVIDER
        ? chalk.green(formatE2BSecurityBadge())
        : provider === 'ollama'
        ? formatOllamaBadge(ollamaStatus)
        : provider === 'lmstudio'
        ? formatLmStudioBadge(lmStudioStatus)
        : provider === 'gemini'
          ? formatGeminiBadge()
          : provider === 'firstParty'
            ? formatBadge(getFirstPartyAuthState())
            : formatBadge(getAuthState(provider))
    const currentUrl =
      provider === 'ollama'
        ? getOllamaBaseUrl()
        : provider === 'lmstudio'
          ? getLmStudioBaseUrl()
          : isCustomProvider(provider)
            ? getCustomProvider(provider)?.baseUrl ?? null
            : null
    return (
      <Box flexDirection="column" paddingLeft={1}>
        {header}
        <Box>
          <Text bold>{name}</Text>
          <Text> {badge}</Text>
        </Box>
        {currentUrl && (
          <Text dimColor>Base URL: {currentUrl}</Text>
        )}
        <Box marginTop={1} flexDirection="column">
          {options.map((option, i) => {
            const isSelected = i === view.selectedIndex
            const prefix = isSelected ? '>' : ' '
            const label = labelConfigureOption(option, provider)
            return (
              <Text
                key={option.kind}
                bold={isSelected}
                color={isSelected ? 'claude' : undefined}
                dimColor={!isSelected}
              >
                {prefix} {label}
              </Text>
            )
          })}
        </Box>
        <Box marginTop={1}>
          <Text dimColor>
            ↑↓ navigate · Enter to select · Esc to go back
          </Text>
        </Box>
      </Box>
    )
  }

  if (view.kind === 'ollama_url_input') {
    return (
      <Box flexDirection="column" paddingLeft={1}>
        {header}
        <Text bold>Set Ollama base URL</Text>
        <Text dimColor>
          Default: <Text color="suggestion">{OLLAMA_DEFAULT_BASE}</Text>
        </Text>
        <Text dimColor>
          Accepts full URLs (http://host:port) or host:port shorthand.
        </Text>
        {view.error && (
          <Box marginTop={1}>
            <Text color="error">{view.error}</Text>
          </Box>
        )}
        <Box marginTop={1}>
          <Text>URL: </Text>
          <TextInput
            value={ollamaUrlInput}
            onChange={setOllamaUrlInput}
            onSubmit={handleOllamaUrlSubmit}
            placeholder="http://localhost:11434"
            focus={true}
            showCursor={true}
            columns={inputColumns}
            cursorOffset={ollamaUrlCursorOffset}
            onChangeCursorOffset={setOllamaUrlCursorOffset}
          />
        </Box>
        <Box marginTop={1}>
          <Text dimColor>Enter to submit · Esc to go back</Text>
        </Box>
      </Box>
    )
  }

  if (view.kind === 'lmstudio_url_input') {
    return (
      <Box flexDirection="column" paddingLeft={1}>
        {header}
        <Text bold>Set LM Studio base URL</Text>
        <Text dimColor>
          Default: <Text color="suggestion">{LMSTUDIO_DEFAULT_BASE}</Text>
        </Text>
        <Text dimColor>
          Accepts full URLs (http://host:port/v1) or host:port shorthand.
        </Text>
        {view.error && (
          <Box marginTop={1}>
            <Text color="error">{view.error}</Text>
          </Box>
        )}
        <Box marginTop={1}>
          <Text>URL: </Text>
          <TextInput
            value={lmStudioUrlInput}
            onChange={setLmStudioUrlInput}
            onSubmit={handleLmStudioUrlSubmit}
            placeholder="http://localhost:1234/v1"
            focus={true}
            showCursor={true}
            columns={inputColumns}
            cursorOffset={lmStudioUrlCursorOffset}
            onChangeCursorOffset={setLmStudioUrlCursorOffset}
          />
        </Box>
        <Box marginTop={1}>
          <Text dimColor>Enter to submit Â· Esc to go back</Text>
        </Box>
      </Box>
    )
  }

  if (view.kind === 'custom_form') {
    const isAdd = view.mode === 'add'
    const step = currentCustomStep(view)!
    const total = view.steps.length
    const title = isAdd ? 'Add custom provider' : `Edit "${view.id}"`

    const stepMeta: Record<
      CustomFormStep,
      { label: string; placeholder: string; hint?: string; mask?: string }
    > = {
      id: {
        label: 'Provider id',
        placeholder: 'myproxy',
        hint: 'lowercase letters, digits, "-" or "_" — env var MYPROXY_API_KEY',
      },
      name: { label: 'Display name', placeholder: 'My Internal AI Proxy' },
      protocol: { label: 'Protocol', placeholder: '' },
      baseUrl: {
        label: 'Base URL',
        placeholder: 'https://api.example.com/v1',
        hint: 'include /v1 if the server needs it',
      },
      apiKey: {
        label: 'API key',
        placeholder: '(optional — Enter to skip)',
        hint: 'stored locally; blank = skip (add) or keep existing (edit)',
        mask: '•',
      },
    }

    // On edit the field is pre-filled; Enter keeps it, so only the fields the
    // user wants to change need editing.
    const editHint =
      !isAdd && step !== 'protocol' && step !== 'apiKey'
        ? 'Editing a copy of the current value — Enter keeps it, or change it.'
        : undefined

    // Fields already collected, shown for context.
    const collected: Array<[string, string]> = []
    if (isAdd && view.draft.id) collected.push(['id', view.draft.id])
    if (view.draft.name) collected.push(['name', view.draft.name])
    if (view.steps.indexOf('protocol') < view.step) {
      collected.push(['protocol', view.draft.protocol])
    }
    if (view.draft.baseUrl) collected.push(['base URL', view.draft.baseUrl])

    return (
      <Box flexDirection="column" paddingLeft={1}>
        {header}
        <Text bold>{title}</Text>
        <Text dimColor>
          Step {view.step + 1} of {total}
          {!isAdd ? ' · id is fixed' : ''}
        </Text>
        {collected.length > 0 && (
          <Box marginTop={1} flexDirection="column">
            {collected.map(([k, v]) => (
              <Text key={k} dimColor>
                {k}: {v}
              </Text>
            ))}
          </Box>
        )}
        {view.error && (
          <Box marginTop={1}>
            <Text color="error">{view.error}</Text>
          </Box>
        )}
        {step === 'protocol' ? (
          <Box marginTop={1} flexDirection="column">
            <Text>Protocol:</Text>
            {(['openai', 'anthropic'] as const).map((p, i) => {
              const sel = i === view.protocolCursor
              return (
                <Text
                  key={p}
                  bold={sel}
                  color={sel ? 'claude' : undefined}
                  dimColor={!sel}
                >
                  {sel ? '>' : ' '} {p}
                  {p === 'openai' ? '  (Chat Completions)' : '  (Messages API)'}
                </Text>
              )
            })}
            <Box marginTop={1}>
              <Text dimColor>↑↓ select · Enter to continue · Esc to cancel</Text>
            </Box>
          </Box>
        ) : (
          <Box flexDirection="column">
            <Box marginTop={1}>
              <Text>{stepMeta[step].label}: </Text>
              <TextInput
                value={customFieldInput}
                onChange={setCustomFieldInput}
                onSubmit={handleCustomFieldSubmit}
                placeholder={stepMeta[step].placeholder}
                mask={stepMeta[step].mask}
                focus={true}
                showCursor={true}
                columns={inputColumns}
                cursorOffset={customFieldCursorOffset}
                onChangeCursorOffset={setCustomFieldCursorOffset}
              />
            </Box>
            {stepMeta[step].hint && <Text dimColor>{stepMeta[step].hint}</Text>}
            {editHint && <Text dimColor>{editHint}</Text>}
            <Box marginTop={1}>
              <Text dimColor>Enter to continue · Esc to cancel</Text>
            </Box>
          </Box>
        )}
      </Box>
    )
  }

  if (view.kind === 'custom_delete') {
    const name = getManageableProviderName(view.id)
    const options = [
      `Yes, remove "${name}"`,
      'Cancel',
    ]
    return (
      <Box flexDirection="column" paddingLeft={1}>
        {header}
        <Text bold>Remove custom provider</Text>
        <Text>
          Delete <Text bold>{name}</Text> from settings?
        </Text>
        <Text dimColor>
          This only edits your settings.json — stored API keys are kept.
        </Text>
        <Box marginTop={1} flexDirection="column">
          {options.map((label, i) => {
            const isSelected = i === view.selectedIndex
            return (
              <Text
                key={label}
                bold={isSelected}
                color={isSelected ? 'claude' : undefined}
                dimColor={!isSelected}
              >
                {isSelected ? '>' : ' '} {label}
              </Text>
            )
          })}
        </Box>
        <Box marginTop={1}>
          <Text dimColor>↑↓ navigate · Enter to select · Esc to go back</Text>
        </Box>
      </Box>
    )
  }

  if (view.kind === 'anthropic_login') {
    return <AnthropicLogin onDone={handleAnthropicLoginDone} />
  }

  if (view.kind === 'e2b_login') {
    return <E2BSecurityLogin onDone={handleE2BLoginDone} />
  }

  if (view.kind === 'provider_login') {
    const provider = view.provider
    return (
      <ProviderLoginFlow
        provider={provider as APIProvider}
        onDone={(success) => {
          refresh()
          setView({
            kind: 'result',
            provider,
            tone: success ? 'success' : 'error',
            message: success
              ? `${getManageableProviderName(provider)} credentials saved.`
              : `${getManageableProviderName(provider)} login cancelled.`,
          })
        }}
      />
    )
  }

  if (view.kind === 'result') {
    return (
      <Box flexDirection="column" paddingLeft={1}>
        {header}
        <Text color={view.tone === 'success' ? 'success' : 'error'}>
          {view.tone === 'success' ? '✓ ' : '✗ '}
          {view.message}
        </Text>
        <Box marginTop={1}>
          <Text dimColor>Enter to return to the provider list</Text>
        </Box>
      </Box>
    )
  }

  return null
}

// ─── Entry point ────────────────────────────────────────────────

export const call: LocalJSXCommandCall = async (onDone, context) => (
  <ProviderManager onDone={onDone} setMessages={context.setMessages} />
)
