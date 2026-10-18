/**
 * Custom provider support — user-defined LLM providers.
 *
 * Custom providers are declared in settings.json under `customProviders`.
 * They support two wire protocols:
 *   - 'anthropic': server speaks the Anthropic Messages API. Routed through
 *     the real Anthropic SDK pointed at the custom base URL (same as agentrouter).
 *   - 'openai': server speaks OpenAI Chat Completions. Registered with the
 *     openai-compat lane using the generic transformer.
 *
 * Security: base URLs are exfiltration vectors. This module reads ONLY from
 * trusted sources: userSettings and policySettings. Project / local settings
 * are silently ignored.
 *
 * API keys are NOT stored in the settings file. They follow the standard
 * provider-key flow: env var `<ID>_API_KEY` (uppercased id) → stored key
 * written by `/login`.
 */

import {
  getSettingsForSource,
  updateSettingsForSource,
} from './settings/settings.js'
import { loadProviderKey } from '../services/api/auth/api_key_manager.js'
import { API_PROVIDERS } from './model/providerRegistry.js'

/**
 * Built-in provider ids a custom provider must not shadow. A colliding id would
 * re-route the built-in provider through the custom branch and, because keys
 * share one flat store (`provider-keys.json`) and one env-var scheme, would
 * leak the built-in's stored/env credential to the custom base URL.
 */
const RESERVED_PROVIDER_IDS = new Set<string>([...API_PROVIDERS, 'firstParty'])

/** True when `id` collides with a built-in provider id. */
export function isReservedProviderId(id: string): boolean {
  return RESERVED_PROVIDER_IDS.has(id)
}

export interface CustomProvider {
  /** Canonical id — lowercase alphanumeric, hyphens, underscores. */
  id: string
  /** Human-readable display name. */
  name: string
  /** Wire protocol the server speaks. */
  protocol: 'anthropic' | 'openai'
  /** Full base URL of the API. */
  baseUrl: string
  /** Optional manual model catalog, used as fallback. */
  models?: Array<{ id: string; name?: string }>
}

/** Cached list of custom providers — reset on settings reload. */
let _cachedCustomProviders: CustomProvider[] | null = null

/**
 * Return all custom providers from trusted settings sources.
 * Results are cached for the process lifetime unless resetCustomProviderCache()
 * is called (e.g. after /provider or settings changes).
 */
export function getCustomProviders(): readonly CustomProvider[] {
  if (_cachedCustomProviders !== null) return _cachedCustomProviders

  const providers: CustomProvider[] = []
  const seen = new Set<string>()

  // Read only trusted sources — userSettings first, then policySettings.
  for (const source of ['userSettings', 'policySettings'] as const) {
    const settings = getSettingsForSource(source)
    const customProviders = settings?.customProviders
    if (!Array.isArray(customProviders)) continue

    for (const entry of customProviders) {
      if (!entry || typeof entry !== 'object') continue
      const { id, name, protocol, baseUrl, models } = entry as CustomProvider
      if (!id || !name || !protocol || !baseUrl) continue
      // Defense in depth: a reserved id would shadow a built-in provider and
      // could reuse its stored/env credential. Ignore such entries entirely.
      if (isReservedProviderId(id)) continue
      if (seen.has(id)) continue  // first source wins on collision
      seen.add(id)
      providers.push({ id, name, protocol, baseUrl, models })
    }
  }

  _cachedCustomProviders = providers
  return providers
}

/** Reset the cache so the next getCustomProviders() call re-reads settings. */
export function resetCustomProviderCache(): void {
  _cachedCustomProviders = null
}

/** Returns true when the given id matches a user-defined custom provider. */
export function isCustomProvider(id: string): boolean {
  return getCustomProviders().some(p => p.id === id)
}

/** Returns the custom provider entry for the given id, or null. */
export function getCustomProvider(id: string): CustomProvider | null {
  return getCustomProviders().find(p => p.id === id) ?? null
}

/**
 * True when the id resolves to a custom provider that is declared ONLY in
 * policySettings (managed) and not in userSettings. Such providers are shown
 * and activatable but cannot be edited or removed locally.
 */
export function isPolicyOnlyCustomProvider(id: string): boolean {
  if (!isCustomProvider(id)) return false
  return !getUserCustomProviders().some(p => p.id === id)
}

/**
 * Returns the API key for a custom provider.
 * Lookup order: env var `<ID_UPPERCASED>_API_KEY` → stored provider key.
 */
export function getCustomProviderApiKey(id: string): string | null {
  // env var: e.g. id='myproxy' → MYPROXY_API_KEY, id='my-proxy' → MY_PROXY_API_KEY
  const envKey = `${id.toUpperCase().replace(/-/g, '_')}_API_KEY`
  if (process.env[envKey]) return process.env[envKey] ?? null
  // stored key via /login — uses the same per-provider JSON file as builtins
  return loadProviderKey(id)
}

/**
 * Returns the base URL for a custom provider, looking up env var override first.
 * Env var: `<ID_UPPERCASED>_BASE_URL`, e.g. MYPROXY_BASE_URL.
 */
export function getCustomProviderBaseUrl(id: string): string | null {
  const envKey = `${id.toUpperCase().replace(/-/g, '_')}_BASE_URL`
  if (process.env[envKey]) return process.env[envKey] ?? null
  return getCustomProvider(id)?.baseUrl ?? null
}

// ─── Mutation (used by the /provider wizard) ──────────────────────
//
// Only the userSettings source is editable. policySettings is managed
// (admin/enterprise) and read-only here.

/** The custom providers as declared directly in userSettings (unmerged). */
export function getUserCustomProviders(): CustomProvider[] {
  const settings = getSettingsForSource('userSettings')
  const list = settings?.customProviders
  if (!Array.isArray(list)) return []
  return list.filter(
    (entry): entry is CustomProvider =>
      !!entry &&
      typeof entry === 'object' &&
      typeof (entry as CustomProvider).id === 'string' &&
      typeof (entry as CustomProvider).name === 'string' &&
      typeof (entry as CustomProvider).protocol === 'string' &&
      typeof (entry as CustomProvider).baseUrl === 'string' &&
      !isReservedProviderId((entry as CustomProvider).id),
  )
}

/**
 * Persist the given list as userSettings.customProviders (replacing the
 * existing array) and drop the in-memory cache so the next read sees it.
 * Returns an error message on failure, or null on success.
 */
export function saveUserCustomProviders(
  providers: CustomProvider[],
): string | null {
  const { error } = updateSettingsForSource('userSettings', {
    customProviders: providers,
  })
  if (error) return error.message
  resetCustomProviderCache()
  return null
}

/** Add a new custom provider. Errors if the id is already taken or reserved. */
export function addUserCustomProvider(provider: CustomProvider): string | null {
  if (isReservedProviderId(provider.id)) {
    return `"${provider.id}" is a built-in provider id and cannot be used for a custom provider.`
  }
  const current = getUserCustomProviders()
  if (current.some(p => p.id === provider.id)) {
    return `A custom provider with id "${provider.id}" already exists.`
  }
  return saveUserCustomProviders([...current, provider])
}

/**
 * Update an existing custom provider (matched by id). The id itself is
 * immutable here — change the name/protocol/baseUrl only.
 */
export function updateUserCustomProvider(
  id: string,
  patch: Omit<CustomProvider, 'id'>,
): string | null {
  const current = getUserCustomProviders()
  if (!current.some(p => p.id === id)) {
    return `No user custom provider with id "${id}".`
  }
  return saveUserCustomProviders(
    current.map(p => (p.id === id ? { id, ...patch } : p)),
  )
}

/** Remove a custom provider by id. */
export function removeUserCustomProvider(id: string): string | null {
  const current = getUserCustomProviders()
  return saveUserCustomProviders(current.filter(p => p.id !== id))
}
