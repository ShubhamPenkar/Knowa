const KEY = 'knowa.lastProjectId'

export function getLastProjectId() {
  try {
    return localStorage.getItem(KEY) || ''
  } catch {
    return ''
  }
}

export function setLastProjectId(id) {
  try {
    if (id) localStorage.setItem(KEY, id)
  } catch {
    /* ignore */
  }
}
