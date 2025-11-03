// Client-safe search functionality

export interface SearchResult {
  id: string
  title: string
  description: string
  href: string
  track: string
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  duration: string
  type: 'lesson' | 'module' | 'track'
  keywords: string[]
  relevanceScore?: number
  progress?: number
  aiExplanation?: string
}

export interface SearchFilters {
  tracks: string[]
  difficulty: string[]
  contentType: string[]
  duration: string[]
}

// Client-side search function that calls the API
export async function searchLessons(
  query: string, 
  filters: SearchFilters = { tracks: [], difficulty: [], contentType: [], duration: [] }
): Promise<SearchResult[]> {
  if (!query.trim()) {
    return []
  }

  try {
    const response = await fetch('/api/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, filters })
    })

    if (!response.ok) {
      throw new Error(`Search failed: ${response.status}`)
    }

    const data = await response.json()
    return data.results || []
  } catch (error) {
    console.error('Search error:', error)
    return []
  }
}

// Client-side suggestions function
export async function getSearchSuggestions(query: string): Promise<string[]> {
  if (!query.trim() || query.length < 2) {
    return []
  }

  try {
    const response = await fetch(`/api/search?type=suggestions&q=${encodeURIComponent(query)}`)
    
    if (!response.ok) {
      throw new Error(`Suggestions failed: ${response.status}`)
    }

    const data = await response.json()
    return data.suggestions || []
  } catch (error) {
    console.error('Suggestions error:', error)
    return []
  }
}

// Simple text matching function (for client-side use)
export function calculateRelevanceScore(item: SearchResult, query: string): number {
  const searchQuery = query.toLowerCase()
  let score = 0

  // Title match (highest weight)
  if (item.title.toLowerCase().includes(searchQuery)) {
    score += 0.5
  }

  // Description match
  if (item.description.toLowerCase().includes(searchQuery)) {
    score += 0.3
  }

  // Keywords match
  const keywordMatches = item.keywords.filter(keyword => 
    keyword.toLowerCase().includes(searchQuery) || searchQuery.includes(keyword.toLowerCase())
  ).length
  score += keywordMatches * 0.1

  // Track match
  if (item.track.toLowerCase().includes(searchQuery)) {
    score += 0.2
  }

  // Exact word matches get bonus
  const words = searchQuery.split(' ')
  const titleWords = item.title.toLowerCase().split(' ')
  const exactMatches = words.filter(word => titleWords.includes(word)).length
  score += exactMatches * 0.1

  return Math.min(score, 1) // Cap at 1.0
}

// Apply filters to results (for client-side use)
export function applyFilters(results: SearchResult[], filters: SearchFilters): SearchResult[] {
  return results.filter(result => {
    // Track filter
    if (filters.tracks.length > 0 && !filters.tracks.includes(result.track)) {
      return false
    }

    // Difficulty filter
    if (filters.difficulty.length > 0 && !filters.difficulty.includes(result.difficulty)) {
      return false
    }

    // Content type filter
    if (filters.contentType.length > 0 && !filters.contentType.includes(result.type)) {
      return false
    }

    // Duration filter (simplified)
    if (filters.duration.length > 0) {
      const duration = result.duration.toLowerCase()
      const hasShort = filters.duration.includes('short') && (duration.includes('min') && !duration.includes('hour'))
      const hasMedium = filters.duration.includes('medium') && (duration.includes('hour') || duration.includes('60'))
      const hasLong = filters.duration.includes('long') && (duration.includes('hour') && !duration.includes('30'))
      
      if (!hasShort && !hasMedium && !hasLong) {
        return false
      }
    }

    return true
  })
}