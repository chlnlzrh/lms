import { NextRequest, NextResponse } from 'next/server'
import { buildSearchIndex } from '@/lib/search-index-builder'
import { calculateRelevanceScore, applyFilters, type SearchResult, type SearchFilters } from '@/lib/search'

// Cache the search index in memory (rebuilds on server restart)
let searchIndex: SearchResult[] = []
let indexBuilt = false

async function getSearchIndex(): Promise<SearchResult[]> {
  if (!indexBuilt) {
    searchIndex = await buildSearchIndex()
    indexBuilt = true
  }
  return searchIndex
}

// Main search function (server-side)
async function performSearch(
  query: string, 
  filters: SearchFilters = { tracks: [], difficulty: [], contentType: [], duration: [] }
): Promise<SearchResult[]> {
  if (!query.trim()) {
    return []
  }

  const index = await getSearchIndex()

  // Calculate relevance scores
  const results = index
    .map(item => ({
      ...item,
      relevanceScore: calculateRelevanceScore(item, query)
    }))
    .filter(item => item.relevanceScore > 0)

  // Apply filters
  const filteredResults = applyFilters(results, filters)

  // Sort by relevance score
  const sortedResults = filteredResults.sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0))

  return sortedResults
}

// Get search suggestions (server-side)
async function performSuggestions(query: string): Promise<string[]> {
  if (!query.trim() || query.length < 2) {
    return []
  }

  const index = await getSearchIndex()
  const suggestions = new Set<string>()

  // Get title-based suggestions
  index.forEach(item => {
    if (item.title.toLowerCase().includes(query.toLowerCase())) {
      suggestions.add(item.title)
    }
    
    // Get keyword suggestions
    item.keywords.forEach(keyword => {
      if (keyword.toLowerCase().includes(query.toLowerCase())) {
        suggestions.add(keyword)
      }
    })
  })

  return Array.from(suggestions).slice(0, 10)
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const query = searchParams.get('q') || ''
    const type = searchParams.get('type') || 'search' // 'search' or 'suggestions'
    
    // Parse filters
    const tracks = searchParams.get('tracks')?.split(',').filter(Boolean) || []
    const difficulty = searchParams.get('difficulty')?.split(',').filter(Boolean) || []
    const contentType = searchParams.get('contentType')?.split(',').filter(Boolean) || []
    const duration = searchParams.get('duration')?.split(',').filter(Boolean) || []

    const filters = {
      tracks,
      difficulty,
      contentType,
      duration
    }

    if (type === 'suggestions') {
      const suggestions = await performSuggestions(query)
      return NextResponse.json({ suggestions })
    }

    const results = await performSearch(query, filters)
    
    return NextResponse.json({
      query,
      results,
      totalCount: results.length,
      filters: filters
    })

  } catch (error) {
    console.error('Search API error:', error)
    return NextResponse.json(
      { error: 'Search temporarily unavailable' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const { query, filters } = await request.json()

    if (!query) {
      return NextResponse.json(
        { error: 'Query is required' },
        { status: 400 }
      )
    }

    const results = await performSearch(query, filters || {
      tracks: [],
      difficulty: [],
      contentType: [],
      duration: []
    })

    return NextResponse.json({
      query,
      results,
      totalCount: results.length,
      filters: filters || {}
    })

  } catch (error) {
    console.error('Search API error:', error)
    return NextResponse.json(
      { error: 'Search temporarily unavailable' },
      { status: 500 }
    )
  }
}