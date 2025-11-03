'use client'

import { useState, useEffect, useCallback } from 'react'
import { Icons } from '@/components/ui/icons'
import { SearchFilters } from './search-filters'
import { SearchResults } from './search-results'
import { SearchInput } from './search-input'
import { useDebounce } from '@/hooks/use-debounce'
import { searchLessons, type SearchResult, type SearchFilters as SearchFiltersType } from '@/lib/search'

interface SearchInterfaceProps {
  className?: string
  placeholder?: string
  showFilters?: boolean
}

export function SearchInterface({ 
  className = '', 
  placeholder = 'Search lessons, topics, or ask a question...',
  showFilters = true 
}: SearchInterfaceProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [filters, setFilters] = useState<SearchFiltersType>({
    tracks: [],
    difficulty: [],
    contentType: [],
    duration: []
  })
  const [isLoading, setIsLoading] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [aiEnabled, setAiEnabled] = useState(false)
  const [aiInsights, setAiInsights] = useState<string>('')
  const [recentSearches, setRecentSearches] = useState<string[]>([])

  const debouncedQuery = useDebounce(query, 300)

  // Load recent searches from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('recent-searches')
    if (stored) {
      setRecentSearches(JSON.parse(stored))
    }
  }, [])

  // Save recent searches to localStorage
  const saveRecentSearch = useCallback((searchQuery: string) => {
    if (!searchQuery.trim()) return
    
    setRecentSearches(current => {
      const updated = [searchQuery, ...current.filter(s => s !== searchQuery)].slice(0, 10)
      localStorage.setItem('recent-searches', JSON.stringify(updated))
      return updated
    })
  }, [])

  // Perform search when query or filters change
  useEffect(() => {
    const performSearch = async () => {
      if (!debouncedQuery.trim()) {
        setResults([])
        setShowResults(false)
        setAiInsights('')
        return
      }

      setIsLoading(true)
      setShowResults(true)

      try {
        // Minimum loading time to prevent rapid flashing
        const searchPromise = searchLessons(debouncedQuery, filters)
        const minLoadTime = new Promise(resolve => setTimeout(resolve, 300))
        
        const [searchResults] = await Promise.all([searchPromise, minLoadTime])
        
        // Only update if we're still searching for the same query
        if (debouncedQuery === query.trim()) {
          setResults(searchResults)

          // AI enhancement if enabled
          if (aiEnabled && searchResults.length > 0) {
            try {
              const aiResponse = await fetch('/api/search/ai-enhance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: debouncedQuery, results: searchResults.slice(0, 5) })
              })
              
              if (aiResponse.ok) {
                const data = await aiResponse.json()
                setAiInsights(data.insights || '')
              }
            } catch (aiError) {
              console.warn('AI enhancement failed:', aiError)
              setAiInsights('')
            }
          } else {
            setAiInsights('')
          }

          // Save to recent searches
          saveRecentSearch(debouncedQuery)
        }
      } catch (error) {
        console.error('Search error:', error)
        setResults([])
        setAiInsights('')
      } finally {
        setIsLoading(false)
      }
    }

    performSearch()
  }, [debouncedQuery, filters, aiEnabled, saveRecentSearch, query])

  const handleQueryChange = (newQuery: string) => {
    setQuery(newQuery)
  }

  const handleFilterChange = (newFilters: SearchFiltersType) => {
    setFilters(newFilters)
  }

  const clearSearch = () => {
    setQuery('')
    setResults([])
    setShowResults(false)
    setAiInsights('')
  }

  const popularSearches = [
    'prompt engineering',
    'SQL fundamentals', 
    'Claude Code',
    'data modeling',
    'Snowflake optimization',
    'machine learning basics'
  ]

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Search Input */}
      <SearchInput
        value={query}
        onChange={handleQueryChange}
        placeholder={placeholder}
        aiEnabled={aiEnabled}
        onAiToggle={setAiEnabled}
        isLoading={isLoading}
        recentSearches={recentSearches}
        onClear={clearSearch}
      />

      {/* AI Enhancement Notice */}
      {aiEnabled && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Icons.Bot className="w-4 h-4 text-blue-600" />
            <span className="text-xs text-blue-800 dark:text-blue-200">
              AI-enhanced search is enabled. Results include semantic understanding and learning path recommendations.
            </span>
          </div>
        </div>
      )}

      {/* Search Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Filters Sidebar */}
        {showFilters && (
          <div className="lg:col-span-1">
            <SearchFilters 
              filters={filters}
              onChange={handleFilterChange}
              resultCount={results.length}
            />
          </div>
        )}

        {/* Main Content */}
        <div className={showFilters ? "lg:col-span-3" : "lg:col-span-4"}>
          {!showResults ? (
            // Zero State
            <div className="space-y-6">
              {/* Popular Searches */}
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h3 className="text-sm font-bold text-gray-900 dark:text-white mb-4">
                  Popular Searches
                </h3>
                <div className="flex flex-wrap gap-2">
                  {popularSearches.map((term) => (
                    <button
                      key={term}
                      onClick={() => setQuery(term)}
                      className="px-3 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                    >
                      {term}
                    </button>
                  ))}
                </div>
              </div>

              {/* Search Tips */}
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h3 className="text-sm font-bold text-gray-900 dark:text-white mb-4">
                  Search Tips
                </h3>
                <div className="space-y-3 text-xs text-gray-600 dark:text-gray-300">
                  <div>
                    <strong>Try specific terms:</strong> "prompt engineering", "SQL joins", "data modeling"
                  </div>
                  <div>
                    <strong>Search by tools:</strong> "Claude", "Snowflake", "dbt", "Python"
                  </div>
                  <div>
                    <strong>Ask questions:</strong> "How to optimize queries?" or "What is machine learning?"
                  </div>
                  <div>
                    <strong>Use filters:</strong> Filter by track, difficulty, or content type
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // Search Results Container - Stable height to prevent layout shifts
            <div className="min-h-screen">
              <SearchResults 
                results={results}
                query={query}
                isLoading={isLoading}
                aiInsights={aiInsights}
                aiEnabled={aiEnabled}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}