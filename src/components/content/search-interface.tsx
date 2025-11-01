'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { Icons } from '@/components/ui/icons'
import { cn } from '@/lib/utils'
import { contentClient } from '@/lib/content-client'
import { SearchIndexEntry } from '@/types/content'
import { progressTracker } from '@/lib/progress-tracker'
import Link from 'next/link'

interface SearchFilters {
  track?: 'ai' | 'data-engineering'
  type?: 'lesson' | 'module'
  complexity?: 'F' | 'I' | 'A'
}

interface SearchInterfaceProps {
  className?: string
  onResultSelect?: (result: SearchIndexEntry) => void
  placeholder?: string
  showFilters?: boolean
}

export function SearchInterface({ 
  className, 
  onResultSelect, 
  placeholder = "Search lessons, modules, tracks...",
  showFilters = true 
}: SearchInterfaceProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchIndexEntry[]>([])
  const [filters, setFilters] = useState<SearchFilters>({})
  const [isLoading, setIsLoading] = useState(false)
  const [isOpen, setIsOpen] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(-1)

  // Debounced search function
  const debouncedSearch = useCallback(
    debounce(async (searchQuery: string, searchFilters: SearchFilters) => {
      if (searchQuery.trim().length < 2) {
        setResults([])
        setIsLoading(false)
        return
      }

      try {
        const searchResults = await contentClient.searchContent(searchQuery, searchFilters)
        setResults(searchResults)
      } catch (error) {
        console.error('Search error:', error)
        setResults([])
      } finally {
        setIsLoading(false)
      }
    }, 300),
    []
  )

  // Search when query or filters change
  useEffect(() => {
    if (query.trim().length >= 2) {
      setIsLoading(true)
      debouncedSearch(query, filters)
    } else {
      setResults([])
      setIsLoading(false)
    }
  }, [query, filters, debouncedSearch])

  // Enhanced results with progress information
  const enhancedResults = useMemo(() => {
    const userProgress = progressTracker.getUserProgress()
    
    return results.map(result => ({
      ...result,
      isCompleted: result.type === 'lesson' 
        ? userProgress.lessons[result.id]?.isCompleted || false
        : false,
      isBookmarked: result.type === 'lesson' 
        ? userProgress.lessons[result.id]?.bookmarked || false
        : false,
      lastAccessed: result.type === 'lesson'
        ? userProgress.lessons[result.id]?.lastAccessedAt
        : undefined
    }))
  }, [results])

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault()
          setSelectedIndex(prev => 
            prev < enhancedResults.length - 1 ? prev + 1 : prev
          )
          break
        case 'ArrowUp':
          e.preventDefault()
          setSelectedIndex(prev => prev > 0 ? prev - 1 : prev)
          break
        case 'Enter':
          e.preventDefault()
          if (selectedIndex >= 0 && enhancedResults[selectedIndex]) {
            handleResultSelect(enhancedResults[selectedIndex])
          }
          break
        case 'Escape':
          setIsOpen(false)
          setSelectedIndex(-1)
          break
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, selectedIndex, enhancedResults])

  const handleResultSelect = (result: SearchIndexEntry) => {
    setIsOpen(false)
    setSelectedIndex(-1)
    
    if (onResultSelect) {
      onResultSelect(result)
    } else {
      // Navigate to result
      const url = result.type === 'lesson' 
        ? `/talent-development/${result.track}/lesson/${result.id.split('-').slice(1).join('-')}`
        : `/talent-development/${result.track}/module-${result.id.split('-').slice(-1)[0]}`
      
      window.location.href = url
    }
  }

  const getResultIcon = (result: SearchIndexEntry) => {
    if (result.type === 'module') return Icons.BookOpen
    return result.track === 'ai' ? Icons.Bot : Icons.Database
  }

  const getResultUrl = (result: SearchIndexEntry) => {
    if (result.type === 'lesson') {
      const slug = result.id.split('-').slice(1).join('-')
      return `/talent-development/${result.track}/lesson/${slug}`
    } else {
      const moduleNumber = result.id.split('-').slice(-1)[0]
      return `/talent-development/${result.track}/module-${moduleNumber}`
    }
  }

  return (
    <div className={cn("relative", className)}>
      {/* Search Input */}
      <div className="relative">
        <Icons.Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setIsOpen(true)}
          className={cn(
            "w-full pl-10 pr-4 py-2 text-xs",
            "bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md",
            "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent",
            "placeholder-gray-400"
          )}
        />
        
        {isLoading && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
            <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
          </div>
        )}
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="flex items-center space-x-2 mt-2">
          <select
            value={filters.track || ''}
            onChange={(e) => setFilters(prev => ({ 
              ...prev, 
              track: e.target.value as 'ai' | 'data-engineering' | undefined 
            }))}
            className="text-xs border border-gray-200 dark:border-gray-700 rounded px-2 py-1 bg-white dark:bg-gray-800"
          >
            <option value="">All Tracks</option>
            <option value="ai">AI Training</option>
            <option value="data-engineering">Data Engineering</option>
          </select>
          
          <select
            value={filters.type || ''}
            onChange={(e) => setFilters(prev => ({ 
              ...prev, 
              type: e.target.value as 'lesson' | 'module' | undefined 
            }))}
            className="text-xs border border-gray-200 dark:border-gray-700 rounded px-2 py-1 bg-white dark:bg-gray-800"
          >
            <option value="">All Types</option>
            <option value="lesson">Lessons</option>
            <option value="module">Modules</option>
          </select>
          
          <select
            value={filters.complexity || ''}
            onChange={(e) => setFilters(prev => ({ 
              ...prev, 
              complexity: e.target.value as 'F' | 'I' | 'A' | undefined 
            }))}
            className="text-xs border border-gray-200 dark:border-gray-700 rounded px-2 py-1 bg-white dark:bg-gray-800"
          >
            <option value="">All Levels</option>
            <option value="F">Foundation</option>
            <option value="I">Intermediate</option>
            <option value="A">Advanced</option>
          </select>
        </div>
      )}

      {/* Results Dropdown */}
      {isOpen && (query.trim().length >= 2 || results.length > 0) && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg z-50 max-h-96 overflow-y-auto">
          {enhancedResults.length > 0 ? (
            <div className="py-2">
              <div className="px-3 py-1 text-xs text-gray-500 dark:text-gray-400 border-b border-gray-100 dark:border-gray-700">
                {enhancedResults.length} result{enhancedResults.length !== 1 ? 's' : ''} found
              </div>
              
              {enhancedResults.map((result, index) => {
                const IconComponent = getResultIcon(result)
                const isSelected = index === selectedIndex
                
                return (
                  <Link
                    key={result.id}
                    href={getResultUrl(result)}
                    className={cn(
                      "block px-3 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer border-b border-gray-50 dark:border-gray-700 last:border-b-0",
                      isSelected && "bg-blue-50 dark:bg-blue-900/20"
                    )}
                    onClick={() => handleResultSelect(result)}
                  >
                    <div className="flex items-start space-x-3">
                      <IconComponent className={cn(
                        "w-4 h-4 mt-0.5 flex-shrink-0",
                        result.track === 'ai' ? "text-blue-600" : "text-green-600"
                      )} />
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 mb-1">
                          <h4 className="text-xs font-normal text-gray-900 dark:text-white truncate">
                            {result.title}
                          </h4>
                          
                          <div className="flex items-center space-x-1">
                            {result.isCompleted && (
                              <Icons.Circle className="w-3 h-3 text-green-600 fill-current" />
                            )}
                            {result.isBookmarked && (
                              <Icons.Bookmark className="w-3 h-3 text-yellow-600 fill-current" />
                            )}
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2 text-xs text-gray-500 dark:text-gray-400">
                          <span className={cn(
                            "px-1.5 py-0.5 rounded text-xs",
                            result.type === 'lesson' 
                              ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
                              : "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300"
                          )}>
                            {result.type}
                          </span>
                          
                          <span>{result.track === 'ai' ? 'AI Training' : 'Data Engineering'}</span>
                          <span>•</span>
                          <span>{result.module}</span>
                        </div>
                        
                        <p className="text-xs text-gray-600 dark:text-gray-300 mt-1 line-clamp-2">
                          {result.content.substring(0, 120)}...
                        </p>
                      </div>
                    </div>
                  </Link>
                )
              })}
            </div>
          ) : query.trim().length >= 2 ? (
            <div className="px-3 py-4 text-xs text-gray-500 dark:text-gray-400 text-center">
              {isLoading ? 'Searching...' : 'No results found'}
            </div>
          ) : null}
        </div>
      )}

      {/* Backdrop to close dropdown */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  )
}

// Utility function for debouncing
function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout
  return (...args: Parameters<T>) => {
    clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}