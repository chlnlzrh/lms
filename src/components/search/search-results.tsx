'use client'

import { useState } from 'react'
import { Icons } from '@/components/ui/icons'
import Link from 'next/link'
import { type SearchResult } from '@/lib/search'

interface SearchResultsProps {
  results: SearchResult[]
  query: string
  isLoading: boolean
  aiInsights?: string
  aiEnabled: boolean
}

export function SearchResults({ 
  results, 
  query, 
  isLoading, 
  aiInsights, 
  aiEnabled 
}: SearchResultsProps) {
  const [currentPage, setCurrentPage] = useState(1)
  const [sortBy, setSortBy] = useState<'relevance' | 'title' | 'difficulty' | 'duration'>('relevance')
  
  const resultsPerPage = 10
  const totalPages = Math.ceil(results.length / resultsPerPage)
  const startIndex = (currentPage - 1) * resultsPerPage
  const endIndex = startIndex + resultsPerPage
  const currentResults = results.slice(startIndex, endIndex)

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'beginner': return 'text-green-600 bg-green-100 dark:bg-green-900/30'
      case 'intermediate': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30'
      case 'advanced': return 'text-red-600 bg-red-100 dark:bg-red-900/30'
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30'
    }
  }

  const getTrackIcon = (track: string) => {
    const trackIcons: { [key: string]: any } = {
      ai: Icons.Bot,
      data_engineer: Icons.Database,
      saas: Icons.Cloud,
      sfdc: Icons.Zap,
      snowflake_tune: Icons.Snowflake,
      workato: Icons.Link,
      ba: Icons.BarChart3,
      data_gov: Icons.Shield,
      devops_engineer: Icons.Settings,
      finance: Icons.DollarSign,
      hr: Icons.Users,
      qa: Icons.CheckCircle,
      rpa: Icons.Bot
    }
    return trackIcons[track] || Icons.BookOpen
  }

  const highlightText = (text: string, query: string) => {
    if (!query) return text
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
    return text.split(regex).map((part, index) => 
      regex.test(part) ? (
        <mark key={index} className="bg-yellow-200 dark:bg-yellow-800 text-gray-900 dark:text-white">
          {part}
        </mark>
      ) : part
    )
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="animate-pulse space-y-3">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
              <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
              <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
              <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
            </div>
          </div>
        ))}
      </div>
    )
  }

  if (results.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8 text-center">
        <Icons.Search className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          No results found
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
          We couldn't find any content matching "{query}". Try:
        </p>
        <ul className="text-sm text-gray-600 dark:text-gray-300 space-y-1">
          <li>• Checking your spelling</li>
          <li>• Using different keywords</li>
          <li>• Being more general in your search</li>
          <li>• Removing filters to broaden results</li>
        </ul>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* AI Insights */}
      {aiEnabled && aiInsights && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <Icons.Bot className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-2">
                AI Insights
              </h4>
              <p className="text-sm text-blue-800 dark:text-blue-200">
                {aiInsights}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Results Header */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-600 dark:text-gray-300">
          Showing {startIndex + 1}-{Math.min(endIndex, results.length)} of {results.length} results for "{query}"
        </div>
        
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-600 dark:text-gray-300">Sort by:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="text-xs border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
          >
            <option value="relevance">Relevance</option>
            <option value="title">Title</option>
            <option value="difficulty">Difficulty</option>
            <option value="duration">Duration</option>
          </select>
        </div>
      </div>

      {/* Results List */}
      <div className="space-y-4">
        {currentResults.map((result, index) => {
          const IconComponent = getTrackIcon(result.track)
          
          return (
            <div
              key={`${result.id}-${index}`}
              className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow"
            >
              {/* Result Header */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <Link
                    href={result.href}
                    className="text-lg font-medium text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 line-clamp-2"
                  >
                    {highlightText(result.title, query)}
                  </Link>
                  
                  <div className="flex items-center space-x-3 mt-2">
                    {/* Track Badge */}
                    <div className="flex items-center space-x-1">
                      <IconComponent className="w-3 h-3 text-gray-500" />
                      <span className="text-xs text-gray-600 dark:text-gray-300 capitalize">
                        {result.track.replace('_', ' ')}
                      </span>
                    </div>
                    
                    {/* Difficulty Badge */}
                    <span className={`px-2 py-0.5 text-xs rounded-full ${getDifficultyColor(result.difficulty)}`}>
                      {result.difficulty}
                    </span>
                    
                    {/* Duration */}
                    <div className="flex items-center space-x-1">
                      <Icons.Clock className="w-3 h-3 text-gray-400" />
                      <span className="text-xs text-gray-600 dark:text-gray-300">
                        {result.duration}
                      </span>
                    </div>
                    
                    {/* Content Type */}
                    <span className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 px-2 py-0.5 rounded capitalize">
                      {result.type}
                    </span>
                  </div>
                </div>
                
                {/* Bookmark Button */}
                <button className="p-1 text-gray-400 hover:text-yellow-600 dark:hover:text-yellow-400">
                  <Icons.Bookmark className="w-4 h-4" />
                </button>
              </div>

              {/* Description */}
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3 line-clamp-3">
                {highlightText(result.description, query)}
              </p>

              {/* AI Match Explanation */}
              {aiEnabled && result.aiExplanation && (
                <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded p-3 mb-3">
                  <div className="flex items-start space-x-2">
                    <Icons.Bot className="w-3 h-3 text-blue-600 flex-shrink-0 mt-0.5" />
                    <p className="text-xs text-blue-800 dark:text-blue-200">
                      <strong>Why this matches:</strong> {result.aiExplanation}
                    </p>
                  </div>
                </div>
              )}

              {/* Keywords */}
              {result.keywords && result.keywords.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-3">
                  {result.keywords.slice(0, 5).map((keyword, i) => (
                    <span
                      key={i}
                      className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 px-2 py-0.5 rounded"
                    >
                      {highlightText(keyword, query)}
                    </span>
                  ))}
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center justify-between pt-3 border-t border-gray-100 dark:border-gray-700">
                <div className="flex items-center space-x-3">
                  <Link
                    href={result.href}
                    className="text-xs bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 transition-colors"
                  >
                    View {result.type}
                  </Link>
                  
                  {result.progress && (
                    <div className="flex items-center space-x-1">
                      <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                        <div 
                          className="bg-green-600 h-1 rounded-full" 
                          style={{ width: `${result.progress}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-300">
                        {result.progress}%
                      </span>
                    </div>
                  )}
                </div>
                
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  Relevance: {Math.round((result.relevanceScore || 0) * 100)}%
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center space-x-2">
          <button
            onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
            disabled={currentPage === 1}
            className="px-3 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            Previous
          </button>
          
          <div className="flex space-x-1">
            {[...Array(totalPages)].map((_, i) => {
              const page = i + 1
              const isCurrentPage = page === currentPage
              
              if (totalPages > 7 && Math.abs(page - currentPage) > 2 && page !== 1 && page !== totalPages) {
                return null
              }
              
              return (
                <button
                  key={page}
                  onClick={() => setCurrentPage(page)}
                  className={`px-2 py-1 text-xs rounded ${
                    isCurrentPage
                      ? 'bg-blue-600 text-white'
                      : 'border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
                  }`}
                >
                  {page}
                </button>
              )
            })}
          </div>
          
          <button
            onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage === totalPages}
            className="px-3 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}