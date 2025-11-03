'use client'

import { useState, useRef, useEffect } from 'react'
import { Icons } from '@/components/ui/icons'

interface SearchInputProps {
  value: string
  onChange: (value: string) => void
  placeholder: string
  aiEnabled: boolean
  onAiToggle: (enabled: boolean) => void
  isLoading: boolean
  recentSearches: string[]
  onClear: () => void
}

export function SearchInput({
  value,
  onChange,
  placeholder,
  aiEnabled,
  onAiToggle,
  isLoading,
  recentSearches,
  onClear
}: SearchInputProps) {
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-complete suggestions
  useEffect(() => {
    if (value.length > 2) {
      // This would typically call an API for suggestions
      const mockSuggestions = [
        'prompt engineering best practices',
        'SQL query optimization',
        'machine learning fundamentals',
        'data modeling techniques',
        'Snowflake performance tuning'
      ].filter(s => s.toLowerCase().includes(value.toLowerCase()))
      
      setSuggestions([...mockSuggestions, ...recentSearches.filter(s => 
        s.toLowerCase().includes(value.toLowerCase()) && s !== value
      )].slice(0, 5))
    } else {
      setSuggestions(recentSearches.slice(0, 5))
    }
  }, [value, recentSearches])

  const handleInputFocus = () => {
    setShowSuggestions(true)
  }

  const handleInputBlur = () => {
    // Delay to allow clicking on suggestions
    setTimeout(() => setShowSuggestions(false), 200)
  }

  const handleSuggestionClick = (suggestion: string) => {
    onChange(suggestion)
    setShowSuggestions(false)
    inputRef.current?.focus()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setShowSuggestions(false)
      inputRef.current?.blur()
    }
  }

  return (
    <div className="relative">
      <div className="relative">
        {/* Search Input */}
        <div className="relative">
          <Icons.Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          
          <input
            ref={inputRef}
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onFocus={handleInputFocus}
            onBlur={handleInputBlur}
            onKeyDown={handleKeyDown}
            placeholder={aiEnabled ? `${placeholder} (AI Enhanced)` : placeholder}
            className="w-full pl-12 pr-24 py-3 text-sm border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
          />

          {/* Right Side Controls */}
          <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex items-center space-x-2">
            {/* Loading Spinner */}
            {isLoading && (
              <Icons.Loader2 className="w-4 h-4 text-gray-400 animate-spin" />
            )}

            {/* Clear Button */}
            {value && !isLoading && (
              <button
                onClick={onClear}
                className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <Icons.X className="w-4 h-4" />
              </button>
            )}

            {/* AI Toggle */}
            <button
              onClick={() => onAiToggle(!aiEnabled)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                aiEnabled
                  ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
              title={aiEnabled ? 'Disable AI search' : 'Enable AI search'}
            >
              <Icons.Bot className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* AI Enhancement Indicator */}
        {aiEnabled && (
          <div className="absolute -bottom-6 left-0 flex items-center space-x-1 text-xs text-blue-600 dark:text-blue-400">
            <Icons.Bot className="w-3 h-3" />
            <span>AI-enhanced search active</span>
          </div>
        )}
      </div>

      {/* Suggestions Dropdown */}
      {showSuggestions && (suggestions.length > 0 || recentSearches.length > 0) && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50 max-h-64 overflow-y-auto">
          {/* Recent Searches */}
          {!value && recentSearches.length > 0 && (
            <div className="p-3 border-b border-gray-100 dark:border-gray-700">
              <div className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-2 flex items-center space-x-1">
                <Icons.Clock className="w-3 h-3" />
                <span>Recent Searches</span>
              </div>
              {recentSearches.slice(0, 5).map((search, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestionClick(search)}
                  className="block w-full text-left px-2 py-1 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                >
                  {search}
                </button>
              ))}
            </div>
          )}

          {/* Search Suggestions */}
          {suggestions.length > 0 && (
            <div className="p-3">
              <div className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-2">
                Suggestions
              </div>
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="block w-full text-left px-2 py-1 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                >
                  <Icons.Search className="inline w-3 h-3 mr-2 text-gray-400" />
                  {suggestion}
                </button>
              ))}
            </div>
          )}

          {/* Quick Tips */}
          {!value && (
            <div className="p-3 border-t border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-750">
              <div className="text-xs text-gray-500 dark:text-gray-400">
                <strong>Tips:</strong> Try "prompt engineering", "SQL basics", or ask questions like "How to optimize Snowflake?"
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}