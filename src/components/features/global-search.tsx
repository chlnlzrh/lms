'use client'

import { useState, useEffect } from 'react'
import { SearchInterface } from '@/components/content/search-interface'
import { SearchIndexEntry } from '@/types/content'
import { Icons } from '@/components/ui/icons'
import { cn } from '@/lib/utils'

interface GlobalSearchProps {
  className?: string
  onClose?: () => void
  isOpen?: boolean
}

export function GlobalSearch({ className, onClose, isOpen = false }: GlobalSearchProps) {
  const [results, setResults] = useState<SearchIndexEntry[]>([])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        if (onClose && isOpen) {
          onClose()
        }
      }
      
      if (e.key === 'Escape' && isOpen && onClose) {
        onClose()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center p-4 pt-[10vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Search Modal */}
      <div className={cn(
        "relative w-full max-w-2xl bg-white dark:bg-gray-800 rounded-lg shadow-2xl border border-gray-200 dark:border-gray-700",
        className
      )}>
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xs font-bold text-gray-900 dark:text-white">
              Search Learning Content
            </h2>
            <button
              onClick={onClose}
              className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <Icons.X className="w-4 h-4" />
            </button>
          </div>
          
          <SearchInterface
            placeholder="Search across 778+ lessons and modules..."
            showFilters={true}
            onResultSelect={(result) => {
              // Navigate to the result
              const url = result.type === 'lesson' 
                ? `/talent-development/${result.track}/lesson/${result.id.split('-').slice(1).join('-')}`
                : `/talent-development/${result.track}/module-${result.id.split('-').slice(-1)[0]}`
              
              window.location.href = url
              if (onClose) onClose()
            }}
          />
        </div>
        
        <div className="p-4 text-xs text-gray-500 dark:text-gray-400 border-t border-gray-100 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <span>Search across AI Training and Data Engineering tracks</span>
            <div className="flex items-center space-x-2">
              <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">↑↓</kbd>
              <span>navigate</span>
              <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">↵</kbd>
              <span>select</span>
              <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">esc</kbd>
              <span>close</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}