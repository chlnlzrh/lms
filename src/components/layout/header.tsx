'use client'

import { useEffect } from 'react'
import { Icons } from '@/components/ui/icons'
import { cn } from '@/lib/utils'

interface HeaderProps {
  onSearchClick?: () => void
}

export function Header({ onSearchClick }: HeaderProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        if (onSearchClick) {
          onSearchClick()
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [onSearchClick])

  const handleSearchClick = () => {
    if (onSearchClick) {
      onSearchClick()
    }
  }

  return (
    <header className="fixed top-0 left-16 right-0 h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 z-40">
      <div className="flex items-center justify-between h-full px-6">
        {/* Search Bar */}
        <div className="flex-1 max-w-md">
          <button
            onClick={handleSearchClick}
            className={cn(
              "w-full flex items-center pl-10 pr-4 py-2 text-xs text-left",
              "bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md",
              "hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors",
              "text-gray-400"
            )}
          >
            <Icons.Search className="absolute left-3 w-4 h-4" />
            <span>Search lessons, modules, tracks...</span>
            <div className="ml-auto flex items-center space-x-1">
              <kbd className="px-1.5 py-0.5 text-xs bg-gray-200 dark:bg-gray-600 rounded">⌘</kbd>
              <kbd className="px-1.5 py-0.5 text-xs bg-gray-200 dark:bg-gray-600 rounded">K</kbd>
            </div>
          </button>
        </div>

        {/* User Info */}
        <div className="flex items-center space-x-4">
          <div className="text-xs text-gray-600 dark:text-gray-300">
            Welcome back, <span className="font-bold">John Doe</span>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
              <span className="text-xs font-bold text-white">JD</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}