'use client'

import { ReactNode, useState } from 'react'
import { Sidebar } from '@/components/navigation/sidebar'
import { Header } from './header'
import { GlobalSearch } from '@/components/features/global-search'
import { cn } from '@/lib/utils'

interface MainLayoutProps {
  children: ReactNode
  className?: string
}

export function MainLayout({ children, className }: MainLayoutProps) {
  const [isSearchOpen, setIsSearchOpen] = useState(false)

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Sidebar />
      <Header onSearchClick={() => setIsSearchOpen(true)} />
      
      <main className={cn(
        "ml-16 mt-16 transition-all duration-300",
        className
      )}>
        <div className="p-6">
          {children}
        </div>
      </main>

      <GlobalSearch 
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
      />
    </div>
  )
}