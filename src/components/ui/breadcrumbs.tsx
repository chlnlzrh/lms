'use client'

import Link from 'next/link'
import { Icons } from './icons'
import { cn } from '@/lib/utils'

interface BreadcrumbItem {
  label: string
  href?: string
}

interface BreadcrumbsProps {
  items: BreadcrumbItem[]
  className?: string
}

export function Breadcrumbs({ items, className }: BreadcrumbsProps) {
  if (items.length === 0) return null

  return (
    <nav className={cn("flex items-center space-x-2 text-xs", className)}>
      {items.map((item, index) => {
        const isLast = index === items.length - 1
        
        return (
          <div key={index} className="flex items-center space-x-2">
            {item.href && !isLast ? (
              <Link 
                href={item.href}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                {item.label}
              </Link>
            ) : (
              <span className={cn(
                isLast 
                  ? "text-gray-900 dark:text-white font-normal" 
                  : "text-gray-500 dark:text-gray-400"
              )}>
                {item.label}
              </span>
            )}
            
            {!isLast && (
              <Icons.ChevronRight className="w-3 h-3 text-gray-400" />
            )}
          </div>
        )
      })}
    </nav>
  )
}