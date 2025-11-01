'use client'

import { useState, useEffect } from 'react'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import { navigationData } from '@/data/navigation'
import { SidebarMenuItem } from './sidebar-menu-item'
import { Icons } from '@/components/ui/icons'

export function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(true)
  const [expandedSections, setExpandedSections] = useState<string[]>([])
  const pathname = usePathname()

  // Load expanded sections from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('sidebar-expanded-sections')
    if (saved) {
      setExpandedSections(JSON.parse(saved))
    }
  }, [])

  // Save expanded sections to localStorage
  useEffect(() => {
    localStorage.setItem('sidebar-expanded-sections', JSON.stringify(expandedSections))
  }, [expandedSections])

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => 
      prev.includes(sectionId)
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    )
  }

  const handleMouseEnter = () => {
    if (isCollapsed) {
      setIsCollapsed(false)
    }
  }

  const handleMouseLeave = () => {
    setIsCollapsed(true)
  }

  return (
    <aside 
      className={cn(
        "fixed left-0 top-0 h-full bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 transition-all duration-300 z-50",
        isCollapsed ? "w-16" : "w-80"
      )}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className={cn(
          "flex items-center border-b border-gray-200 dark:border-gray-700",
          isCollapsed ? "px-4 py-4 justify-center" : "px-6 py-4"
        )}>
          {isCollapsed ? (
            <Icons.Menu className="w-5 h-5 text-gray-600" />
          ) : (
            <h1 className="text-xs font-bold text-gray-900 dark:text-white">
              Learning Management System
            </h1>
          )}
        </div>

        {/* Navigation Menu */}
        <nav className="flex-1 overflow-y-auto py-4">
          <div className={cn(
            "space-y-0.5",
            isCollapsed ? "px-2" : "px-4"
          )}>
            {navigationData.map((item) => (
              <SidebarMenuItem
                key={item.id}
                item={item}
                isExpanded={expandedSections.includes(item.id)}
                onToggle={toggleSection}
                currentPath={pathname}
                sidebarCollapsed={isCollapsed}
              />
            ))}
          </div>
        </nav>

        {/* Footer */}
        <div className={cn(
          "border-t border-gray-200 dark:border-gray-700 p-4",
          isCollapsed && "px-2"
        )}>
          {!isCollapsed && (
            <div className="text-xs text-gray-500">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Active Features</span>
              </div>
              <div className="flex items-center space-x-2 mt-1">
                <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                <span>Coming Soon</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </aside>
  )
}