'use client'

import { useState } from 'react'
import Link from 'next/link'
import { MenuItem } from '@/types/navigation'
import { Icons, IconName } from '@/components/ui/icons'
import { cn } from '@/lib/utils'
import * as Collapsible from '@radix-ui/react-collapsible'

interface SidebarMenuItemProps {
  item: MenuItem
  level?: number
  isExpanded?: boolean
  onToggle?: (id: string) => void
  currentPath?: string
  sidebarCollapsed?: boolean
}

export function SidebarMenuItem({
  item,
  level = 0,
  isExpanded = false,
  onToggle,
  currentPath = '',
  sidebarCollapsed = false
}: SidebarMenuItemProps) {
  const IconComponent = Icons[item.icon as IconName] || Icons.Circle
  const hasChildren = item.children && item.children.length > 0
  const isActive = item.href === currentPath
  const isComingSoon = item.status === 'coming-soon'

  const handleClick = () => {
    if (hasChildren && onToggle) {
      onToggle(item.id)
    }
  }

  const MenuContent = () => (
    <div className={cn(
      "flex items-center w-full",
      level === 0 ? "py-1.5" : "py-1",
      level > 0 && "ml-4"
    )}>
      <IconComponent 
        className={cn(
          "flex-shrink-0",
          level === 0 ? "w-4 h-4" : "w-3 h-3",
          "mr-3"
        )} 
      />
      
      {!sidebarCollapsed && (
        <>
          <span className={cn(
            "text-xs font-normal flex-1 truncate",
            level === 0 && item.children && "font-bold"
          )}>
            {item.label}
          </span>
          
          {item.lessonCount && (
            <span className="text-xs text-gray-400 ml-2">
              ({item.lessonCount})
            </span>
          )}
          
          {isComingSoon && (
            <span className="text-xs bg-orange-100 text-orange-600 px-2 py-0.5 rounded ml-2">
              Coming Soon
            </span>
          )}
          
          {hasChildren && (
            <Icons.ChevronRight 
              className={cn(
                "w-3 h-3 ml-2 transition-transform",
                isExpanded && "rotate-90"
              )}
            />
          )}
        </>
      )}
    </div>
  )

  const itemClasses = cn(
    "block w-full text-left transition-colors duration-200",
    isActive 
      ? "text-black dark:text-white" 
      : isComingSoon 
        ? "text-gray-400 cursor-not-allowed"
        : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300",
    !isComingSoon && "cursor-pointer"
  )

  if (hasChildren) {
    return (
      <Collapsible.Root open={isExpanded}>
        <div className="space-y-0.5">
          <div className="flex items-center">
            {/* Main link - navigates to the page */}
            {item.href && !isComingSoon ? (
              <Link href={item.href} className={cn(itemClasses, "flex-1")}>
                <div className={cn(
                  "flex items-center",
                  level === 0 ? "py-1.5" : "py-1",
                  level > 0 && "ml-4"
                )}>
                  <IconComponent 
                    className={cn(
                      "flex-shrink-0",
                      level === 0 ? "w-4 h-4" : "w-3 h-3",
                      "mr-3"
                    )} 
                  />
                  
                  {!sidebarCollapsed && (
                    <>
                      <span className={cn(
                        "text-xs font-normal flex-1 truncate",
                        level === 0 && item.children && "font-bold"
                      )}>
                        {item.label}
                      </span>
                      
                      {item.lessonCount && (
                        <span className="text-xs text-gray-400 ml-2">
                          ({item.lessonCount})
                        </span>
                      )}
                      
                      {isComingSoon && (
                        <span className="text-xs bg-orange-100 text-orange-600 px-2 py-0.5 rounded ml-2">
                          Coming Soon
                        </span>
                      )}
                    </>
                  )}
                </div>
              </Link>
            ) : (
              <div className={cn(itemClasses, "flex-1")}>
                <div className={cn(
                  "flex items-center",
                  level === 0 ? "py-1.5" : "py-1",
                  level > 0 && "ml-4"
                )}>
                  <IconComponent 
                    className={cn(
                      "flex-shrink-0",
                      level === 0 ? "w-4 h-4" : "w-3 h-3",
                      "mr-3"
                    )} 
                  />
                  
                  {!sidebarCollapsed && (
                    <>
                      <span className={cn(
                        "text-xs font-normal flex-1 truncate",
                        level === 0 && item.children && "font-bold"
                      )}>
                        {item.label}
                      </span>
                      
                      {item.lessonCount && (
                        <span className="text-xs text-gray-400 ml-2">
                          ({item.lessonCount})
                        </span>
                      )}
                      
                      {isComingSoon && (
                        <span className="text-xs bg-orange-100 text-orange-600 px-2 py-0.5 rounded ml-2">
                          Coming Soon
                        </span>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}
            
            {/* Expand/collapse button */}
            {!sidebarCollapsed && (
              <Collapsible.Trigger asChild>
                <button 
                  className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded ml-1"
                  onClick={handleClick}
                  disabled={isComingSoon}
                  title={isExpanded ? "Collapse" : "Expand"}
                >
                  <Icons.ChevronRight 
                    className={cn(
                      "w-3 h-3 transition-transform text-gray-400 hover:text-gray-600",
                      isExpanded && "rotate-90"
                    )}
                  />
                </button>
              </Collapsible.Trigger>
            )}
          </div>
          
          <Collapsible.Content className="overflow-hidden">
            <div className="space-y-0.5">
              {item.children?.map((child) => (
                <SidebarMenuItem
                  key={child.id}
                  item={child}
                  level={level + 1}
                  currentPath={currentPath}
                  sidebarCollapsed={sidebarCollapsed}
                  onToggle={onToggle}
                />
              ))}
            </div>
          </Collapsible.Content>
        </div>
      </Collapsible.Root>
    )
  }

  if (item.href && !isComingSoon) {
    return (
      <Link href={item.href} className={itemClasses}>
        <MenuContent />
      </Link>
    )
  }

  return (
    <button 
      className={itemClasses}
      disabled={isComingSoon}
      title={isComingSoon ? `${item.label} - Coming Soon` : item.label}
    >
      <MenuContent />
    </button>
  )
}