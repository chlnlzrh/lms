'use client'

import Link from 'next/link'
import { TrackInfo } from '@/types/content'
import { Icons } from '@/components/ui/icons'
import { cn } from '@/lib/utils'

interface TrackOverviewProps {
  track: TrackInfo
  trackIcon?: string
  trackColor?: string
  className?: string
  basePath?: string
}

export function TrackOverview({ track, trackIcon, trackColor, className, basePath = 'talent-development' }: TrackOverviewProps) {
  // Use provided props or fallback to defaults
  const icon = trackIcon || 'BookOpen'
  const color = trackColor || 'blue'
  
  // Map icon names to actual icon components
  const iconMap: { [key: string]: any } = {
    Bot: Icons.Bot,
    Database: Icons.Database,
    Cloud: Icons.Cloud,
    Zap: Icons.Zap,
    Snowflake: Icons.Snowflake,
    Link: Icons.Link,
    BarChart3: Icons.BarChart3,
    Shield: Icons.Shield,
    Settings: Icons.Settings,
    DollarSign: Icons.DollarSign,
    Users: Icons.Users,
    Archive: Icons.Archive,
    Briefcase: Icons.Briefcase,
    CheckCircle: Icons.CheckCircle,
    TrendingUp: Icons.TrendingUp,
    FileSearch: Icons.FileSearch,
    PieChart: Icons.PieChart,
    BookOpen: Icons.BookOpen,
    Megaphone: Icons.Megaphone
  }

  const IconComponent = iconMap[icon] || Icons.BookOpen

  return (
    <div className={cn(
      "bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6",
      className
    )}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <IconComponent className={`w-6 h-6 text-${color}-600`} />
          <div>
            <h2 className="text-xs font-bold text-gray-900 dark:text-white">
              {track.title}
            </h2>
            <p className="text-xs text-gray-600 dark:text-gray-300 mt-1">
              {track.description}
            </p>
          </div>
        </div>
        
        <Link 
          href={`/${basePath}/${track.id}`}
          className={`bg-${color}-600 text-white px-4 py-2 rounded text-xs font-normal hover:bg-${color}-700 transition-colors`}
        >
          View Track
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="text-center">
          <div className={`text-xl font-bold text-${color}-600 mb-1`}>
            {track.totalModules}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-300">Modules</div>
        </div>
        
        <div className="text-center">
          <div className={`text-xl font-bold text-${color}-600 mb-1`}>
            {track.totalLessons}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-300">Lessons</div>
        </div>
        
        <div className="text-center">
          <div className={`text-xl font-bold text-${color}-600 mb-1`}>
            {track.estimatedDuration}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-300">Duration</div>
        </div>
        
        <div className="text-center">
          <div className={`text-xl font-bold text-${color}-600 mb-1`}>
            Active
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-300">Status</div>
        </div>
      </div>


      {/* Module Preview */}
      <div className="mt-6">
        <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-3">
          Modules ({track.modules.length})
        </h3>
        <div className="space-y-2">
          {track.modules.slice(0, 3).map((module) => (
            <Link
              key={module.id}
              href={`/${basePath}/${track.id}/module-${module.moduleNumber}`}
              className="block p-3 rounded border border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="text-xs font-normal text-gray-900 dark:text-white">
                    {module.title}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300 mt-1">
                    {module.lessonCount} lessons • {module.duration}
                  </div>
                </div>
                <Icons.ChevronRight className="w-3 h-3 text-gray-400" />
              </div>
            </Link>
          ))}
          
          {track.modules.length > 3 && (
            <Link
              href={`/${basePath}/${track.id}`}
              className="block p-3 rounded border border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-center"
            >
              <span className={`text-xs text-${color}-600 font-normal`}>
                View all {track.modules.length} modules
              </span>
            </Link>
          )}
        </div>
      </div>
    </div>
  )
}