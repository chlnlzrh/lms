'use client'

import Link from 'next/link'
import { ModuleDescription } from '@/types/content'
import { Icons } from '@/components/ui/icons'
import { ProgressBar } from '@/components/ui/progress-bar'
import { cn } from '@/lib/utils'

interface ModuleCardProps {
  module: ModuleDescription
  progress?: {
    completedLessons: number
    isCompleted: boolean
  }
  className?: string
}

export function ModuleCard({ module, progress, className }: ModuleCardProps) {
  const trackColors = {
    ai: 'blue',
    'data-engineering': 'green',
    saas: 'purple'
  }

  const color = trackColors[module.track]
  const completionPercentage = progress 
    ? Math.round((progress.completedLessons / module.lessonCount) * 100)
    : 0

  return (
    <Link
      href={`/talent-development/${module.track}/module-${module.moduleNumber}`}
      className={cn(
        "block bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-all",
        progress?.isCompleted && "ring-2 ring-green-200 dark:ring-green-800",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <span className={`text-xs px-2 py-1 rounded bg-${color}-100 text-${color}-700 dark:bg-${color}-900 dark:text-${color}-300`}>
              Module {module.moduleNumber}
            </span>
            {progress?.isCompleted && (
              <Icons.Circle className="w-3 h-3 text-green-600 fill-current" />
            )}
          </div>
          
          <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-2">
            {module.title}
          </h3>
          
          <p className="text-xs text-gray-600 dark:text-gray-300 line-clamp-2">
            {module.description}
          </p>
        </div>
        
        <Icons.ChevronRight className="w-4 h-4 text-gray-400 mt-1 flex-shrink-0" />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <div className={`text-lg font-bold text-${color}-600 mb-1`}>
            {module.lessonCount}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-300">Lessons</div>
        </div>
        
        <div className="text-center">
          <div className={`text-lg font-bold text-${color}-600 mb-1`}>
            {module.duration.replace(' hours', 'h')}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-300">Duration</div>
        </div>
        
        <div className="text-center">
          <div className={`text-lg font-bold text-${color}-600 mb-1`}>
            {module.labCount}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-300">Labs</div>
        </div>
      </div>

      {/* Progress */}
      {progress && (
        <div className="space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-gray-600 dark:text-gray-300">Progress</span>
            <span className="text-gray-900 dark:text-white">
              {progress.completedLessons} / {module.lessonCount}
            </span>
          </div>
          <ProgressBar 
            value={progress.completedLessons} 
            max={module.lessonCount}
            size="sm"
          />
        </div>
      )}

      {/* Prerequisites */}
      {module.prerequisites.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
          <div className="text-xs text-gray-600 dark:text-gray-300 mb-1">
            Prerequisites:
          </div>
          <div className="text-xs text-gray-900 dark:text-white">
            {module.prerequisites.join(', ')}
          </div>
        </div>
      )}

      {/* Learning Objectives Preview */}
      {module.learningObjectives.length > 0 && (
        <div className="mt-3">
          <div className="text-xs text-gray-600 dark:text-gray-300 mb-1">
            You'll learn:
          </div>
          <ul className="text-xs text-gray-900 dark:text-white space-y-1">
            {module.learningObjectives.slice(0, 2).map((objective, index) => (
              <li key={index} className="flex items-start space-x-2">
                <span className="text-gray-400 mt-0.5">•</span>
                <span className="line-clamp-1">{objective}</span>
              </li>
            ))}
            {module.learningObjectives.length > 2 && (
              <li className="text-gray-500">
                +{module.learningObjectives.length - 2} more objectives
              </li>
            )}
          </ul>
        </div>
      )}
    </Link>
  )
}