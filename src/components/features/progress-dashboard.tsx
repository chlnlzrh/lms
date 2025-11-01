'use client'

import { useState, useEffect } from 'react'
import { progressTracker, UserProgress } from '@/lib/progress-tracker'
import { Icons } from '@/components/ui/icons'
import { ProgressBar } from '@/components/ui/progress-bar'
import { cn } from '@/lib/utils'

interface ProgressDashboardProps {
  className?: string
}

export function ProgressDashboard({ className }: ProgressDashboardProps) {
  const [userProgress, setUserProgress] = useState<UserProgress | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const loadProgress = () => {
      try {
        const progress = progressTracker.getUserProgress()
        setUserProgress(progress)
      } catch (error) {
        console.error('Error loading progress:', error)
      } finally {
        setIsLoading(false)
      }
    }

    loadProgress()

    // Refresh progress every minute to update time spent
    const interval = setInterval(loadProgress, 60000)
    return () => clearInterval(interval)
  }, [])

  if (isLoading) {
    return (
      <div className={cn("space-y-4", className)}>
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-24 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (!userProgress) {
    return (
      <div className={cn("text-center py-8", className)}>
        <Icons.AlertCircle className="w-8 h-8 text-gray-400 mx-auto mb-2" />
        <p className="text-xs text-gray-500">Unable to load progress data</p>
      </div>
    )
  }

  const totalLessonsCompleted = Object.values(userProgress.lessons).filter(l => l.isCompleted).length
  const totalTimeSpent = Math.round(userProgress.totalTimeSpent / 60 * 10) / 10 // Convert to hours
  const currentStreaks = Object.values(userProgress.tracks).map(t => t.currentStreak)
  const maxCurrentStreak = Math.max(...currentStreaks, 0)
  const totalAchievements = userProgress.achievements.length

  const stats = [
    {
      icon: Icons.BookOpen,
      label: 'Lessons Completed',
      value: totalLessonsCompleted,
      color: 'blue'
    },
    {
      icon: Icons.Clock,
      label: 'Time Learned',
      value: `${totalTimeSpent}h`,
      color: 'green'
    },
    {
      icon: Icons.Flame,
      label: 'Current Streak',
      value: maxCurrentStreak,
      color: 'orange'
    },
    {
      icon: Icons.Trophy,
      label: 'Achievements',
      value: totalAchievements,
      color: 'purple'
    }
  ]

  return (
    <div className={cn("space-y-6", className)}>
      {/* Quick Stats */}
      <div>
        <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
          Learning Progress
        </h2>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {stats.map((stat) => {
            const IconComponent = stat.icon
            return (
              <div
                key={stat.label}
                className={cn(
                  "p-4 rounded-lg border",
                  `bg-${stat.color}-50 dark:bg-${stat.color}-900/20`,
                  `border-${stat.color}-200 dark:border-${stat.color}-800`
                )}
              >
                <div className="flex items-center space-x-3">
                  <IconComponent className={`w-5 h-5 text-${stat.color}-600`} />
                  <div>
                    <div className={`text-xl font-bold text-${stat.color}-600 mb-1`}>
                      {stat.value}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-300">
                      {stat.label}
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Track Progress */}
      {Object.keys(userProgress.tracks).length > 0 && (
        <div>
          <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Track Progress
          </h3>
          
          <div className="space-y-4">
            {Object.values(userProgress.tracks).map((track) => (
              <div
                key={track.trackId}
                className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
              >
                <div className="flex items-center space-x-2 mb-3">
                  {track.trackId === 'ai' ? (
                    <Icons.Bot className="w-4 h-4 text-blue-600" />
                  ) : (
                    <Icons.Database className="w-4 h-4 text-green-600" />
                  )}
                  <span className="text-xs font-bold text-gray-900 dark:text-white">
                    {track.trackId === 'ai' ? 'AI Training' : 'Data Engineering'}
                  </span>
                </div>
                
                <div className="space-y-3">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-600 dark:text-gray-300">Progress</span>
                    <span className="text-gray-900 dark:text-white">
                      {track.completedLessons} / {track.totalLessons || '?'} lessons
                    </span>
                  </div>
                  
                  <ProgressBar
                    value={track.completedLessons}
                    max={track.totalLessons || 100}
                    size="sm"
                    className="mb-3"
                  />
                  
                  <div className="grid grid-cols-3 gap-2 text-xs text-center">
                    <div>
                      <div className="font-bold text-gray-900 dark:text-white">
                        {track.currentStreak}
                      </div>
                      <div className="text-gray-500">Streak</div>
                    </div>
                    <div>
                      <div className="font-bold text-gray-900 dark:text-white">
                        {Math.round(track.timeSpent / 60 * 10) / 10}h
                      </div>
                      <div className="text-gray-500">Time</div>
                    </div>
                    <div>
                      <div className="font-bold text-gray-900 dark:text-white">
                        {track.completedModules}
                      </div>
                      <div className="text-gray-500">Modules</div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Activity */}
      {Object.keys(userProgress.lessons).length > 0 && (
        <div>
          <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Recent Activity
          </h3>
          
          <div className="space-y-2">
            {progressTracker.getRecentLessons(5).map((lesson) => (
              <div
                key={lesson.lessonId}
                className="flex items-center space-x-3 p-3 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700"
              >
                <div className={cn(
                  "w-2 h-2 rounded-full",
                  lesson.isCompleted ? "bg-green-500" : "bg-blue-500"
                )} />
                
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-normal text-gray-900 dark:text-white truncate">
                    {lesson.lessonId.split('-').slice(1).join(' ').replace(/([A-Z])/g, ' $1')}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {lesson.lastAccessedAt.toLocaleDateString()} • {lesson.timeSpent}min
                  </div>
                </div>
                
                <div className="flex items-center space-x-1">
                  {lesson.bookmarked && (
                    <Icons.Bookmark className="w-3 h-3 text-yellow-600 fill-current" />
                  )}
                  {lesson.isCompleted && (
                    <Icons.Circle className="w-3 h-3 text-green-600 fill-current" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Progress Export/Reset Actions */}
      <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={() => {
            const data = progressTracker.exportProgress()
            const blob = new Blob([data], { type: 'application/json' })
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `lms-progress-${new Date().toISOString().split('T')[0]}.json`
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            URL.revokeObjectURL(url)
          }}
          className="text-xs text-blue-600 hover:text-blue-700 transition-colors"
        >
          Export Progress
        </button>
        
        <button
          onClick={() => {
            if (confirm('Are you sure you want to reset all progress? This cannot be undone.')) {
              progressTracker.resetProgress()
              setUserProgress(progressTracker.getUserProgress())
            }
          }}
          className="text-xs text-red-600 hover:text-red-700 transition-colors"
        >
          Reset Progress
        </button>
      </div>
    </div>
  )
}