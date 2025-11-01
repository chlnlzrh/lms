import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { ProgressBar } from '@/components/ui/progress-bar'
import { Icons } from '@/components/ui/icons'
import { progressTracker } from '@/lib/progress-tracker'
import { Suspense } from 'react'

function ProfilePageContent() {
  const userProgress = progressTracker.getUserProgress()
  const recentLessons = progressTracker.getRecentLessons(5)
  const bookmarkedLessons = progressTracker.getBookmarkedLessons()

  const breadcrumbItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Profile' }
  ]

  // Calculate stats
  const totalLessonsCompleted = Object.values(userProgress.lessons).filter(l => l.isCompleted).length
  const totalTimeSpent = Math.round(userProgress.totalTimeSpent / 60 * 10) / 10 // Convert to hours
  const currentStreaks = Object.values(userProgress.tracks).map(t => t.currentStreak)
  const maxCurrentStreak = Math.max(...currentStreaks, 0)

  // Achievement definitions
  const achievements = [
    { id: 'first-lesson', name: 'First Steps', description: 'Completed your first lesson', icon: '🌟' },
    { id: '10-lessons', name: 'Getting Started', description: 'Completed 10 lessons', icon: '📚' },
    { id: '50-lessons', name: 'Learning Momentum', description: 'Completed 50 lessons', icon: '🚀' },
    { id: '100-lessons', name: 'Knowledge Builder', description: 'Completed 100 lessons', icon: '🏗️' },
    { id: 'week-streak', name: 'Consistent Learner', description: '7-day learning streak', icon: '🔥' },
    { id: 'month-streak', name: 'Dedicated Student', description: '30-day learning streak', icon: '💎' },
    { id: '10-hours', name: 'Time Investment', description: '10 hours of learning', icon: '⏰' },
    { id: '50-hours', name: 'Expert Path', description: '50 hours of learning', icon: '🎓' }
  ]

  const unlockedAchievements = achievements.filter(a => userProgress.achievements.includes(a.id))
  const nextAchievements = achievements.filter(a => !userProgress.achievements.includes(a.id)).slice(0, 3)

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Breadcrumbs */}
        <Breadcrumbs items={breadcrumbItems} />

        {/* Profile Header */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-start space-x-4">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <span className="text-xl font-bold text-white">JD</span>
            </div>
            
            <div className="flex-1">
              <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-1">
                John Doe
              </h1>
              <p className="text-xs text-gray-600 dark:text-gray-300 mb-4">
                Data Engineer • Joined {userProgress.joinedAt.toLocaleDateString()}
              </p>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                  <div className="text-xl font-bold text-blue-600 mb-1">{totalLessonsCompleted}</div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Lessons Completed</div>
                </div>
                
                <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded">
                  <div className="text-xl font-bold text-green-600 mb-1">{totalTimeSpent}h</div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Time Learned</div>
                </div>
                
                <div className="text-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                  <div className="text-xl font-bold text-purple-600 mb-1">{maxCurrentStreak}</div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Current Streak</div>
                </div>
                
                <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                  <div className="text-xl font-bold text-orange-600 mb-1">{unlockedAchievements.length}</div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Achievements</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Track Progress */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
              Track Progress
            </h2>
            
            <div className="space-y-4">
              {Object.values(userProgress.tracks).map((track) => (
                <div key={track.trackId} className="border border-gray-100 dark:border-gray-700 rounded p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    {track.trackId === 'ai' ? (
                      <Icons.Bot className="w-4 h-4 text-blue-600" />
                    ) : (
                      <Icons.Database className="w-4 h-4 text-green-600" />
                    )}
                    <span className="text-xs font-bold text-gray-900 dark:text-white">
                      {track.trackId === 'ai' ? 'AI Training' : 'Data Engineering'}
                    </span>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600 dark:text-gray-300">Progress</span>
                      <span className="text-gray-900 dark:text-white">
                        {track.completedLessons} / {track.totalLessons} lessons
                      </span>
                    </div>
                    <ProgressBar 
                      value={track.completedLessons} 
                      max={track.totalLessons || 100}
                      size="sm" 
                    />
                    
                    <div className="grid grid-cols-3 gap-2 text-xs text-center">
                      <div>
                        <div className="font-bold text-gray-900 dark:text-white">{track.currentStreak}</div>
                        <div className="text-gray-500">Streak</div>
                      </div>
                      <div>
                        <div className="font-bold text-gray-900 dark:text-white">{Math.round(track.timeSpent / 60 * 10) / 10}h</div>
                        <div className="text-gray-500">Time</div>
                      </div>
                      <div>
                        <div className="font-bold text-gray-900 dark:text-white">{track.completedModules}</div>
                        <div className="text-gray-500">Modules</div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Achievements */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
              Achievements
            </h2>
            
            {unlockedAchievements.length > 0 ? (
              <div className="space-y-3 mb-6">
                <h3 className="text-xs font-bold text-green-600 mb-2">Unlocked ({unlockedAchievements.length})</h3>
                {unlockedAchievements.map((achievement) => (
                  <div key={achievement.id} className="flex items-center space-x-3 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded">
                    <span className="text-2xl">{achievement.icon}</span>
                    <div>
                      <div className="text-xs font-bold text-gray-900 dark:text-white">{achievement.name}</div>
                      <div className="text-xs text-gray-600 dark:text-gray-300">{achievement.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
            
            {nextAchievements.length > 0 && (
              <div className="space-y-3">
                <h3 className="text-xs font-bold text-gray-600 mb-2">Next Goals</h3>
                {nextAchievements.map((achievement) => (
                  <div key={achievement.id} className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded opacity-75">
                    <span className="text-2xl grayscale">{achievement.icon}</span>
                    <div>
                      <div className="text-xs font-bold text-gray-900 dark:text-white">{achievement.name}</div>
                      <div className="text-xs text-gray-600 dark:text-gray-300">{achievement.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Recent Activity */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
              Recent Activity
            </h2>
            
            {recentLessons.length > 0 ? (
              <div className="space-y-3">
                {recentLessons.map((lesson) => (
                  <div key={lesson.lessonId} className="flex items-center space-x-3 p-3 border border-gray-100 dark:border-gray-700 rounded">
                    <div className={`w-2 h-2 rounded-full ${lesson.isCompleted ? 'bg-green-500' : 'bg-blue-500'}`} />
                    <div className="flex-1">
                      <div className="text-xs font-normal text-gray-900 dark:text-white">
                        {lesson.lessonId.split('-').slice(1).join(' ').replace(/([A-Z])/g, ' $1')}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {lesson.lastAccessedAt.toLocaleDateString()} • {lesson.timeSpent}min
                      </div>
                    </div>
                    <span className="text-xs text-gray-500">
                      {lesson.isCompleted ? '✓' : '→'}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Icons.BookOpen className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  No recent activity. Start learning to track your progress!
                </p>
              </div>
            )}
          </div>

          {/* Bookmarks */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
              Bookmarked Lessons ({bookmarkedLessons.length})
            </h2>
            
            {bookmarkedLessons.length > 0 ? (
              <div className="space-y-3">
                {bookmarkedLessons.slice(0, 5).map((lesson) => (
                  <div key={lesson.lessonId} className="flex items-center space-x-3 p-3 border border-gray-100 dark:border-gray-700 rounded">
                    <Icons.Bookmark className="w-3 h-3 text-yellow-600 fill-current" />
                    <div className="flex-1">
                      <div className="text-xs font-normal text-gray-900 dark:text-white">
                        {lesson.lessonId.split('-').slice(1).join(' ').replace(/([A-Z])/g, ' $1')}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {lesson.isCompleted ? 'Completed' : 'In Progress'}
                      </div>
                    </div>
                    {lesson.isCompleted && (
                      <Icons.Circle className="w-3 h-3 text-green-600 fill-current" />
                    )}
                  </div>
                ))}
                
                {bookmarkedLessons.length > 5 && (
                  <div className="text-center pt-2">
                    <span className="text-xs text-gray-500">
                      +{bookmarkedLessons.length - 5} more bookmarks
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8">
                <Icons.Bookmark className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  No bookmarked lessons yet. Bookmark lessons for quick access.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Learning Preferences */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Learning Preferences
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-3">Export Progress</h3>
              <p className="text-xs text-gray-600 dark:text-gray-300 mb-3">
                Download your learning progress and achievements.
              </p>
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
                className="bg-blue-600 text-white px-4 py-2 rounded text-xs font-normal hover:bg-blue-700"
              >
                Export Data
              </button>
            </div>
            
            <div>
              <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-3">Reset Progress</h3>
              <p className="text-xs text-gray-600 dark:text-gray-300 mb-3">
                Clear all progress data and start fresh.
              </p>
              <button 
                onClick={() => {
                  if (confirm('Are you sure you want to reset all progress? This cannot be undone.')) {
                    progressTracker.resetProgress()
                    window.location.reload()
                  }
                }}
                className="bg-red-600 text-white px-4 py-2 rounded text-xs font-normal hover:bg-red-700"
              >
                Reset All Data
              </button>
            </div>
          </div>
        </div>
      </div>
    </MainLayout>
  )
}

function LoadingFallback() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="h-32 bg-gray-200 rounded mb-6"></div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-64 bg-gray-200 rounded"></div>
            <div className="h-64 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    </MainLayout>
  )
}

export default function ProfilePage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <ProfilePageContent />
    </Suspense>
  )
}